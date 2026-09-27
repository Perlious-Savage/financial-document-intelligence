"""Review-priority scoring and routing.

This is deliberately *not* a "risk model". CORD is a receipt corpus; there is no
meaningful fraud or AML target in it, and inventing one would produce a number that
measures nothing. What is genuinely predictable here is whether an extraction is
likely to be wrong and therefore worth a human's time.

The training label is real rather than synthetic: for each document, is extraction
quality below a usable bar (document-level entity F1 < 0.8 against the CORD ground
truth)? See train_review.py. The model predicts that from confidence and validation
findings, which makes this ordinary selective prediction.

Known limitation: the label and features were built from the keyword baseline's
predictions, which carry no confidence scores. The two confidence features are
therefore 1.0 whenever any monetary field was extracted and 0.0 otherwise: during
training they act as a presence flag, not as a confidence.

Before any model is trained the API still works, using the deterministic rule below.
Which path produced a decision is always reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .checks import FEATURE_ORDER, findings_to_features
from .schemas import ExtractedReceipt, Finding, ReviewDecision, Severity

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS / "review_model.joblib"
THRESHOLD_PATH = ARTIFACTS / "review_threshold.json"

DEFAULT_THRESHOLD = 0.5


def rule_based_priority(features: dict[str, float]) -> float:
    """Deterministic fallback, used when no trained model is present.

    Not a learned score and not presented as one. It is a transparent weighting of the
    same signals the model uses, so the service degrades to something explainable
    rather than to nothing.
    """
    score = 0.0
    score += 0.40 * (1.0 - features.get("mean_confidence", 0.0))
    score += 0.20 * min(features.get("missing_field_count", 0.0) / 3.0, 1.0)
    score += 0.25 * min(features.get("failed_check_count", 0.0) / 3.0, 1.0)
    score += 0.15 * min(features.get("error_severity_count", 0.0), 1.0)
    return round(min(max(score, 0.0), 1.0), 4)


class ReviewScorer:
    """Scores documents for human review, preferring a trained model when available."""

    def __init__(self) -> None:
        self.model = None
        self.threshold = DEFAULT_THRESHOLD
        self.source = "rules"
        self._load()

    def _load(self) -> None:
        """Load the trained scorer, but only if it earned its place.

        A model that could not satisfy its error budget during training has failed its
        acceptance criterion, and a failed model does not get deployed. In that case
        the deterministic triage runs instead: it still routes on hard rules - failed
        checks, missing fields, low confidence - and it makes no error-budget claim,
        which is honest about what it can offer.
        """
        if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
            return
        try:
            import joblib

            selection = json.loads(THRESHOLD_PATH.read_text())
            if not selection.get("budget_met", False):
                self.source = "rules (trained model did not meet its error budget)"
                return

            self.model = joblib.load(MODEL_PATH)
            self.threshold = selection["threshold"]
            self.source = "model"
        except Exception:
            # A missing or unreadable model must not take the service down.
            self.model = None
            self.source = "rules"

    def score(self, features: dict[str, float]) -> float:
        if self.model is None:
            return rule_based_priority(features)
        import numpy as np

        vector = np.array([[features.get(name, 0.0) for name in FEATURE_ORDER]])
        return float(self.model.predict_proba(vector)[0][1])

    def decide(self, receipt: ExtractedReceipt, findings: list[Finding]) -> ReviewDecision:
        features = findings_to_features(receipt, findings)
        priority = self.score(features)

        reasons: list[str] = []
        # Hard rules first. Some conditions route to a human regardless of what any
        # model says, which is how review actually works in a compliance setting.
        for finding in findings:
            if not finding.passed and finding.severity == Severity.ERROR:
                reasons.append(f"failed check: {finding.check_id}")
        if receipt.missing_fields():
            reasons.append(f"missing fields: {', '.join(receipt.missing_fields())}")
        if receipt.min_confidence < 0.70 and receipt.field_confidences:
            reasons.append(f"low field confidence ({receipt.min_confidence:.2f})")

        review_required = bool(reasons) or priority >= self.threshold
        if not reasons and review_required:
            reasons.append(f"review priority {priority:.2f} at or above threshold")

        return ReviewDecision(
            review_required=review_required,
            priority=priority,
            reasons=reasons or ["all checks passed"],
        )


def train_review_model(
    features: list[dict[str, float]],
    labels: list[int],
    target_error_budget: float = 0.05,
) -> dict:
    """Train a calibrated review-priority model.

    Calibration matters: the threshold is chosen against an error budget, and that is
    only meaningful if a predicted 0.9 actually means roughly nine times out of ten.

    The threshold is selected on a validation split and then frozen. Selecting it on
    the same data used to report the result would inflate the number.
    """
    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss
    from sklearn.model_selection import train_test_split

    X = np.array([[f.get(name, 0.0) for name in FEATURE_ORDER] for f in features])
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y if len(set(labels)) > 1 else None
    )

    base = GradientBoostingClassifier(random_state=0)
    # Calibration needs at least one example per class per fold. On a small or skewed
    # training set, asking for 3 folds raises rather than degrading, so fold count is
    # capped by the rarest class.
    rarest = int(min(np.bincount(y_train))) if len(set(y_train)) > 1 else 0
    folds = max(2, min(3, rarest))
    if rarest < 2:
        base.fit(X_train, y_train)
        model = base
    else:
        model = CalibratedClassifierCV(base, cv=folds, method="isotonic")
        model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    # Highest threshold whose auto-accepted set stays inside the error budget.
    #
    # The default is 0.0, meaning accept nothing. This matters: if no threshold can
    # satisfy the budget, the safe answer is to route every document to a human, not
    # to wave them all through. An earlier version defaulted to 1.0 and produced 98%
    # auto-acceptance at an 83% error rate - the exact opposite of what a review
    # system is for. In a safety path the fallback must fail closed.
    chosen = 0.0
    coverage = 0.0
    budget_met = False
    for candidate in np.linspace(0.05, 0.95, 91):
        auto = probabilities < candidate
        if auto.sum() == 0:
            continue
        error_rate = float(y_val[auto].mean())
        if error_rate <= target_error_budget:
            chosen = float(candidate)
            coverage = float(auto.mean())
            budget_met = True

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    result = {
        "threshold": chosen,
        "target_error_budget": target_error_budget,
        "auto_accept_coverage": coverage,
        "val_pr_auc": float(average_precision_score(y_val, probabilities))
        if len(set(y_val)) > 1
        else None,
        "val_brier": float(brier_score_loss(y_val, probabilities))
        if len(set(y_val)) > 1
        else None,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "features": FEATURE_ORDER,
        "budget_met": budget_met,
        "base_error_rate": float(y_val.mean()),
        "note": (
            "Threshold selected on the validation split and frozen. When no threshold "
            "satisfies the budget, the model fails closed: nothing is auto-accepted."
        ),
    }
    THRESHOLD_PATH.write_text(json.dumps(result, indent=2))
    return result
