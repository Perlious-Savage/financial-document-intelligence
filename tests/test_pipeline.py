"""Pipeline and API tests. These run without a GPU, which is why CI can run them."""

import io
from decimal import Decimal

from fastapi.testclient import TestClient

from src.api import app
from src.extract import from_token_predictions
from src.graph import analyze
from src.schemas import ExtractedReceipt

client = TestClient(app)

JPEG = b"\xff\xd8\xff" + b"x" * 200


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_rejects_non_document_upload():
    response = client.post(
        "/analyze", files={"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code == 415


def test_rejects_empty_upload():
    response = client.post("/analyze", files={"file": ("x.jpg", io.BytesIO(b""), "image/jpeg")})
    assert response.status_code == 400


def test_analyze_returns_findings_and_decision():
    response = client.post("/analyze", files={"file": ("d.jpg", io.BytesIO(JPEG), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["findings"]
    assert "review_required" in body["decision"]


def test_token_predictions_become_decimals():
    receipt = from_token_predictions(
        "t",
        ["Total", "22.000"],
        ["O", "B-total.total_price"],
        [1.0, 1.0],
    )
    # Indonesian thousands separator: 22.000 is twenty-two thousand.
    assert receipt.total == Decimal("22000")


def test_low_confidence_takes_the_review_branch():
    receipt = ExtractedReceipt(
        doc_id="low",
        merchant="A",
        date="2024-01-01",
        total=Decimal("10"),
        subtotal=Decimal("10"),
        field_confidences={"total": 0.2},
    )
    result = analyze(receipt)
    assert result.decision.review_required
    assert "below floor" in result.decision.reasons[0]


def test_high_confidence_clean_receipt_is_not_flagged():
    receipt = ExtractedReceipt(
        doc_id="clean",
        merchant="A",
        date="2024-01-01",
        subtotal=Decimal("100"),
        tax=Decimal("10"),
        total=Decimal("110"),
        field_confidences={"merchant": 0.96, "date": 0.94, "total": 0.95},
    )
    result = analyze(receipt)
    assert not result.decision.review_required


def test_entity_metric_requires_exact_span_and_type():
    from src.metrics import extract_entities, precision_recall_f1

    reference = [["B-merchant", "I-merchant", "O", "B-total"]]

    # exact match
    assert precision_recall_f1(reference, reference)[2] == 1.0

    # a partially extracted field is not a usable field, and must not score
    partial = [["B-merchant", "O", "O", "B-total"]]
    assert precision_recall_f1(reference, partial)[2] == 0.5

    # right span, wrong type
    wrong_type = [["B-total", "I-total", "O", "B-total"]]
    assert precision_recall_f1(reference, wrong_type)[2] == 0.5

    # malformed sequences open a new entity rather than being dropped silently
    assert ("merchant", 0, 2) in extract_entities(["I-merchant", "I-merchant"])


def test_baseline_spans_cover_label_and_value():
    """CORD annotates the printed label and its value as one entity."""
    from src.baseline import predict_tags

    tags = predict_tags(["TOTAL", "60.000"])
    assert tags == ["B-total.total_price", "I-total.total_price"]


def test_baseline_does_not_swallow_the_following_line():
    from src.baseline import predict_tags

    tags = predict_tags(["TAX", "5.455", "Subtotal", "60.000"])
    assert tags[1] == "I-sub_total.tax_price"
    assert tags[2] == "B-sub_total.subtotal_price"


def test_baseline_prefers_the_longer_trigger_phrase():
    """'TOTAL DISC' is a discount line, not the grand total."""
    from src.baseline import predict_tags

    assert predict_tags(["TOTAL", "DISC", "-60.000"])[0] == "B-sub_total.discount_price"


def test_threshold_fails_closed_when_budget_is_unreachable(tmp_path, monkeypatch):
    """If no threshold meets the error budget, auto-accept nothing.

    The earlier default accepted everything, which produced 98% auto-acceptance at an
    83% error rate. A review system that fails open is worse than no review system.
    """
    from src import risk
    from src.risk import train_review_model

    # train_review_model writes its model and threshold to disk. Redirect them, or every
    # test run overwrites the committed artifacts with this toy model.
    monkeypatch.setattr(risk, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(risk, "THRESHOLD_PATH", tmp_path / "t.json")
    monkeypatch.setattr(risk, "ARTIFACTS", tmp_path)

    # Features carry no signal and almost every document is bad: no threshold can
    # deliver a 5% error rate.
    features = [{"mean_confidence": 0.5} for _ in range(100)]
    labels = [1] * 80 + [0] * 20

    result = train_review_model(features, labels, target_error_budget=0.05)
    assert result["budget_met"] is False
    assert result["threshold"] == 0.0
    assert result["auto_accept_coverage"] == 0.0


def test_scorer_refuses_a_model_that_failed_its_budget(tmp_path, monkeypatch):
    """A model that could not meet its error budget does not get deployed."""
    import json

    from src import risk

    monkeypatch.setattr(risk, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(risk, "THRESHOLD_PATH", tmp_path / "t.json")
    (tmp_path / "m.joblib").write_bytes(b"not actually a model")
    (tmp_path / "t.json").write_text(json.dumps({"threshold": 0.0, "budget_met": False}))

    scorer = risk.ReviewScorer()
    assert scorer.model is None
    assert "did not meet its error budget" in scorer.source
