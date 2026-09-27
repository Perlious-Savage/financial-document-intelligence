"""Extraction backends behind one interface.

Two backends, selected by MODEL_BACKEND:

  stub    deterministic, no model, no GPU. Returns a fixed synthetic receipt so the
          container and the API run and can be tested anywhere, including CI.
  remote  calls a GPU inference service over HTTP. The serving cell in
          notebooks/run_colab.ipynb is a starting point, but this path has not been
          verified end to end (see that notebook's notes).

The CPU container ships with `stub`, so the deployed API exercises validation and
routing, not the model. The fine-tuned model has only been run offline, through
`trainer.predict` in src/train_extractor.py.
"""

from __future__ import annotations

import os
from decimal import Decimal

from .schemas import ExtractedReceipt, LineItem, Provenance
from .tools import parse_money

BACKEND = os.environ.get("MODEL_BACKEND", "stub").lower()
ENDPOINT = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")

# CORD category -> the field it populates in our schema.
FIELD_MAP = {
    "total.total_price": "total",
    "sub_total.subtotal_price": "subtotal",
    "sub_total.tax_price": "tax",
    "sub_total.service_price": "service_charge",
    "sub_total.discount_price": "discount",
}


def from_token_predictions(
    doc_id: str,
    words: list[str],
    tags: list[str],
    confidences: list[float] | None = None,
) -> ExtractedReceipt:
    """Assemble BIO token predictions into a validated receipt.

    Groups consecutive tokens sharing a tag, then maps CORD categories onto schema
    fields. Monetary strings go through parse_money so the receipt holds Decimals,
    never floats or raw strings.
    """
    confidences = confidences or [1.0] * len(words)
    spans: list[tuple[str, list[str], list[float]]] = []
    for word, tag, conf in zip(words, tags, confidences):
        category = tag[2:] if tag[:2] in ("B-", "I-") else None
        if category is None:
            continue
        if tag.startswith("B-") or not spans or spans[-1][0] != category:
            spans.append((category, [word], [conf]))
        else:
            spans[-1][1].append(word)
            spans[-1][2].append(conf)

    receipt = ExtractedReceipt(doc_id=doc_id)
    item_names: list[str] = []
    item_prices: list[Decimal] = []

    for category, tokens, confs in spans:
        text = " ".join(tokens)
        confidence = sum(confs) / len(confs)

        if category in FIELD_MAP:
            value = parse_money(text)
            if value is not None:
                setattr(receipt, FIELD_MAP[category], value)
                receipt.field_confidences[FIELD_MAP[category]] = round(confidence, 4)
                receipt.provenance[FIELD_MAP[category]] = Provenance(source_text=text)
        elif category == "menu.nm":
            item_names.append(text)
        elif category == "menu.price":
            price = parse_money(text)
            if price is not None:
                item_prices.append(price)

    for index, name in enumerate(item_names):
        receipt.line_items.append(
            LineItem(
                name=name,
                quantity=Decimal("1"),
                unit_price=item_prices[index] if index < len(item_prices) else None,
                total_price=item_prices[index] if index < len(item_prices) else None,
            )
        )
    if receipt.merchant is None and item_names:
        receipt.merchant = item_names[0]
    return receipt


def _stub(doc_id: str) -> ExtractedReceipt:
    """A fixed, obviously-synthetic receipt. Never presented as a model prediction."""
    return ExtractedReceipt(
        doc_id=doc_id,
        merchant="STUB BACKEND - no model loaded",
        date="2024-01-01",
        line_items=[
            LineItem(name="item a", quantity=Decimal("2"), unit_price=Decimal("15000")),
            LineItem(name="item b", quantity=Decimal("1"), unit_price=Decimal("20000")),
        ],
        subtotal=Decimal("50000"),
        tax=Decimal("5000"),
        total=Decimal("55000"),
        field_confidences={"merchant": 0.5, "date": 0.5, "total": 0.5},
    )


def _remote(doc_id: str, image_bytes: bytes) -> ExtractedReceipt:
    import httpx

    if not ENDPOINT:
        raise RuntimeError("MODEL_BACKEND=remote but MODEL_ENDPOINT is not set")
    response = httpx.post(
        f"{ENDPOINT}/predict",
        files={"file": (doc_id, image_bytes)},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return from_token_predictions(
        doc_id, payload["words"], payload["tags"], payload.get("confidences")
    )


def extract(doc_id: str, image_bytes: bytes | None = None) -> ExtractedReceipt:
    if BACKEND == "remote":
        if image_bytes is None:
            raise ValueError("remote backend needs the document bytes")
        return _remote(doc_id, image_bytes)
    if BACKEND == "stub":
        return _stub(doc_id)
    # Fail loudly rather than silently falling back to the stub on a typo.
    raise ValueError(f"Unknown MODEL_BACKEND {BACKEND!r}; supported: 'stub', 'remote'")
