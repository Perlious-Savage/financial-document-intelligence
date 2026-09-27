# AI-Powered Financial Document Intelligence

Structured field extraction from financial documents, with deterministic validation and
confidence-based routing to human review.

**Model:** [perlious-Savage/layoutlmv3-cord-extraction](https://huggingface.co/perlious-Savage/layoutlmv3-cord-extraction)

Every figure below comes from code in this repository. Raw outputs are in [`artifacts/`](artifacts/);
results from the Colab GPU runs were transcribed there from the run logs, and `mlflow.db` is the
machine-written record of the hyperparameter sweep.

---

## Results

Fine-tuned LayoutLMv3 against a no-model baseline, on the CORD-v2 test split
(100 documents, 1,309 entities, 59 field types).

| Method | Precision | Recall | F1 |
|:---|---:|---:|---:|
| Keyword + position rules | 0.557 | 0.468 | 0.509 |
| **LayoutLMv3 fine-tuned** | **0.942** | **0.955** | **0.948** |

Scoring is **entity-level**: a field counts only when its type *and* full span match the reference
exactly. A partially extracted merchant name scores zero, because a partially extracted field is
not a usable one. Implementation in [`src/metrics.py`](src/metrics.py).

**Both methods are evaluated on CORD's ground-truth words and bounding boxes** (`apply_ocr=False`),
so these figures measure extraction *given perfect OCR*. No OCR engine is part of this pipeline, and
accuracy on raw scans would be lower.

<details>
<summary><b>Training configuration</b></summary>

| | |
|:---|:---|
| Base model | `microsoft/layoutlmv3-base` |
| Epochs | 8 |
| Learning rate | 5e-5 |
| Batch size | 4 |
| Training documents | 800 |
| Label classes | 59 |
| Hardware | A100, 711s |

Validation F1 by epoch: 0.820, 0.923, 0.921, 0.950, 0.946, 0.952, 0.955, 0.955.
Plateaus from epoch 6, so the final two epochs added nothing.

</details>

### Hyperparameter sweep

Four learning rates at 4 epochs each, tracked in MLflow. The aggregate spread is the
misleading number, so it is worth reading the rows rather than the summary.

| lr | epochs | F1 | precision | recall |
|---:|---:|---:|---:|---:|
| 3e-05 | 4 | **0.9280** | 0.920 | 0.936 |
| 5e-05 | 4 | **0.9263** | 0.917 | 0.936 |
| 1e-04 | 4 | **0.9260** | 0.915 | 0.937 |
| 1e-05 | 4 | **0.8127** | 0.797 | 0.829 |

Learning rate appears to move F1 by **0.115**, but the entire spread comes from one
configuration. The top three sit within **0.002** of each other, which is at or below
seed noise for a single-seed run, so the fine-tune is effectively insensitive to learning
rate across that band.

And `1e-05` is not a worse learning rate, it is **undertrained**. Its evaluation loss was
still descending steeply when training stopped (1.562, 1.037, 0.823, 0.744, no plateau),
so at 4 epochs it had simply not converged.

The controlled comparison is more useful. Holding learning rate and batch size fixed and
changing only the epoch count:

| config | F1 |
|:---|---:|
| lr 5e-05, 4 epochs | 0.9263 |
| lr 5e-05, 8 epochs (reference run) | **0.9484** |

**Training budget moves the result more than learning rate does within a sensible band:**
+0.022 from doubling epochs, against 0.002 across the top three learning rates. One seed
per configuration, so differences below roughly 0.005 are not meaningful.

One methodological caveat: the sweep scores every configuration on the **test** split, so its
"best" learning rate is selected on test data. The headline 0.948 comes from the default
configuration, fixed before the sweep, and was not selected this way.

![MLflow runs](docs/mlflow-runs.png)

Raw results in [`artifacts/sweep_results.json`](artifacts/sweep_results.json). The tracked runs
are committed in `mlflow.db` and can be inspected directly:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Six runs are stored. The four above, plus two single-epoch runs on a 40-document subset that were
smoke tests for the sweep harness - their low scores are expected and are not results.

### Where the gain came from

The aggregate jump is not spread evenly. The rule baseline handles fields printed beside their own
label, and scores **exactly zero** on fields identified by position alone: a quantity in a column,
a unit price placed relative to its line item. Closing that gap is what a layout-aware model is for.

| Field | Support | Rules | LayoutLMv3 |
|:---|---:|---:|---:|
| `menu.cnt` | 220 | 0.000 | **0.984** |
| `menu.unitprice` | 67 | 0.000 | **0.933** |
| `menu.sub.nm` | 36 | 0.000 | **0.892** |
| `menu.sub.price` | 20 | 0.000 | **0.947** |
| `menu.sub.cnt` | 17 | 0.000 | **0.944** |

### Where it still fails

Every remaining weak field is a rare class, and the pattern is consistent rather than scattered.

| Field | Support | F1 |
|:---|---:|---:|
| `total.emoneyprice` | 2 | 0.400 |
| `menu.etc` | 3 | 0.000 |
| `total.total_etc` | 3 | 0.000 |
| `menu.itemsubtotal` | 6 | 0.000 |
| `menu.discountprice` | 10 | 0.571 |

None has more than ten test examples. Two are catch-all `etc` categories, where scarcity compounds
with genuine ambiguity about what belongs in them. This is a data-coverage limitation, not an
architectural one.

---

## What is measured, and what is not

| Component | Status |
|:---|:---|
| LayoutLMv3 extraction | **Measured** - 0.948 F1, single run |
| Rule baseline | **Measured** - 0.509 F1, deterministic |
| Per-field analysis | **Measured** - gain located in position-dependent fields |
| Entity-level metric | **Tested** - implemented here, covered by tests |
| Decimal validation | **Tested** - reconciliation logic covered |
| LangGraph routing | **Verified** - both branches exercised |
| FastAPI + Docker | **CI-verified** - builds and serves without a GPU |
| Semantic retrieval | **Implemented, not justified** - see below |
| Review-priority model | **Implemented, not deployed** - see below |
| MLflow | **Used** - 4-configuration sweep, sensitivity analysis |
| Model-backed API extraction | **Not implemented** - the shipped container runs a stub; the model has only run offline |

Not measured, and therefore not claimed: latency, cost per document, KYC or compliance
performance, confidence intervals on any figure.

### Two negative results

**Semantic retrieval is not justified by the evidence.** PostgreSQL with pgvector indexes the
corpus, and `GET /documents/{id}/similar` returns nearest neighbours by cosine distance. It works
qualitatively: the nearest neighbour of `cord-test-0000` is the same vendor and product code at a
different quantity. But against a TF-IDF baseline on planted near-duplicates, both scored
**Recall@5 = 1.000**. The benchmark is saturated. 100 documents is a small index, and duplicates
built by reordering tokens stay lexically close, which is the case TF-IDF handles well. A real
test needs a larger index with distractors and genuinely paraphrased duplicates.

**Selective automation is gated on extraction quality.** The review-priority model trains on real
labels: whether extraction fell below a usable quality bar, measured against ground truth. With
the rule baseline, median document-level F1 is 0.545 and 84% of validation documents fall below the bar, so no
confidence threshold can deliver a 5% error budget. The correct outcome is zero automation, because
you cannot route your way out of a weak extractor. Since the model failed its acceptance criterion
it is **not deployed**: the deterministic triage runs instead, and the service reports which path
is active.

---

## Architecture

```
                        Input document
                              |
                              v
                     +------------------+
                     |    Extraction    |  stub in the shipped container;
                     |                  |  LayoutLMv3 run offline on GPU
                     +--------+---------+
                              v
                      Structured JSON        Pydantic schema
                              |
                              v
                     +------------------+
                     |   Validation     |  Decimal arithmetic
                     |                  |  line items -> subtotal -> total
                     +--------+---------+
                              v
                         confidence?
                    +---------+---------+
                  low                  high
                    |                    |
                    v                    v
              human review        review priority     scikit-learn
                    |                    |
                    +---------+----------+
                              v
                     +------------------+
                     | FastAPI (CPU)    |
                     +---+----------+---+
                         |          |
                         v          v
                  PostgreSQL    GPU inference
                   + pgvector    (remote, not verified)

     MLflow tracks training runs.
     LangGraph orchestrates validation, the confidence branch and scoring;
     extraction runs before the graph is invoked.
```

**The CPU container is the deployable unit.** Extraction is the only stage needing a GPU, so it
sits behind an interface selected by `MODEL_BACKEND` (`stub` or `remote`). The service starts, serves
and passes CI with no GPU attached. Note what that means: the shipped `stub` returns a fixed synthetic
receipt, so the deployed API demonstrates validation and routing, not extraction. The `remote` path
and the notebook's serving cell have not been verified end to end.

---

## Design decisions

**All monetary arithmetic happens in Python, never in a model.** Language models produce plausible
numbers rather than correct ones, and the failure is silent: a wrong total looks exactly like a
right one. Computation lives in [`src/tools.py`](src/tools.py) using `Decimal`, so that error class
is gone by construction rather than by prompting. `Decimal` specifically because `0.1 + 0.2` does
not equal `0.3` in binary floating point, and a cent of drift decides whether a receipt reconciles.

**The routing decision derives from structured findings, never from generated text.** Checks emit
`Finding` objects, and the decision is a function of those objects. A model cannot introduce an
unsupported claim into the decision path.

**Abstention is modelled in the schema, not yet in the evaluation.** `FieldStatus` in
[`src/schemas.py`](src/schemas.py) distinguishes correct, incorrect and abstained, and checks with
missing inputs skip rather than fail. The reported metrics are standard precision, recall and F1;
a separate abstention rate is not computed.

**A model that fails its acceptance criterion is not deployed.** The review scorer only goes live
if its threshold satisfied the configured error budget during training. Otherwise the service falls
back to deterministic triage, which makes no error-budget claim.

**LangGraph is used for explicit state transitions, not to run an agent.** No model chooses what
happens next. What earns the graph is the conditional edge: documents below the confidence floor
skip priority scoring entirely and route straight to a human.

---

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env          # add your PostgreSQL connection string
pytest tests/ -q
python smoke_test.py          # end-to-end check
uvicorn src.api:app --reload  # docs at http://localhost:8000/docs
```

Reproduce the measured results:

```bash
python eval_baseline.py                       # rule baseline   -> 0.509 F1
python -m src.train_extractor --epochs 8      # fine-tune (GPU) -> 0.948 F1
python index_corpus.py && python eval_retrieval.py
python train_review.py
```

The GPU steps also run from [`notebooks/run_colab.ipynb`](notebooks/run_colab.ipynb).

### Endpoints

| Endpoint | Purpose |
|:---|:---|
| `POST /extract` | Document in, structured fields out |
| `POST /analyze` | Full pipeline: extract, validate, route |
| `GET /documents/{id}/similar` | Semantic retrieval over the indexed corpus |
| `GET /metrics` | Measured results, or a clear statement that none exist |
| `GET /health` | Status and active model backend |

---

## Project layout

```
src/
  schemas.py          Pydantic contract shared across the pipeline
  data.py             CORD-v2 loader and BIO label construction
  extract.py          Extraction backends behind one interface
  baseline.py         Keyword and position rules, no model
  train_extractor.py  LayoutLMv3 fine-tune, MLflow tracked
  metrics.py          Entity-level scoring
  tools.py            Deterministic financial arithmetic (Decimal)
  checks.py           Validation, emits structured findings
  risk.py             Review-priority scoring and routing
  graph.py            LangGraph state machine
  index.py            Embeddings and pgvector retrieval
  api.py              FastAPI service
artifacts/            Measured results, committed as evidence
tests/                Test suite
```

---

## Limitations

- **CORD-v2 is a receipt dataset**, not a KYC or onboarding corpus. This measures document
  extraction, not regulatory performance.
- **Single training run.** No repeated seeds, no confidence intervals. With 100 test documents the
  interval around 0.948 is not tight, and a rerun would move the third decimal.
- **Evaluation assumes perfect OCR.** Words and boxes come from CORD's annotations, not an OCR engine.
- **The deployed API does not run the model.** It ships with a stub backend; the fine-tuned model has
  only been run offline via `trainer.predict`.
- **The validation layer is only partly fitted to receipts.** `date` is not an annotated CORD category
  and `merchant` falls back to the first item name, so the required-fields check flags most real
  receipts; line-item quantity is fixed at 1, so the per-line arithmetic check cannot fail on
  extracted data.
- **Retrieval is semantic search, not RAG.** Nothing retrieved is injected into a generation step.
- **Indonesian number formatting.** Amounts use `.` as a thousands separator, so `12.000` is twelve
  thousand. Parsing assumes this convention.

Comparable published work (`nielsr/layoutlmv3-finetuned-cord`) reports 0.964 F1 under a different
evaluation setup, which is context rather than a like-for-like comparison.

---

## Dataset and licensing

Training data: [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), CC-BY-4.0.
Base model: `microsoft/layoutlmv3-base`, CC-BY-NC-4.0, so the fine-tuned model inherits the
non-commercial term.
