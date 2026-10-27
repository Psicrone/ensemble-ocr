# ensemble.evaluation

Evaluation for the extraction pipeline. It uses [evalkit](https://github.com/Psicrone/evalkit)
as a library (`pip install "evalkit @ git+https://github.com/Psicrone/evalkit"`). evalkit
does the generic field-by-field scoring, and this package adds everything specific to
documents.

| module | what it does | needs evalkit |
|---|---|---|
| `specs.py` | Document-type specs: one YAML field list per class. Builds the extractor's JSON Schema (`json_schema()`) and evalkit comparators (`eval_schema()`) | only for `eval_schema()` |
| `labels.py` | Labelling sheets from `docs/<doc_type>/` (pre-filled with draft values if given), import back to gold, label checks | yes |
| `adapter.py` | `ExtractionResult` → evalkit values plus per-field confidence and status | yes |
| `score.py` | `score_pipeline(extract, documents, gold, specs=..., runs=3)` runs the pipeline and scores it. `score_results` scores saved results | yes |

```bash
python -m ensemble.evaluation init receipt --out specs            # starter spec per class
python -m ensemble.evaluation json-schema --specs specs --out schemas
python -m ensemble.evaluation template docs/ --specs specs --out labels [--prefill drafts.jsonl]
python -m ensemble.evaluation import labels/*.csv --specs specs --out gold.jsonl
python -m ensemble.evaluation check gold.jsonl --specs specs
python -m ensemble.evaluation score results.jsonl gold.jsonl --specs specs --out results/run1
```

```python
import asyncio, functools
from ensemble.extract import extract
from ensemble.evaluation.labels import documents_from_gold
from ensemble.evaluation.score import score_pipeline

docs = documents_from_gold("gold.jsonl", "docs/")
path, res = asyncio.run(score_pipeline(functools.partial(extract, config=cfg), docs, "gold.jsonl",
                                       specs="specs/", runs=3, out="results/baseline"))
```

Gold records: `{"doc_id": "<doc_type>/<file stem>", "doc_type", "source": "<path under docs/>", "fields": {...}}`.
The report covers field accuracy, precision and recall per field and per doc type. It
also gives the critical-field pass rate, accepted precision (`accepted` / `corrected`),
review rate (`needs_review` / `failed`), the confidence ROC with an auto-accept table, and
run-to-run stability.

Tests: `tests/evaluation/` (offline, fake `extract`). Example spec: `examples/specs/invoices.yaml`.
