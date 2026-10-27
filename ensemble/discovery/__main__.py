"""python -m ensemble.discovery {scan,classify,propose,prelabel,all} DOCS [options]

    scan       count documents and say whether they are already sorted into class folders
    classify   group documents into --k classes          -> OUT/classes.json
    propose    propose a field spec per class            -> OUT/specs/<class>.yaml
    prelabel   draft every document's field values       -> OUT/prelabels.jsonl
    all        classify, propose and prelabel in one go

OUT defaults to DOCS/_discovery, so nothing from the documents lands in a repository.
Check classes.json and the specs between steps; each step reads what the last one wrote.
Then: evalkit labels template DOCS --specs OUT/specs --prefill OUT/prelabels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .discover import classify, prelabel, propose, scan
from .model import CachedModel, GeminiJsonModel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ensemble.discovery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["scan", "classify", "propose", "prelabel", "all"])
    ap.add_argument("docs", type=Path, help="folder of documents (PDF or images)")
    ap.add_argument("--out", type=Path, help="output folder (default: DOCS/_discovery)")
    ap.add_argument("--k", type=int, default=5, help="number of classes for an unsorted folder")
    ap.add_argument("--examples", type=int, default=5, help="documents shown per class when proposing fields")
    ap.add_argument("--only", nargs="+", help="prelabel only these classes")
    ap.add_argument("--overwrite", action="store_true", help="replace existing specs/<class>.yaml")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--project", help="GCP project (default: $GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--location", help="Vertex AI region (default: $GOOGLE_CLOUD_LOCATION or us-central1)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    out = a.out or a.docs / "_discovery"
    if a.step == "scan":
        docs = scan(a.docs)
        print(
            json.dumps(
                {
                    "documents": len(docs),
                    "sorted_into_class_folders": bool(docs) and all(d.folder_class for d in docs),
                    "folder_classes": dict(Counter(d.folder_class for d in docs if d.folder_class)),
                },
                indent=2,
            )
        )
        return 0

    model = CachedModel(GeminiJsonModel(a.model, a.project, a.location), out / ".cache")
    if a.step in ("classify", "all"):
        res = classify(a.docs, out, model, k=a.k, workers=a.workers)
        print(f"classes -> {out / 'classes.json'}: {dict(Counter(res['assignments'].values()))}")
    if a.step in ("propose", "all"):
        for name, spec in propose(a.docs, out, model, examples=a.examples, overwrite=a.overwrite).items():
            print(f"spec {name}: {len(spec['fields'])} fields -> {out / 'specs' / (name + '.yaml')}")
    if a.step in ("prelabel", "all"):
        path = prelabel(a.docs, out, model, workers=a.workers, only=a.only)
        print(f"drafts -> {path}")
    print(f"model calls: {model.misses} new, {model.hits} from cache")
    if (out / "errors.jsonl").exists():
        print(f"some documents had problems, see {out / 'errors.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
