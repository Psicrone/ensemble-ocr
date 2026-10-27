"""``python -m ensemble.evaluation init|template|import|check|score|json-schema``."""
from __future__ import annotations

import json
from pathlib import Path

from evalkit.structured import read_records

from .labels import check_labels, import_labels, scan_documents, write_templates
from .specs import load_specs, starter_spec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m ensemble.evaluation",
                                 description="Document-type specs, ground-truth labelling and scoring.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write a starter spec for a document type")
    p.add_argument("doc_type")
    p.add_argument("--out", default="specs", help="specs directory")

    p = sub.add_parser("template", help="blank labelling sheets for a folder of documents")
    p.add_argument("docs", help="documents folder; first-level subfolders name the doc type")
    p.add_argument("--specs", required=True)
    p.add_argument("--out", default="labels")
    p.add_argument("--doc-type", help="use this type for every document instead of folder names")
    p.add_argument("--max-items", type=int, default=5, help="table rows per document in the CSV")
    p.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    p.add_argument("--prefill", help="draft values (gold-shaped JSONL, e.g. from a model) to pre-fill")

    p = sub.add_parser("import", help="filled CSV/JSONL sheets -> gold.jsonl (canonicalised and checked)")
    p.add_argument("files", nargs="+")
    p.add_argument("--specs", required=True)
    p.add_argument("--out", default="gold.jsonl")

    p = sub.add_parser("check", help="validate a gold file against the specs")
    p.add_argument("gold")
    p.add_argument("--specs", required=True)

    p = sub.add_parser("score", help="score saved ExtractionResults against gold (writes an evalkit report)")
    p.add_argument("results", help="JSONL of ExtractionResult, doc_id set to the gold doc_id")
    p.add_argument("gold")
    p.add_argument("--specs", required=True)
    p.add_argument("--out", default="results/eval")

    p = sub.add_parser("json-schema", help="write one JSON Schema per doc type for the pipeline")
    p.add_argument("--specs", required=True)
    p.add_argument("--out", default="schemas")

    a = ap.parse_args(argv)
    if a.cmd == "init":
        path = Path(a.out) / f"{a.doc_type}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit(f"{path} already exists")
        path.write_text(starter_spec(a.doc_type), encoding="utf-8")
        print(f"wrote {path}; edit its fields, then run `python -m ensemble.evaluation template`")
        return
    specs = load_specs(a.specs)
    if a.cmd == "template":
        docs = scan_documents(a.docs)
        for path in write_templates(docs, specs, a.out, max_items=a.max_items, fmt=a.format,
                                    doc_type=a.doc_type, prefill=a.prefill):
            print(f"wrote {path}")
        print(f"{len(docs)} documents; doc types: {sorted(specs)}")
        if (Path(a.out) / "_unassigned.csv").exists():
            print("Some documents have no known doc type (see _unassigned.csv): put them in a "
                  "folder named after their type, or pass --doc-type.")
    elif a.cmd == "import":
        recs = import_labels(a.files, specs)
        with open(a.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        print(f"wrote {len(recs)} gold records to {a.out}")
        _print_issues(check_labels(recs, specs))
    elif a.cmd == "check":
        _print_issues(check_labels(read_records(a.gold), specs))
    elif a.cmd == "score":
        from .score import score_results
        path, res = score_results(a.results, a.gold, specs=specs, out=a.out)
        f = res.summary["fields"]
        print(f"field accuracy {f['accuracy']:.3f}  precision {f['precision']:.3f}  recall {f['recall']:.3f}")
        print(f"report: {Path(path).resolve()}")
    elif a.cmd == "json-schema":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for t, s in specs.items():
            path = out / f"{t}.v{s.version}.json"
            path.write_text(json.dumps(s.json_schema(), indent=2), encoding="utf-8")
            print(f"wrote {path}")


def _print_issues(issues):
    if not issues:
        print("labels OK")
        return
    print(f"{len(issues)} problem(s):")
    for i in issues:
        print(f"  {i['doc_id']}  {i['field'] or '-'}: {i['problem']}")
