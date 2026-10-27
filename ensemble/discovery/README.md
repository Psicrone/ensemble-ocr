# ensemble.discovery: from unlabelled documents to labelling sheets

For when you have sample documents but no document types, field lists or labels yet.
Gemini groups the documents into classes, proposes the fields for each class, and
drafts every document's values. A person then corrects drafts instead of typing labels.

```bash
export GOOGLE_CLOUD_PROJECT=my-project GOOGLE_CLOUD_LOCATION=us-central1

python -m ensemble.discovery scan     docs/          # how many documents, already sorted into class folders?
python -m ensemble.discovery classify docs/ --k 5    # -> docs/_discovery/classes.json      (check it)
python -m ensemble.discovery propose  docs/          # -> docs/_discovery/specs/<class>.yaml (edit them)
python -m ensemble.discovery prelabel docs/          # -> docs/_discovery/prelabels.jsonl
```

Then make the sheets with the labelling tools: `labels template docs/ --specs docs/_discovery/specs
--prefill docs/_discovery/prelabels.jsonl`, correct them, and `labels import` them into `gold.jsonl`.

- **Sorted folders skip a step.** If the documents are already in `docs/<class>/<file>`, the folder
  names are the classes and `classify` makes no model calls.
- **Edits are kept.** `propose` never overwrites an existing `specs/<class>.yaml` unless you pass
  `--overwrite`, and each step reads the files the previous one wrote.
- **Reruns are free and stable.** Every answer is cached in `_discovery/.cache` by the full request
  (model, prompt, schema, document bytes), and calls run at temperature 0 with a fixed seed.
- **Nothing leaves the documents' folder.** Output goes to `docs/_discovery/` by default, so keep
  that folder out of any repository.
- **Failures don't stop a run.** A document the model could not read goes to `errors.jsonl`.

Formats are the ones the labelling tools read: a spec is `{doc_type, version, description, dayfirst,
fields: [{path, type, required, critical, description, example, values, pattern}]}`, and a draft is
`{"doc_id", "doc_type", "source", "fields"}` where `doc_id` is the file's path under `docs/` without
its extension.

From Python, any object with `generate(prompt, blobs, schema) -> dict` can stand in for Gemini:

```python
from ensemble.discovery import CachedModel, GeminiJsonModel, classify, propose, prelabel

model = CachedModel(GeminiJsonModel("gemini-2.5-pro"), "docs/_discovery/.cache")
classify("docs", "docs/_discovery", model, k=5)
```
