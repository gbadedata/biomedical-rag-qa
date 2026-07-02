# biomedical-rag-qa

A small, reusable toolkit for **retrieval-augmented question answering over biomedical
literature**. It ingests abstracts, indexes them for search, retrieves the passages that
bear on a question, and generates a grounded yes / no / maybe answer with an LLM. Every
stage is a separate, testable module, and the whole thing is benchmarked against naive
baselines on the public PubMedQA dataset.

```
ingest ─▶ corpus ─▶ retrieve ─▶ generate ─▶ evaluate
```
REST and GraphQL ingestion, section chunking, BM25 or dense-FAISS retrieval, grounded
LLM generation, and evaluation against baselines.

## Why it exists

Medical teams sit on large volumes of unstructured text (abstracts, trial records,
label documents) and increasingly want to query it in natural language. RAG is the
standard pattern for doing that safely: ground the model in retrieved evidence rather
than its training memory. This repo is a compact, transparent reference implementation
with the evaluation attached, so the retrieval and answering steps can be measured
rather than assumed.

## Install

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # downloads PubMedQA ori_pqal.json into data/
```

A 40-item sample (`data/sample_pqal.json`) is committed for quick tests, so the fetch
step is optional for a first look.

## Use

```bash
# Retrieve passages for a question (and, if ANTHROPIC_API_KEY is set, answer it)
python -m biomedqa.cli ask --data data/ori_pqal.json --retriever bm25 --k 5 \
    --question "Does metformin reduce cancer risk in type 2 diabetes?"

# Reproduce the full evaluation (writes results/evaluation.json)
python -m biomedqa.cli eval --data data/ori_pqal.json

# Ingest fresh text from public APIs
python -m biomedqa.cli fetch-europepmc --query "psoriasis biologics" --max 100 --out epmc.json
python -m biomedqa.cli fetch-ctgov --condition "atrial fibrillation" --max 100 --out ctgov.json
python -m biomedqa.cli fetch-targets --efo EFO_0000305 --out targets.json   # breast carcinoma
```

The generation step reads `ANTHROPIC_API_KEY` and, optionally, `ANTHROPIC_MODEL`
(set it to a current model name). Without a key, `ask` still runs retrieval and prints
the passages; generation is skipped.

## Results (PubMedQA, 1,000 labelled questions)

Corpus: 3,358 passages, ~60 tokens each. Decisions: yes 552, no 338, maybe 110.

**Retrieval.** Does the index find the right supporting passages?

| Retriever | P@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| BM25 (lexical) | **0.943** | **0.673** | **0.731** | **0.959** |
| Dense, LSA + FAISS | 0.802 | 0.602 | 0.687 | 0.846 |
| Random baseline | 0.000 | 0.001 | 0.002 | 0.001 |

BM25 places a gold passage first for 94% of questions. The dense index (truncated-SVD
LSA vectors in FAISS) trails it here because the questions share so much vocabulary with
their source abstracts that lexical matching is hard to beat; a biomedical transformer
embedder is the intended production swap-in (see design notes).

**Answering.** Does retrieved context help decide yes/no/maybe? A logistic-regression
decision head, test set n=300, reported next to the majority-class baseline.

| Feature set | Accuracy | Macro-F1 |
|---|---|---|
| Majority baseline (always "yes") | 0.553 | 0.237 |
| Question only (no retrieval) | 0.517 | 0.410 |
| Retrieved context (BM25 top-3) | 0.460 | 0.337 |
| Gold context (perfect retrieval) | 0.550 | 0.415 |

The classifier lifts macro-F1 well above the baseline by learning the minority classes,
but no linear condition beats the baseline's *accuracy*, and per-class the reader
collapses on "maybe" (F1 0.20). PubMedQA is built to require reasoning, so this is the
expected ceiling for a bag-of-words head, and it is precisely why the answer step in the
pipeline uses an LLM over the retrieved passages rather than a linear model.

## Design notes

- **Swappable retrieval.** `retrieve.py` puts BM25, dense-LSA-FAISS and a random floor
  behind one interface. Moving to transformer embeddings is a change to one method
  (`DenseRetriever._embed`); the FAISS index and search loop are unchanged.
- **Grounded generation.** `generate.py` instructs the model to answer only from the
  numbered passages, cite them, and return strict JSON, so outputs are checkable.
- **Two API styles.** `ingest_pubmed.py` handles cursor pagination (Europe PMC) and
  token pagination (ClinicalTrials.gov v2); `ingest_opentargets.py` handles a GraphQL
  endpoint. All with retry/backoff and normalisation to a shared record schema.
- **Baselines everywhere.** Nothing is reported without the baseline it has to clear.

## Data and provenance

- **PubMedQA.** Jin et al., *PubMedQA: A Dataset for Biomedical Research Question
  Answering*, EMNLP 2019. Pulled from the official repo
  (github.com/pubmedqa/pubmedqa, MIT). Not redistributed here; `scripts/fetch_data.py`
  downloads it.
- `data/cache/europepmc_sample.json` is reshaped from real PubMedQA records into the
  ingest schema, as an offline example of that path.

## Tests

```bash
python -m pytest -q            # 6 tests: corpus construction, chunking, metric maths, BM25
```

## Licence

MIT. Author: Oluwagbade Odimayo · gbadejosef@gmail.com · github.com/gbadedata
