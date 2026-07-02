# biomedical-rag-qa

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![tests](https://github.com/gbadedata/biomedical-rag-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/gbadedata/biomedical-rag-qa/actions/workflows/tests.yml)

A reusable toolkit for **retrieval-augmented question answering over biomedical
literature**. It ingests abstracts from public APIs, indexes them for search, retrieves
the passages that bear on a clinical or biological question, and generates a grounded
yes / no / maybe answer with a language model. Every stage is a separate, testable
module, and the retrieval and answering steps are benchmarked against naive baselines on
the public PubMedQA dataset, so the system's behaviour is measured rather than assumed.

> **Headline results (PubMedQA, 1,000 labelled questions).** BM25 retrieval places a
> supporting passage first for **94.3%** of questions (MRR **0.959**) and beats a dense
> FAISS index on every metric. A linear decision reader, by contrast, cannot beat the
> majority-class accuracy and collapses on ambiguous cases, which is the empirical reason
> the pipeline answers with an LLM over the retrieved evidence rather than a classifier.

---

## Contents

- [Overview](#overview)
- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [The data](#the-data)
- [Method](#method)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Limitations and how far to trust the results](#limitations-and-how-far-to-trust-the-results)
- [Reproducing the evaluation](#reproducing-the-evaluation)
- [Project structure](#project-structure)
- [Tests](#tests)
- [References](#references)
- [License](#license)

---

## Overview

Retrieval-augmented generation (RAG) is the standard pattern for answering questions over
a body of documents without letting a model invent facts: retrieve the passages that are
actually relevant, then make the model answer from those, with citations. This repository
is a compact, transparent implementation of that pattern for biomedical text, with the
evaluation attached.

The pipeline has five stages, each an importable module:

```
ingest ─▶ corpus ─▶ retrieve ─▶ generate ─▶ evaluate
```

It ships with a command-line interface, unit tests, and a full evaluation on PubMedQA
whose numbers are reproduced under `results/`.

---

## Why this exists

Medical and life-science teams hold large volumes of unstructured text: abstracts, trial
records, product labels, congress material. The recurring request is to query that text in
natural language and get an answer that can be trusted and traced back to a source.

RAG addresses the two failure modes of asking a bare language model. It grounds the answer
in retrieved evidence rather than parametric memory, which reduces fabrication, and it
makes the answer auditable, because the passages it used are shown. The engineering
question is then not "can a model answer" but two measurable sub-questions: does retrieval
surface the right evidence, and does that evidence actually improve the answer. This
project is built to answer both with numbers, on real biomedical questions, which is
exactly the discipline needed before any such system is put in front of a client or a
clinician.

---

## Architecture

The system is a linear pipeline of small, single-responsibility modules. Each can be used,
tested, and replaced on its own.

| Stage | Module | What it does | Why it is built this way |
|---|---|---|---|
| Ingest | `ingest_pubmed.py` | Pulls abstracts from Europe PMC (REST, cursor pagination) and ClinicalTrials.gov v2 (REST, token pagination), with retry and backoff | Two different pagination and payload shapes exercise the schema variability real data layers must absorb |
| Ingest | `ingest_opentargets.py` | Queries the Open Targets Platform (GraphQL) for disease-to-target associations | Adds a structured biology layer and a second API style (GraphQL) alongside REST |
| Corpus | `corpus.py` | Loads PubMedQA, turns each abstract into passages, and builds the gold map; includes a sliding-window chunker for free-form documents | Clean separation of "documents in" from "how they are indexed" |
| Retrieve | `retrieve.py` | Three retrievers behind one interface: BM25, dense (LSA + FAISS), and a random floor | One interface means the retriever is a swappable component, not a hard-wired choice |
| Generate | `generate.py` | Calls the Anthropic API to produce a grounded yes / no / maybe answer that must cite the passages and return strict JSON | Grounding and a machine-checkable output contract make the answer auditable |
| Evaluate | `evaluate.py` | Computes retrieval metrics (recall@k, precision@k, MRR) and an answering diagnostic against baselines | Nothing is reported without the baseline it must clear |
| Interface | `cli.py` | Exposes `eval`, `ask`, and the three ingest commands | The toolkit is usable from the shell, not only as a library |

---

## Installation

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # downloads PubMedQA ori_pqal.json into data/
```

A 40-item sample (`data/sample_pqal.json`) is committed for quick tests and a first look,
so the download step is optional to get started.

---

## Usage

```bash
# Retrieve passages for a question (and, if ANTHROPIC_API_KEY is set, generate an answer)
python -m biomedqa.cli ask --data data/ori_pqal.json --retriever bm25 --k 5 \
    --question "Does metformin reduce cancer risk in type 2 diabetes?"

# Reproduce the full evaluation (writes results/evaluation.json)
python -m biomedqa.cli eval --data data/ori_pqal.json

# Ingest fresh text from public APIs
python -m biomedqa.cli fetch-europepmc --query "psoriasis biologics" --max 100 --out epmc.json
python -m biomedqa.cli fetch-ctgov     --condition "atrial fibrillation" --max 100 --out ctgov.json
python -m biomedqa.cli fetch-targets   --efo EFO_0000305 --out targets.json   # breast carcinoma
```

The generation step reads `ANTHROPIC_API_KEY` and, optionally, `ANTHROPIC_MODEL` (set it
to a current model name). Without a key, `ask` still runs retrieval and prints the
passages, and the generation step is skipped. The evaluation numbers below do not depend
on any model call.

---

## The data

**PubMedQA** (Jin et al., 2019) provides 1,000 expert-labelled biomedical questions. Each
question is paired with the PubMed abstract it was written from, already split into
labelled sections (BACKGROUND, METHODS, RESULTS, and so on), plus a yes / no / maybe
decision and a long-form answer.

The toolkit turns this into a retrieval corpus by treating each abstract section as one
passage. The gold supporting passages for a question are that question's own sections,
which gives a reproducible retrieval benchmark on genuine text. After construction:

- **1,000 questions**, **3,358 passages**, about **60 tokens** per passage
- Decision labels: **yes 552, no 338, maybe 110**, so a majority-class baseline sits at
  **55.3%** accuracy
- On average **3.36 gold passages per question**

The full file is not redistributed here; `scripts/fetch_data.py` downloads it from the
official repository (MIT licensed).

---

## Method

**Building the retrieval benchmark.** Every question's gold set is the passages from its
own abstract. Retrieval is then scored by how well a retriever ranks a question's gold
passages against the pool of all 3,358 passages. This is a standard, defensible way to derive
a retrieval benchmark from PubMedQA, and it lets three retrievers be compared on identical
ground.

**Retrievers.** `retrieve.py` implements three behind one interface. BM25 is the lexical
baseline. The dense retriever builds TF-IDF vectors, reduces them with truncated SVD
(latent semantic analysis), L2-normalises them, and indexes them in FAISS with
inner-product search, which equals cosine similarity on normalised vectors. A random
retriever provides the floor any real method must clear. The dense path is deliberately
swappable: replacing the vectoriser with a biomedical transformer embedder is a change to
one method, and the FAISS index and search loop stay the same.

**Grounded generation.** `generate.py` instructs the model to decide yes / no / maybe
using only the numbered passages, to cite the passages it used, and to return strict JSON.
This makes the output auditable and machine-checkable, and it prevents the model from
answering from training memory.

**Answering diagnostic.** To measure whether retrieved context actually helps answer the
question, a logistic-regression decision head is trained on four feature sets and each is
reported next to the majority-class baseline: question only (no retrieval), retrieved
context (BM25 top-3), gold context (perfect retrieval, the ceiling), and the baseline
itself. The point of this design is to isolate the contribution of retrieval to the answer.

---

## Results

### Retrieval: does the index find the right supporting passages?

| Retriever | Precision@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| **BM25 (lexical)** | **0.943** | **0.673** | **0.731** | **0.959** |
| Dense (LSA + FAISS) | 0.802 | 0.602 | 0.687 | 0.846 |
| Random baseline | 0.000 | 0.001 | 0.002 | 0.001 |

**Interpretation.** BM25 puts a gold passage in first place for 94.3% of questions, and
its MRR of 0.959 means the first relevant passage is almost always rank 1 or 2. It beats
the dense LSA index on every metric. The reason is that PubMedQA questions are written from
their source abstracts, so they share a great deal of vocabulary with the passages that
answer them, and lexical matching is very hard to beat. Compressing TF-IDF into 256 LSA
dimensions trades some of that precision for a semantic generalisation the benchmark does
not reward. Both real retrievers beat the random floor by a wide margin, which confirms the
metrics and corpus are wired correctly.

Note on recall@1: because each question has about 3.4 gold passages, recall@1 is
mathematically capped at a mean of **0.319** (you can retrieve at most one passage at rank
1). BM25's recall@1 of 0.300 is therefore 94% of the achievable ceiling, which is another
way of seeing that its top rank is almost always correct.

**Implication.** Do not assume "dense beats sparse", measure it. The credible production
move here is a biomedical transformer embedder (for example PubMedBERT or an embedding
API) swapped into the dense retriever and re-measured against this BM25 baseline, not LSA.
On this closed corpus retrieval is close to its structural ceiling, so the real headroom
is the open-domain case (retrieving from millions of unrelated abstracts), where a stronger
embedder and a re-ranking stage earn their keep.

### Answering: does retrieved context help decide yes / no / maybe?

Logistic-regression decision head, held-out test set of 300 questions, reported next to the
majority-class baseline.

| Feature set | Accuracy | Macro-F1 |
|---|---|---|
| Majority baseline (always "yes") | 0.553 | 0.237 |
| Question only (no retrieval) | 0.517 | 0.410 |
| Retrieved context (BM25 top-3) | 0.460 | 0.337 |
| Gold context (perfect retrieval) | 0.550 | 0.415 |

Per-class F1 under gold context: **yes 0.672, no 0.374, maybe 0.200**.

**Interpretation.** The classifier lifts macro-F1 far above the baseline (0.237 to 0.41)
by learning the minority classes the baseline ignores entirely, but no linear condition
beats the baseline's accuracy of 0.553, and per class the reader collapses on the
ambiguous "maybe" class (F1 0.20). Even perfect retrieval only marginally beats
question-only for such a reader, and naively concatenating retrieved passages actually
hurt it. This is expected: PubMedQA is deliberately built to require reasoning over
evidence, and a bag-of-words linear model cannot reason.

**Implication.** Retrieval quality is necessary but not sufficient. The value of RAG
appears only with a capable reader, which is precisely why the pipeline's answer step is
an LLM over the retrieved passages rather than a linear head. It also shows that how the
context is assembled is its own engineering problem, not an afterthought.

### What this tells us

On this benchmark retrieval is close to solved by a strong lexical baseline, and the open
problem is the reader. For a product, the parts that matter are the ones this repository
actually delivers: a modular, swappable, measured pipeline, a grounded generation contract
that forces citations and strict output, and an ingestion layer that copes with real API
messiness. A clever retriever is worth less than a system whose every stage can be measured
and replaced.

---

## Design decisions

- **Swappable retrieval.** BM25, dense LSA + FAISS, and a random floor sit behind one
  interface. Moving to transformer embeddings is a change to `DenseRetriever._embed`
  alone; the index and search loop are unchanged.
- **Grounded, checkable generation.** The model must answer only from the numbered
  passages, cite them, and return strict JSON, so answers can be validated automatically.
- **Two API styles, one schema.** REST with two pagination strategies (Europe PMC cursor,
  ClinicalTrials.gov token) and GraphQL (Open Targets), all normalised to a shared record
  schema, with retry and backoff.
- **Baselines everywhere.** Every reported number sits next to the baseline it must clear,
  including a random-retrieval floor and a majority-class floor.
- **Metrics independent of the LLM.** The reported evaluation uses retrieval metrics and a
  linear diagnostic, so the numbers are fully reproducible without an API key. The
  generation step is implemented and runnable, but its accuracy is not claimed here.

---

## Limitations and how far to trust the results

- **Closed-corpus benchmark.** Because questions come from their own abstracts, retrieval
  is easier here than in open-domain use. Trust the relative ordering (BM25 above dense LSA,
  both far above random) and the "near ceiling" conclusion; treat the absolute recall
  figures as optimistic for open-domain.
- **The dense retriever is LSA, not transformer embeddings.** The build environment could
  not download model weights, so the dense path uses SVD-reduced TF-IDF. The
  dense-versus-lexical result is therefore specific to LSA; a biomedical transformer could
  change it. The code is structured for that swap.
- **The answering number is a diagnostic, not the LLM.** It measures whether the decision
  is linearly separable from the text, not the LLM's actual accuracy.
- **Label semantics.** The yes / no / maybe labels reflect each source abstract's own
  stated conclusion, not ground truth about the world.

**Net:** trust the structural conclusions, that retrieval is strong and lexically driven on
this corpus, that a linear reader cannot solve the decision, and that the design
consequence is an LLM reader over measured retrieval. Treat absolute retrieval numbers as
corpus-specific.

---

## Reproducing the evaluation

```bash
python scripts/fetch_data.py                       # PubMedQA into data/
python -m biomedqa.cli eval --data data/ori_pqal.json
cat results/evaluation.json                         # the numbers above
```

The evaluation is deterministic (fixed seeds), so a rerun reproduces the reported figures.

---

## Project structure

```
biomedical-rag-qa/
├── biomedqa/
│   ├── corpus.py             # PubMedQA loader, passage + gold construction, chunker
│   ├── retrieve.py           # BM25, dense LSA + FAISS, random retrievers (one interface)
│   ├── generate.py           # grounded Anthropic-API answerer (yes/no/maybe + citations)
│   ├── ingest_pubmed.py      # Europe PMC (REST) and ClinicalTrials.gov v2 (REST)
│   ├── ingest_opentargets.py # Open Targets (GraphQL)
│   ├── evaluate.py           # retrieval metrics + answering diagnostic vs baselines
│   └── cli.py                # eval, ask, fetch-* commands
├── tests/                    # 6 unit tests
├── scripts/fetch_data.py     # downloads PubMedQA
├── data/                     # committed sample + API-shaped sample
├── results/evaluation.json   # reproduced metrics
├── requirements.txt
└── LICENSE
```

---

## Tests

```bash
python -m pytest -q      # 6 tests: corpus construction, chunking, metric maths, BM25 retrieval
```

Run automatically on push via GitHub Actions across Python 3.10, 3.11 and 3.12.

---

## References

- Jin, Q., Dhingra, B., Liu, Z., Cohen, W., Lu, X. (2019). *PubMedQA: A Dataset for
  Biomedical Research Question Answering.* EMNLP 2019.
  Data: https://github.com/pubmedqa/pubmedqa (MIT licensed).
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and
  Beyond.*
- Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale similarity search with GPUs
  (FAISS).*
- Ochoa, D. et al. (2023). *The Open Targets Platform.* Nucleic Acids Research.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
