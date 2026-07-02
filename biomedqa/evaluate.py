"""Evaluation for the biomedical RAG pipeline.

Two questions, both answered with real, reproducible numbers on PubMedQA:

1. Retrieval  - does the index surface the right supporting passages?
   Metrics: recall@k, precision@k, MRR, for dense (LSA+FAISS) vs BM25 vs random.

2. Answering  - does retrieved context help decide yes/no/maybe?
   A logistic-regression decision classifier is trained on four feature sets:
   majority baseline, question-only (no retrieval), retrieved-context (BM25 top-k),
   and gold-context (perfect retrieval, the ceiling). Reported as accuracy and macro-F1.

The design mirrors a rule carried from earlier work: a component is only worth using
if it beats a naive baseline, so every number is shown next to the baseline it must clear.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .corpus import Passage, Question, build_corpus, load_pubmedqa
from .retrieve import BM25Retriever, DenseRetriever, RandomRetriever, Retriever

KS = (1, 3, 5, 10)


def retrieval_metrics(retriever: Retriever, questions: list[Question],
                      gold: dict[str, set[str]], ks=KS) -> dict:
    kmax = max(ks)
    recall = {k: [] for k in ks}
    precision = {k: [] for k in ks}
    rr = []
    for q in questions:
        g = gold[q.pmid]
        if not g:
            continue
        ranked = [pid for pid, _ in retriever.search(q.question, k=kmax)]
        # reciprocal rank of the first gold passage
        first = next((r + 1 for r, pid in enumerate(ranked) if pid in g), None)
        rr.append(1.0 / first if first else 0.0)
        for k in ks:
            hits = sum(1 for pid in ranked[:k] if pid in g)
            recall[k].append(hits / len(g))
            precision[k].append(hits / k)
    return {
        "recall_at_k": {k: round(float(np.mean(recall[k])), 4) for k in ks},
        "precision_at_k": {k: round(float(np.mean(precision[k])), 4) for k in ks},
        "mrr": round(float(np.mean(rr)), 4),
        "n_questions": len(rr),
    }


def _classify(train_texts, y_train, test_texts, y_test) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          stop_words="english", max_features=40000)
    xtr = vec.fit_transform(train_texts)
    xte = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
    }


def grounded_classifier_eval(passages: list[Passage], questions: list[Question],
                             gold: dict[str, set[str]], top_k: int = 3,
                             seed: int = 42) -> dict:
    text_by_id = {p.passage_id: p.text for p in passages}
    retr = BM25Retriever().index(passages)

    q_train, q_test = train_test_split(
        questions, test_size=0.30, random_state=seed,
        stratify=[q.decision for q in questions],
    )

    def feats(qs, mode: str):
        out = []
        for q in qs:
            if mode == "question":
                out.append(q.question)
            elif mode == "gold":
                out.append(" ".join(text_by_id[pid] for pid in sorted(gold[q.pmid])))
            elif mode == "retrieved":
                got = [pid for pid, _ in retr.search(q.question, k=top_k)]
                out.append(q.question + " " + " ".join(text_by_id[pid] for pid in got))
            else:
                raise ValueError(mode)
        return out

    y_train = [q.decision for q in q_train]
    y_test = [q.decision for q in q_test]

    # majority-class baseline
    major = Counter(y_train).most_common(1)[0][0]
    maj_pred = [major] * len(y_test)
    baseline = {
        "accuracy": round(float(accuracy_score(y_test, maj_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, maj_pred, average="macro")), 4),
        "predicts": major,
    }

    return {
        "n_train": len(q_train), "n_test": len(q_test), "top_k": top_k,
        "class_distribution": dict(Counter(q.decision for q in questions)),
        "majority_baseline": baseline,
        "question_only": _classify(feats(q_train, "question"), y_train,
                                    feats(q_test, "question"), y_test),
        "retrieved_context": _classify(feats(q_train, "retrieved"), y_train,
                                        feats(q_test, "retrieved"), y_test),
        "gold_context": _classify(feats(q_train, "gold"), y_train,
                                   feats(q_test, "gold"), y_test),
    }


def run(pubmedqa_path: str, out_dir: str = "results") -> dict:
    data = load_pubmedqa(pubmedqa_path)
    passages, gold, questions = build_corpus(data)

    corpus_stats = {
        "n_questions": len(questions),
        "n_passages": len(passages),
        "avg_passages_per_question": round(len(passages) / len(questions), 2),
        "avg_tokens_per_passage": round(
            float(np.mean([len(p.text.split()) for p in passages])), 1),
        "decision_distribution": dict(Counter(q.decision for q in questions)),
    }

    retrievers = [
        DenseRetriever().index(passages),
        BM25Retriever().index(passages),
        RandomRetriever().index(passages),
    ]
    retrieval = {r.name: retrieval_metrics(r, questions, gold) for r in retrievers}
    answering = grounded_classifier_eval(passages, questions, gold, top_k=3)

    results = {"corpus": corpus_stats, "retrieval": retrieval, "answering": answering}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


def _print_summary(r: dict) -> None:
    c = r["corpus"]
    print(f"Corpus: {c['n_questions']} questions, {c['n_passages']} passages "
          f"(~{c['avg_tokens_per_passage']} tokens each). "
          f"Decisions: {c['decision_distribution']}")
    print("\nRetrieval (recall@k / MRR):")
    for name, m in r["retrieval"].items():
        rec = m["recall_at_k"]
        print(f"  {name:16s}  R@1={rec[1]:.3f}  R@5={rec[5]:.3f}  "
              f"R@10={rec[10]:.3f}  MRR={m['mrr']:.3f}")
    a = r["answering"]
    print(f"\nDecision classifier (n_test={a['n_test']}, accuracy / macro-F1):")
    print(f"  majority baseline  acc={a['majority_baseline']['accuracy']:.3f}  "
          f"f1={a['majority_baseline']['macro_f1']:.3f}  (predicts '{a['majority_baseline']['predicts']}')")
    for cond in ("question_only", "retrieved_context", "gold_context"):
        print(f"  {cond:18s} acc={a[cond]['accuracy']:.3f}  f1={a[cond]['macro_f1']:.3f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ori_pqal.json"
    res = run(path)
    _print_summary(res)
