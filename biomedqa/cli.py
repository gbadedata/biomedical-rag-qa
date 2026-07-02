"""Command-line entry point for the biomedqa toolkit.

Examples
--------
  python -m biomedqa.cli eval --data data/ori_pqal.json
  python -m biomedqa.cli ask --data data/ori_pqal.json --k 5 \
      --question "Does metformin reduce cancer risk in type 2 diabetes?"
  python -m biomedqa.cli fetch-europepmc --query "psoriasis biologics" --max 100 --out out.json
  python -m biomedqa.cli fetch-targets --efo EFO_0000305 --out targets.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .corpus import build_corpus, chunk_text, load_pubmedqa
from .retrieve import BM25Retriever, DenseRetriever


def _cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import run, _print_summary
    _print_summary(run(args.data, out_dir=args.out))
    print(f"\nWrote {args.out}/evaluation.json")


def _cmd_ask(args: argparse.Namespace) -> None:
    passages, _, _ = build_corpus(load_pubmedqa(args.data))
    retriever = (DenseRetriever() if args.retriever == "dense" else BM25Retriever())
    retriever.index(passages)
    by_id = {p.passage_id: p for p in passages}
    hits = retriever.search(args.question, k=args.k)
    top = [by_id[pid] for pid, _ in hits]

    print(f"Question: {args.question}\nRetriever: {retriever.name}\n")
    print("Retrieved passages:")
    for i, (pid, score) in enumerate(hits, 1):
        print(f"  [{i}] {pid} ({by_id[pid].section}, score {score:.3f})")
        print(f"      {by_id[pid].text[:160]}...")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from .generate import answer
        ans = answer(args.question, top)
        print(f"\nDecision: {ans.decision}\nJustification: {ans.justification}")
        print(f"Cited passages: {ans.supporting_passages}")
    else:
        print("\n[ANTHROPIC_API_KEY not set: retrieval shown, generation skipped.]")


def _write(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"Wrote {len(records)} records to {path}")


def _cmd_fetch_europepmc(args: argparse.Namespace) -> None:
    from .ingest_pubmed import fetch_europepmc
    _write(args.out, fetch_europepmc(args.query, max_records=args.max))


def _cmd_fetch_ctgov(args: argparse.Namespace) -> None:
    from .ingest_pubmed import fetch_clinicaltrials
    _write(args.out, fetch_clinicaltrials(args.condition, max_records=args.max))


def _cmd_fetch_targets(args: argparse.Namespace) -> None:
    from .ingest_opentargets import associated_targets
    _write(args.out, associated_targets(args.efo, max_rows=args.max))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="biomedqa")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="run retrieval + answering evaluation")
    pe.add_argument("--data", required=True)
    pe.add_argument("--out", default="results")
    pe.set_defaults(func=_cmd_eval)

    pa = sub.add_parser("ask", help="retrieve (and, with a key, answer) a question")
    pa.add_argument("--data", required=True)
    pa.add_argument("--question", required=True)
    pa.add_argument("--k", type=int, default=5)
    pa.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    pa.set_defaults(func=_cmd_ask)

    pp = sub.add_parser("fetch-europepmc", help="ingest abstracts from Europe PMC")
    pp.add_argument("--query", required=True)
    pp.add_argument("--max", type=int, default=200)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=_cmd_fetch_europepmc)

    pc = sub.add_parser("fetch-ctgov", help="ingest study summaries from ClinicalTrials.gov")
    pc.add_argument("--condition", required=True)
    pc.add_argument("--max", type=int, default=200)
    pc.add_argument("--out", required=True)
    pc.set_defaults(func=_cmd_fetch_ctgov)

    pt = sub.add_parser("fetch-targets", help="fetch disease-target links from Open Targets")
    pt.add_argument("--efo", required=True)
    pt.add_argument("--max", type=int, default=50)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=_cmd_fetch_targets)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
