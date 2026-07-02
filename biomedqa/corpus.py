"""Load PubMedQA and turn it into a retrieval corpus.

PubMedQA (Jin et al., 2019) gives, per question, the abstract it was written from,
already split into labelled sections (BACKGROUND, METHODS, RESULTS, ...). We treat
each section as one passage. The gold supporting passages for a question are that
question's own sections, which gives a reproducible retrieval benchmark on real text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class Passage:
    passage_id: str      # f"{pmid}#{i}"
    pmid: str
    section: str         # section label, e.g. RESULTS
    text: str


@dataclass(frozen=True)
class Question:
    pmid: str
    question: str
    decision: str        # yes / no / maybe
    long_answer: str
    year: str | None


def load_pubmedqa(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_corpus(data: dict) -> tuple[list[Passage], dict[str, set[str]], list[Question]]:
    """Return (passages, gold_map, questions).

    gold_map[pmid] is the set of passage_ids that belong to that pmid (its gold context).
    """
    passages: list[Passage] = []
    gold: dict[str, set[str]] = {}
    questions: list[Question] = []
    for pmid, rec in data.items():
        contexts = rec.get("CONTEXTS", []) or []
        labels = rec.get("LABELS", []) or []
        ids: set[str] = set()
        for i, ctx in enumerate(contexts):
            section = labels[i] if i < len(labels) else "UNLABELLED"
            pid = f"{pmid}#{i}"
            passages.append(Passage(pid, pmid, section, ctx.strip()))
            ids.add(pid)
        gold[pmid] = ids
        questions.append(
            Question(
                pmid=pmid,
                question=rec["QUESTION"].strip(),
                decision=rec["final_decision"].strip().lower(),
                long_answer=rec.get("LONG_ANSWER", "").strip(),
                year=str(rec.get("YEAR")) if rec.get("YEAR") else None,
            )
        )
    return passages, gold, questions


_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def chunk_text(text: str, max_tokens: int = 220, overlap: int = 40) -> list[str]:
    """Sliding-window chunker for free-form documents pulled from external APIs.

    PubMedQA is already pre-chunked into sections, so this is used by the ingest path
    when abstracts arrive as a single block of text.
    """
    toks = text.split()
    if len(toks) <= max_tokens:
        return [text.strip()] if text.strip() else []
    out: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(toks), step):
        piece = " ".join(toks[start:start + max_tokens]).strip()
        if piece:
            out.append(piece)
        if start + max_tokens >= len(toks):
            break
    return out


def passages_to_records(passages: Iterable[Passage]) -> list[dict]:
    return [asdict(p) for p in passages]
