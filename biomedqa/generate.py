"""Grounded answer generation over retrieved passages, via the Anthropic API.

This is the generation half of RAG. It is deliberately thin and fully runnable:
give it a question and the passages a retriever returned, and it asks the model for a
yes / no / maybe decision with a short justification that must be grounded in, and cite,
the supplied passages. It never answers from parametric memory alone.

Requires ANTHROPIC_API_KEY in the environment. The model name is read from
ANTHROPIC_MODEL so the pipeline tracks whichever current model you choose.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .corpus import Passage

SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. Answer the clinical/biological "
    "question using ONLY the numbered passages provided. Decide yes, no, or maybe. "
    "Use 'maybe' when the passages are mixed or insufficient. Do not use outside "
    "knowledge and do not speculate beyond the passages. Reply as strict JSON with keys "
    '"decision" (one of yes|no|maybe), "justification" (<= 3 sentences), and '
    '"supporting_passages" (a list of the passage numbers you relied on).'
)


@dataclass
class RagAnswer:
    decision: str
    justification: str
    supporting_passages: list[int]
    raw: str


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = [f"[{i + 1}] ({p.section}) {p.text}" for i, p in enumerate(passages)]
    context = "\n\n".join(blocks)
    return f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer as JSON."


def _parse(text: str) -> RagAnswer:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    payload = json.loads(match.group(0)) if match else {}
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"yes", "no", "maybe"}:
        decision = "maybe"
    return RagAnswer(
        decision=decision,
        justification=str(payload.get("justification", "")).strip(),
        supporting_passages=[int(x) for x in payload.get("supporting_passages", [])
                             if str(x).isdigit()],
        raw=text,
    )


def answer(question: str, passages: list[Passage], *, max_tokens: int = 400,
           model: str | None = None) -> RagAnswer:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the generation step") from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY to run the generation step.")

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return _parse(text)
