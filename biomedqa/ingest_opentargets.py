"""Enrich the corpus with structured biology from the Open Targets Platform GraphQL API.

RAG over abstracts answers "what does the literature say". Open Targets adds a structured
layer: for a disease it returns the genes/targets most strongly associated with it, which
can seed queries, filter a corpus, or ground an answer in curated evidence.

This exercises a GraphQL endpoint (single POST, typed query, page:{index,size}
pagination) alongside the REST sources in ingest_pubmed, so the toolkit handles both
API styles. Endpoint is keyless.
"""
from __future__ import annotations

import time

import requests

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_TIMEOUT = 30

ASSOCIATED_TARGETS = """
query AssociatedTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        score
        target { id approvedSymbol approvedName }
      }
    }
  }
}
"""


def _post(query: str, variables: dict, *, max_retries: int = 5) -> dict:
    delay = 1.0
    for _ in range(max_retries):
        resp = requests.post(ENDPOINT, json={"query": query, "variables": variables},
                             timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            body = resp.json()
            if "errors" in body:
                raise RuntimeError(f"GraphQL errors: {body['errors']}")
            return body["data"]
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        resp.raise_for_status()
    raise RuntimeError("Open Targets query failed after retries")


def associated_targets(efo_id: str, *, max_rows: int = 50, page_size: int = 25) -> list[dict]:
    """Return normalised target-association rows for a disease (by EFO id).

    Example efo_id: 'EFO_0000305' (breast carcinoma).
    """
    out: list[dict] = []
    index = 0
    disease_name = None
    while len(out) < max_rows:
        data = _post(ASSOCIATED_TARGETS,
                     {"efoId": efo_id, "index": index, "size": page_size})
        disease = data.get("disease") or {}
        disease_name = disease.get("name")
        block = disease.get("associatedTargets", {}) or {}
        rows = block.get("rows", [])
        if not rows:
            break
        for row in rows:
            target = row.get("target", {})
            out.append({
                "disease_id": efo_id,
                "disease": disease_name,
                "target_id": target.get("id"),
                "symbol": target.get("approvedSymbol"),
                "target_name": target.get("approvedName"),
                "association_score": round(float(row.get("score", 0.0)), 4),
            })
            if len(out) >= max_rows:
                break
        if (index + 1) * page_size >= block.get("count", 0):
            break
        index += 1
    return out
