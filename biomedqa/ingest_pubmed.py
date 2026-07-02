"""Ingest biomedical text from public REST APIs and normalise it into the corpus schema.

Two sources with deliberately different pagination and payload shapes, to exercise the
kind of schema variability the pipeline has to absorb:

  - Europe PMC   : cursorMark pagination, abstracts under resultType=core.
  - ClinicalTrials.gov v2 : opaque nextPageToken pagination, nested study documents.

Both are keyless for basic use; an api_key argument is threaded through so the same
code works against rate-limited or authenticated deployments. Network calls are wrapped
with exponential backoff. Output records: {source, doc_id, title, text, year}.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterator

import requests

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_TIMEOUT = 30


def _get(url: str, params: dict, *, headers: dict | None = None,
         max_retries: int = 5) -> dict:
    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"GET {url} failed after {max_retries} retries")


def fetch_europepmc(query: str, *, max_records: int = 200, page_size: int = 100,
                    api_key: str | None = None) -> list[dict]:
    """Cursor-paginated search of Europe PMC, returning normalised abstract records."""
    headers = {"User-Agent": "biomedqa/0.1"}
    params_base: dict[str, Any] = {
        "query": query, "format": "json", "resultType": "core",
        "pageSize": min(page_size, 1000),
    }
    if api_key:
        params_base["apiKey"] = api_key

    out: list[dict] = []
    cursor = "*"
    while len(out) < max_records:
        params = dict(params_base, cursorMark=cursor)
        data = _get(EUROPEPMC, params, headers=headers)
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        for r in results:
            abstract = (r.get("abstractText") or "").strip()
            if not abstract:
                continue
            out.append({
                "source": "europepmc",
                "doc_id": r.get("pmid") or r.get("id"),
                "title": (r.get("title") or "").strip(),
                "text": abstract,
                "year": r.get("pubYear"),
            })
            if len(out) >= max_records:
                break
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    return out


def fetch_clinicaltrials(condition: str, *, max_records: int = 200,
                         page_size: int = 100) -> list[dict]:
    """Token-paginated search of ClinicalTrials.gov v2, returning normalised records."""
    fields = [
        "protocolSection.identificationModule.nctId",
        "protocolSection.identificationModule.briefTitle",
        "protocolSection.descriptionModule.briefSummary",
        "protocolSection.statusModule.startDateStruct.date",
    ]
    params_base = {"query.cond": condition, "pageSize": min(page_size, 1000),
                   "fields": "|".join(fields)}
    out: list[dict] = []
    token: str | None = None
    while len(out) < max_records:
        params = dict(params_base)
        if token:
            params["pageToken"] = token
        data = _get(CTGOV, params)
        for study in data.get("studies", []):
            ps = study.get("protocolSection", {})
            ident = ps.get("identificationModule", {})
            summary = ps.get("descriptionModule", {}).get("briefSummary", "").strip()
            if not summary:
                continue
            out.append({
                "source": "clinicaltrials",
                "doc_id": ident.get("nctId"),
                "title": ident.get("briefTitle", "").strip(),
                "text": summary,
                "year": (ps.get("statusModule", {})
                         .get("startDateStruct", {}).get("date")),
            })
            if len(out) >= max_records:
                break
        token = data.get("nextPageToken")
        if not token:
            break
    return out


def to_passages(records: list[dict], chunker: Callable[[str], list[str]]) -> Iterator[dict]:
    """Turn ingested documents into passage records ready for indexing."""
    for rec in records:
        for i, piece in enumerate(chunker(rec["text"])):
            yield {
                "passage_id": f"{rec['doc_id']}#{i}",
                "pmid": str(rec["doc_id"]),
                "section": rec["source"],
                "text": piece,
            }
