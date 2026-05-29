"""Shared utilities: JSON I/O, retry, parsing, formatting."""
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).parent.parent


def load_json(path: str) -> Any:
    with open(ROOT_DIR / path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    p = ROOT_DIR / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def retry(func: Callable, max_retries: int = 3, delay: float = 1.0) -> Any:
    """Retry a callable on exception, with linear backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exc


def format_search_results(results: list[dict]) -> str:
    """Format cached search results for prompt insertion with rank markers."""
    parts = []
    for r in results:
        text = r.get("full_text", "") or r.get("snippet", "")
        parts.append(f"[{r['rank']}] {r['title']}\n{text}\nURL: {r['url']}")
    return "\n\n".join(parts)


def format_search_results_numbered(results: list[dict]) -> str:
    """Format search results with numbered entries for citation-aware prompts."""
    parts = []
    for r in results:
        text = r.get("full_text", "") or r.get("snippet", "")
        parts.append(f"[{r['rank']}] {text}")
    return "\n\n".join(parts)


def parse_structured_table(text: str) -> list[dict]:
    """Parse SEVE structured extraction output into list of claim dicts.

    Expected format per line: N | claim_text | original_snippet | url [conflict]
    Returns list of {id, claim, snippet, url, has_conflict}.
    """
    claims = []
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(https?://\S+)",
            line,
        )
        if match:
            claim_id = int(match.group(1))
            claim_text = match.group(2).strip()
            snippet = match.group(3).strip()
            url = match.group(4).strip()
            has_conflict = "[冲突]" in line
            claims.append({
                "id": claim_id,
                "claim": claim_text,
                "snippet": snippet,
                "url": url,
                "has_conflict": has_conflict,
            })
    return claims


def extract_claims_from_answer(answer: str) -> list[str]:
    """Split an answer into individual claims (sentence-level)."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    return [s.strip() for s in sentences if s.strip()]


def load_questions() -> list[dict]:
    return load_json("data/freshqa_questions.json")


def load_search_cache() -> dict:
    return load_json("data/search_cache.json")
