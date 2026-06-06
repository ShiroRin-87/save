"""Gemini API wrapper via OpenAI-compatible endpoint (yunwu.ai)."""
from openai import OpenAI
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS, API_BASE_URL

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=API_BASE_URL + "/v1", timeout=120.0)

# Per-session token tracking
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def generate(prompt: str, *, temperature: float = TEMPERATURE) -> str:
    """Send a prompt to Gemini (via proxy) and return the text response."""
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    _usage["calls"] += 1
    if hasattr(response, "usage") and response.usage:
        _usage["input_tokens"] += response.usage.prompt_tokens or 0
        _usage["output_tokens"] += response.usage.completion_tokens or 0
    return response.choices[0].message.content or ""


def reset_usage() -> None:
    """Reset per-session token counters."""
    for k in _usage:
        _usage[k] = 0


def get_usage() -> dict:
    """Return cumulative token usage since last reset."""
    return dict(_usage)
