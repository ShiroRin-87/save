"""DeepSeek API wrapper via OpenAI-compatible endpoint."""
from openai import OpenAI
from src.config import DS_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS, API_BASE_URL

_client = OpenAI(api_key=DS_API_KEY, base_url=API_BASE_URL, timeout=600.0)

# Per-session token tracking
_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}
_last_reasoning: str | None = None


def generate(prompt: str, *, temperature: float = TEMPERATURE) -> str:
    """Send a prompt to DeepSeek and return the text response."""
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
    msg = response.choices[0].message
    global _last_reasoning
    _last_reasoning = getattr(msg, "reasoning_content", None)
    if _last_reasoning:
        details = response.usage.completion_tokens_details if response.usage else None
        if details:
            _usage["reasoning_tokens"] += details.reasoning_tokens or 0
    return msg.content or ""


def reset_usage() -> None:
    """Reset per-session token counters."""
    for k in _usage:
        _usage[k] = 0


def get_usage() -> dict:
    """Return cumulative token usage since last reset."""
    return dict(_usage)


def get_last_reasoning() -> str | None:
    """Return reasoning_content from the most recent generate() call."""
    return _last_reasoning
