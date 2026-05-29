"""Gemini API wrapper via OpenAI-compatible endpoint (yunwu.ai)."""
from openai import OpenAI
from src.config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_OUTPUT_TOKENS, API_BASE_URL

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=API_BASE_URL + "/v1", timeout=120.0)


def generate(prompt: str, *, temperature: float = TEMPERATURE) -> str:
    """Send a prompt to Gemini (via proxy) and return the text response."""
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""
