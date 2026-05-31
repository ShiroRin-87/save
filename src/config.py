"""Experiment configuration — all values are fixed per the experiment design."""

# API keys
GEMINI_API_KEY = "sk-QRv4ACt4fQub9e2_Jtftp4__Tf6_y3RY69gUKLSo6GjKjnGUiV4QHNVa8io"
SERP_API_KEY = "871721eb9798e30a558f35e2267167e621537ebba654e296459c93c3548fcf33"

# Model
MODEL_NAME = "gemini-3-flash-preview"
API_BASE_URL = "https://www.ai-wave.org"

# Search
SEARCH_TOP_K = 10
SERPAPI_ENDPOINT = "https://serpapi.com/search"

# Generation
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4096
API_TIMEOUT_SEC = 60

# Parsing
PARSE_RETRY_MAX = 3

# Evaluation
BOOTSTRAP_SAMPLES = 10000
ALPHA = 0.05

# Paths (relative to project root)
DATA_DIR = "data"
FRESHQA_FILE = "data/freshqa_questions.json"
SEARCH_CACHE_FILE = "data/search_cache.json"
OUTPUTS_DIR = "outputs"
