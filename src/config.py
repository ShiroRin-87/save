"""Experiment configuration — all values are fixed per the experiment design."""

# API keys
DS_API_KEY = "sk-2d2f9c783efa49c5b4b501dad0b54107"
SERP_API_KEY = "0848cf62ae0bbdecf5d293937984dc9695234cfb3ec0da929a1d94d7d7b5a45d"

# Model
MODEL_NAME = "deepseek-v4-pro"
API_BASE_URL = "https://api.deepseek.com"

# Search
SEARCH_TOP_K = 10
SERPAPI_ENDPOINT = "https://serpapi.com/search"
JINA_MAX_CHARS = 5000  # per-page max chars (per-result extraction, simple truncation)

# Generation
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 65536
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
