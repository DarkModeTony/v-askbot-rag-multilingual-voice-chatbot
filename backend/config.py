import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "allam-2-7b")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "allam-2-7b")
ONLINE_SEARCH_MODEL = os.getenv("ONLINE_SEARCH_MODEL", "groq/compound-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 48))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", 180))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", 0.16))
STRICT_LATENCY_MODE = os.getenv("STRICT_LATENCY_MODE", "true").lower() == "true"
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
LATENCY_FIRST_MODE = os.getenv("LATENCY_FIRST_MODE", "true").lower() == "true"
DATASET_ONLY_MODE = os.getenv("DATASET_ONLY_MODE", "false").lower() == "true"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))
MAX_RETRIEVAL_CHUNKS = int(os.getenv("MAX_RETRIEVAL_CHUNKS", 4))
GUARDRAIL_GROUNDEDNESS_THRESHOLD = float(os.getenv("GUARDRAIL_GROUNDEDNESS_THRESHOLD", 0.45))
TARGET_LATENCY_MS = float(os.getenv("TARGET_LATENCY_MS", 200.0))

DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "msmarco_indic_samples.json")))
STATIC_DIR = BASE_DIR / "frontend"
