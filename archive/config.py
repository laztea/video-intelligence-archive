# archive/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_openai = None
_chroma = None

def reset():
    """테스트용: 캐시 초기화."""
    global _openai, _chroma
    _openai = None
    _chroma = None

def data_dir() -> Path:
    p = Path(os.environ.get("DATA_DIR", "./data"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def hybrid_alpha() -> float:
    return float(os.environ.get("HYBRID_ALPHA", "0.5"))

def keyframe_threshold() -> float:
    return float(os.environ.get("KEYFRAME_THRESHOLD", "0.3"))

def _make_openai():
    from openai import OpenAI
    return OpenAI()  # OPENAI_API_KEY 환경변수 사용

def get_openai():
    global _openai
    if _openai is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _openai = _make_openai()
    return _openai

def get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.PersistentClient(path=str(data_dir() / "chroma"))
    return _chroma

def telegram_token() -> str | None:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or None

def telegram_chat_id() -> str | None:
    return os.environ.get("TELEGRAM_CHAT_ID") or None
