"""配置：从 .env 读取模型与检索参数。

检索参数都可以在 .env 里调，方便你做 A/B 对比（比如 alpha 取 0.3 / 0.6 / 0.9）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # second-brain/
DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # 大模型
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    temperature: float = 0.2

    # 向量化
    embedding_mode: str = "hash"          # hash | api
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # 检索
    top_k: int = 5
    alpha: float = 0.6                    # 向量分数权重；1-alpha 给 BM25
    min_score: float = 0.18               # 低于该分数视为「没找到」

    @property
    def offline(self) -> bool:
        return not self.llm_api_key


def load_settings() -> Settings:
    _load_dotenv(BASE_DIR / ".env")
    s = Settings()
    s.llm_base_url = os.getenv("LLM_BASE_URL", s.llm_base_url)
    s.llm_api_key = os.getenv("LLM_API_KEY", s.llm_api_key)
    s.llm_model = os.getenv("LLM_MODEL", s.llm_model)
    s.temperature = float(os.getenv("TEMPERATURE", s.temperature))
    s.embedding_mode = os.getenv("EMBEDDING_MODE", s.embedding_mode)
    s.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", s.embedding_base_url)
    s.embedding_api_key = os.getenv("EMBEDDING_API_KEY", s.embedding_api_key)
    s.embedding_model = os.getenv("EMBEDDING_MODEL", s.embedding_model)
    s.embedding_dim = int(os.getenv("EMBEDDING_DIM", s.embedding_dim))
    s.top_k = int(os.getenv("TOP_K", s.top_k))
    s.alpha = float(os.getenv("ALPHA", s.alpha))
    s.min_score = float(os.getenv("MIN_SCORE", s.min_score))
    return s
