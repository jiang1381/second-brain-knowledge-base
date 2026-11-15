"""检索核心：分块 → 向量化 → 混合检索（向量 + BM25）→ 重排。

三个进阶点（也是 RAG 面试的深水区）：
1. 混合检索：向量看「语义」，BM25 看「关键词」，两者互补；
2. 重排：先粗召回 3*top_k，再按「查询词覆盖率 + 标题命中」精排；
3. 可解释：每个结果都带 vector / bm25 / coverage / final 分数，方便调试展示。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# 1) 分块
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    """按段落/句子切分，合并到约 size 长度，块间保留 overlap。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        sentences = (re.split(r"(?<=[。！？.!?])", para)
                     if len(para) > size else [para])
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(buf) + len(sent) > size and buf:
                chunks.append(buf)
                buf = buf[-overlap:] if overlap > 0 else ""
            buf += sent
    if buf.strip():
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# 2) 分词 + 向量化
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """英文按单词，中文按「单字 + 二元组」。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(list(seg))
        tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def hash_embedding(text: str, dim: int = 1024) -> np.ndarray:
    """哈希向量化（离线可用，本质是 hashing TF）。"""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 17) & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Embedder:
    """向量化器：hash（离线）或 api（真实 embedding 接口）。"""

    def __init__(self, mode: str = "hash", dim: int = 1024,
                 base_url: str = "", api_key: str = "", model: str = ""):
        self.mode = mode
        self.dim = dim
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def __call__(self, text: str) -> np.ndarray:
        if self.mode == "api":
            return self._api_embed(text)
        return hash_embedding(text, self.dim)

    def _api_embed(self, text: str) -> np.ndarray:
        url = self.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": text}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# 3) BM25（关键词检索）
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1

    def score(self, query: list[str], i: int) -> float:
        if self.N == 0 or self.avgdl == 0:
            return 0.0
        total = 0.0
        tf, dl = self.tf[i], self.doc_len[i]
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * freq * (self.k1 + 1) / denom
        return total


# ---------------------------------------------------------------------------
# 4) 索引 + 混合检索 + 重排
# ---------------------------------------------------------------------------
class SearchIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[dict] = []
        self.vectors: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
        self.bm25: BM25 | None = None
        self._tokens: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._tokens = [tokenize(c["text"]) for c in chunks]
        if chunks:
            self.vectors = np.vstack([self.embedder(c["text"]) for c in chunks])
        else:
            self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.bm25 = BM25(self._tokens)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5, alpha: float = 0.6) -> list[dict]:
        """混合检索：alpha 给向量，1-alpha 给 BM25，再重排。"""
        if not self.chunks:
            return []

        # 向量分数（已归一化，点积即余弦），映射到 0~1
        q_vec = self.embedder(query)
        vec_raw = self.vectors @ q_vec
        vec01 = (vec_raw + 1.0) / 2.0

        # BM25 分数，按本轮最大值归一化
        q_tokens = tokenize(query)
        bm = [self.bm25.score(q_tokens, i) for i in range(len(self.chunks))]
        bm_max = max(bm) if bm and max(bm) > 0 else 0.0
        bm01 = [s / bm_max if bm_max > 0 else 0.0 for s in bm]

        combined = [alpha * float(v) + (1 - alpha) * float(b)
                    for v, b in zip(vec01, bm01)]

        # 粗召回：取 3*top_k 再重排
        order = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
        candidates = []
        for i in order[: max(top_k * 3, top_k)]:
            chunk = dict(self.chunks[i])
            chunk.update({
                "vector_score": round(float(vec01[i]), 3),
                "bm25_score": round(float(bm01[i]), 3),
                "combined_score": round(float(combined[i]), 3),
            })
            candidates.append(chunk)

        return self._rerank(query, candidates)[:top_k]

    @staticmethod
    def _rerank(query: str, candidates: list[dict]) -> list[dict]:
        """精排：在混合分数基础上，加查询词覆盖率和标题命中。"""
        q = set(tokenize(query))
        for c in candidates:
            doc_tokens = set(tokenize(c["text"]))
            coverage = len(q & doc_tokens) / len(q) if q else 0.0
            title_tokens = set(tokenize(c.get("title", "")))
            title_hit = 1.0 if (q & title_tokens) else 0.0
            c["coverage"] = round(coverage, 3)
            c["title_hit"] = title_hit
            c["final_score"] = round(
                c["combined_score"] + 0.20 * coverage + 0.10 * title_hit, 3)
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates
