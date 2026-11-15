"""核心自测：python tests/test_core.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from core.index import KnowledgeBase
from core.rag import (BM25, Embedder, SearchIndex, chunk_text,
                      hash_embedding, tokenize)


def test_chunking() -> None:
    text = "第一段内容。\n\n第二段内容。" * 40
    chunks = chunk_text(text, size=60, overlap=10)
    assert chunks, "分块不能为空"
    assert all(len(c) <= 120 for c in chunks), "分块长度应受控"


def test_tokenize() -> None:
    toks = tokenize("RAG 检索增强")
    assert "rag" in toks
    assert "检" in toks and "检索" in toks


def test_embedding_similarity() -> None:
    a = hash_embedding("向量检索和余弦相似度", 256)
    b = hash_embedding("向量检索和余弦相似度", 256)
    c = hash_embedding("今天天气很好", 256)
    assert float(a @ b) > 0.99
    assert float(a @ b) > float(a @ c)


def test_bm25() -> None:
    docs = [tokenize("RAG 是检索增强生成"), tokenize("Java 是面向对象语言")]
    bm = BM25(docs)
    q = tokenize("RAG")
    assert bm.score(q, 0) > bm.score(q, 1)


def test_hybrid_search_and_rerank() -> None:
    index = SearchIndex(Embedder("hash", 256))
    index.build([
        {"doc_id": "rag.md", "title": "RAG 原理",
         "text": "RAG 是检索增强生成，能减少幻觉。", "chunk_index": 0},
        {"doc_id": "java.md", "title": "Java 要点",
         "text": "Java 是面向对象的编程语言。", "chunk_index": 0},
    ])
    hits = index.search("RAG 幻觉", top_k=2, alpha=0.6)
    assert hits[0]["doc_id"] == "rag.md", "相关文档应排第一"
    for key in ("vector_score", "bm25_score", "final_score", "coverage", "title_hit"):
        assert key in hits[0], f"缺少可解释字段 {key}"


def test_knowledge_base(tmp: Path) -> None:
    notes = tmp / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "a.md").write_text("# 测试笔记\n\nRAG 是检索增强生成。", encoding="utf-8")

    kb = KnowledgeBase(notes, Embedder("hash", 256))
    assert kb.stats()["documents"] == 1

    kb.add_note("新笔记", "这条笔记讲 Agent 的 ReAct 循环。")
    assert kb.stats()["documents"] == 2

    hits = kb.search("ReAct", 3, 0.6)
    assert hits and "ReAct" in hits[0]["text"]

    assert kb.delete_note("新笔记.md"), "删除应成功"


def main() -> None:
    tmp = ROOT / ".test_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_chunking(); print("✅ 分块")
        test_tokenize(); print("✅ 分词")
        test_embedding_similarity(); print("✅ 向量相似度")
        test_bm25(); print("✅ BM25 关键词检索")
        test_hybrid_search_and_rerank(); print("✅ 混合检索 + 重排")
        test_knowledge_base(tmp); print("✅ 知识库增删查")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n全部测试通过 🎉")


if __name__ == "__main__":
    main()
