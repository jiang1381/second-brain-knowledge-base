"""知识库：管理笔记文档、构建索引、检索。

对外提供：rebuild / stats / list_documents / add_note / delete_note / search / ask。
"""
from __future__ import annotations

import re
from pathlib import Path

from . import loaders
from .answer import answer as answer_question
from .rag import Embedder, SearchIndex, chunk_text


class KnowledgeBase:
    def __init__(self, notes_dir: Path, embedder: Embedder):
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.index = SearchIndex(embedder)
        self.documents: list[dict] = []
        self.rebuild()

    # ---- 索引 ----
    def rebuild(self) -> dict:
        """重新扫描 notes/、分块、向量化。"""
        self.documents = loaders.load_documents(self.notes_dir)
        chunks: list[dict] = []
        for doc in self.documents:
            for i, text in enumerate(chunk_text(doc["text"])):
                chunks.append({
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": text,
                    "chunk_index": i,
                })
        self.index.build(chunks)
        return self.stats()

    def stats(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.index)}

    def list_documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        for chunk in self.index.chunks:
            counts[chunk["doc_id"]] = counts.get(chunk["doc_id"], 0) + 1
        return [
            {"id": d["id"], "title": d["title"], "chunks": counts.get(d["id"], 0)}
            for d in self.documents
        ]

    # ---- 笔记增删 ----
    def add_note(self, title: str, content: str) -> str:
        safe = re.sub(r"[^\w\-]+", "_", title).strip("_") or "note"
        path = self.notes_dir / f"{safe}.md"
        path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
        self.rebuild()
        return str(path.relative_to(self.notes_dir)).replace("\\", "/")

    def delete_note(self, doc_id: str) -> bool:
        base = self.notes_dir.resolve()
        path = (self.notes_dir / doc_id).resolve()
        if base != path and not str(path).startswith(str(base) + "\\"):
            return False                      # 防路径穿越
        if path.exists() and path.is_file():
            path.unlink()
            self.rebuild()
            return True
        return False

    # ---- 检索与问答 ----
    def search(self, query: str, top_k: int, alpha: float) -> list[dict]:
        return self.index.search(query, top_k=top_k, alpha=alpha)

    def ask(self, llm, question: str, top_k: int, alpha: float,
            min_score: float) -> dict:
        chunks = self.search(question, top_k, alpha)
        result = answer_question(llm, question, chunks, min_score)
        result["chunks"] = chunks
        return result
