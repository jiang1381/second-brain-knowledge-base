"""第二大脑 · Web 版（Python 标准库，零重型依赖）。

启动：python app.py  →  http://127.0.0.1:8002

接口：
  GET  /api/meta            基本信息
  GET  /api/docs            笔记列表
  POST /api/ask             提问（RAG）
  POST /api/search          只检索（调试用，不调大模型）
  POST /api/notes           新增笔记
  POST /api/delete          删除笔记
  POST /api/reindex         重建索引
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.config import NOTES_DIR, load_settings
from core.index import KnowledgeBase
from core.llm import LLM, LLMError
from core.rag import Embedder

HOST, PORT = "127.0.0.1", 8002
STATIC_DIR = Path(__file__).parent / "static"


def build():
    settings = load_settings()
    embedder = Embedder(
        mode=settings.embedding_mode, dim=settings.embedding_dim,
        base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    kb = KnowledgeBase(NOTES_DIR, embedder)
    try:
        llm = LLM(settings)
    except LLMError:
        llm = None          # 离线也能用检索，只是不能问答
    return kb, llm, settings


KB, LLM_CLIENT, SETTINGS = build()
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            self._send(200, html.encode("utf-8"), "text/html")
        elif self.path == "/api/meta":
            with LOCK:
                stats = KB.stats()
            self._json(200, {
                "offline": SETTINGS.offline,
                "model": SETTINGS.llm_model,
                "documents": stats["documents"],
                "chunks": stats["chunks"],
                "top_k": SETTINGS.top_k,
                "alpha": SETTINGS.alpha,
            })
        elif self.path == "/api/docs":
            with LOCK:
                docs = KB.list_documents()
            self._json(200, {"documents": docs})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            if self.path == "/api/ask":
                if LLM_CLIENT is None:
                    self._json(400, {"error": "未配置 LLM_API_KEY，只能检索、不能问答。"})
                    return
                data = self._body()
                question = (data.get("question") or "").strip()
                if not question:
                    self._json(400, {"error": "问题不能为空"})
                    return
                alpha = float(data.get("alpha", SETTINGS.alpha))
                top_k = int(data.get("top_k", SETTINGS.top_k))
                t0 = time.time()
                with LOCK:
                    result = KB.ask(LLM_CLIENT, question, top_k, alpha, SETTINGS.min_score)
                result["elapsed"] = round(time.time() - t0, 2)
                self._json(200, result)

            elif self.path == "/api/search":
                data = self._body()
                query = (data.get("query") or "").strip()
                alpha = float(data.get("alpha", SETTINGS.alpha))
                top_k = int(data.get("top_k", SETTINGS.top_k))
                with LOCK:
                    chunks = KB.search(query, top_k, alpha)
                self._json(200, {"chunks": chunks})

            elif self.path == "/api/notes":
                data = self._body()
                title = (data.get("title") or "").strip()
                content = (data.get("content") or "").strip()
                if not title or not content:
                    self._json(400, {"error": "标题和内容都不能为空"})
                    return
                with LOCK:
                    doc_id = KB.add_note(title, content)
                    stats = KB.stats()
                self._json(200, {"id": doc_id, "stats": stats})

            elif self.path == "/api/delete":
                data = self._body()
                with LOCK:
                    ok = KB.delete_note(data.get("id", ""))
                    stats = KB.stats()
                self._json(200, {"ok": ok, "stats": stats})

            elif self.path == "/api/reindex":
                with LOCK:
                    stats = KB.rebuild()
                self._json(200, {"stats": stats})

            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    print("=" * 60)
    print("  🧠 第二大脑已启动：http://127.0.0.1:8002")
    print(f"  笔记：{len(KB.documents)} 篇 | 文本块：{len(KB.index)}")
    print(f"  模式：{'离线（仅检索）' if LLM_CLIENT is None else '在线（' + SETTINGS.llm_model + '）'}")
    print("=" * 60)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
