"""第二大脑 · 命令行版。

用法：
    python cli.py list
    python cli.py search "RAG 怎么减少幻觉"
    python cli.py ask "我上次面试复盘里写了什么"
    python cli.py add "标题" --file note.md
    python cli.py add "标题"          # 从标准输入读内容
    python cli.py delete "某篇.md"
    python cli.py reindex
"""
from __future__ import annotations

import argparse
import sys

from core.config import NOTES_DIR, load_settings
from core.index import KnowledgeBase
from core.llm import LLM, LLMError
from core.rag import Embedder


def build_kb():
    settings = load_settings()
    embedder = Embedder(
        mode=settings.embedding_mode, dim=settings.embedding_dim,
        base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    return KnowledgeBase(NOTES_DIR, embedder), settings


def main() -> None:
    parser = argparse.ArgumentParser(description="第二大脑 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出笔记")
    p = sub.add_parser("search", help="只检索，不调大模型")
    p.add_argument("query")
    p = sub.add_parser("ask", help="提问（RAG）")
    p.add_argument("question")
    p = sub.add_parser("add", help="新增笔记")
    p.add_argument("title")
    p.add_argument("--file", default=None, help="从文件读取内容")
    p = sub.add_parser("delete", help="删除笔记")
    p.add_argument("id")
    sub.add_parser("reindex", help="重建索引")

    args = parser.parse_args()
    kb, settings = build_kb()

    if args.cmd == "list":
        stats = kb.stats()
        print(f"共 {stats['documents']} 篇笔记，{stats['chunks']} 个文本块：")
        for d in kb.list_documents():
            print(f"  - {d['title']}  ({d['id']}, {d['chunks']} 块)")

    elif args.cmd == "search":
        chunks = kb.search(args.query, settings.top_k, settings.alpha)
        print(f"检索到 {len(chunks)} 条：")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['title']}  final={c['final_score']} "
                  f"(vec={c['vector_score']}, bm25={c['bm25_score']})")
            print(f"      {c['text'][:80]}…")

    elif args.cmd == "ask":
        try:
            llm = LLM(settings)
        except LLMError as exc:
            sys.exit(f"❌ {exc}")
        result = kb.ask(llm, args.question, settings.top_k, settings.alpha,
                        settings.min_score)
        print("\n" + result["answer"])
        if result["citations"]:
            print("\n📎 引用：")
            for c in result["citations"]:
                print(f"  - {c['title']}：{c['text'][:60]}…")
        elif result.get("low_confidence"):
            print("\n（置信度较低）")

    elif args.cmd == "add":
        if args.file:
            content = open(args.file, encoding="utf-8").read()
        else:
            print("请输入笔记内容（Ctrl+Z 回车结束）：", file=sys.stderr)
            content = sys.stdin.read()
        doc_id = kb.add_note(args.title, content)
        print(f"已添加：{doc_id}")

    elif args.cmd == "delete":
        print("已删除" if kb.delete_note(args.id) else "未找到该笔记")

    elif args.cmd == "reindex":
        print(kb.rebuild())


if __name__ == "__main__":
    main()
