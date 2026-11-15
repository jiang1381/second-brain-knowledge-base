"""文档加载：把 notes/ 目录下的笔记读成统一的文档结构。

支持：.txt / .md / .html / .docx（docx 需要 python-docx，装了才启用）。
PDF 需要额外库（pypdf），本项目暂不支持，作为扩展项。
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._parts)


def _html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    return parser.text()


def _docx_to_text(path: Path) -> str:
    from docx import Document   # 延迟导入，没装就跳过
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_title(text: str, fallback: str) -> str:
    """优先取 Markdown 一级标题，其次第一行，最后用文件名。"""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
        return line[:60]
    return fallback


def load_documents(notes_dir: Path) -> list[dict]:
    """遍历笔记目录，返回 [{id, title, text, path}]。"""
    notes_dir = Path(notes_dir)
    documents: list[dict] = []
    if not notes_dir.exists():
        return documents

    for path in sorted(notes_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt", ".md"):
                text = path.read_text(encoding="utf-8")
            elif suffix in (".html", ".htm"):
                text = _html_to_text(path.read_text(encoding="utf-8"))
            elif suffix == ".docx":
                text = _docx_to_text(path)
            else:
                continue
        except Exception:
            continue                      # 单个文件读失败不影响整体

        if not text.strip():
            continue
        doc_id = str(path.relative_to(notes_dir)).replace("\\", "/")
        documents.append({
            "id": doc_id,
            "title": _extract_title(text, path.stem),
            "text": text,
            "path": str(path),
        })
    return documents
