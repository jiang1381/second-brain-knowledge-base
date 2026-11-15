"""RAG 问答：拼上下文 → 要求引用 → 解析引用 → 置信度判断。

面试考点：
- 引用溯源：要求模型用 [n] 标注，程序再校验哪些片段真的被引用了；
- 反幻觉：资料不足时让模型明确说「没有」，而不是编。
"""
from __future__ import annotations

import re

# 模型拒答时常见的说法，用来判断「低置信」
NO_INFO_MARKERS = (
    "没有相关", "没有找到", "未找到", "没有足够", "无法据此",
    "知识库里没有", "无法回答", "没有提到",
)

SYSTEM = """你是用户的「第二大脑」助手，基于用户自己的笔记回答问题。

规则：
1. 只依据下面提供的笔记片段回答，不要使用片段之外的知识；
2. 如果片段不足以回答，明确说「我的知识库里没有相关信息」，禁止编造；
3. 回答中用 [编号] 标注引用来源，例如 [1][2]；
4. 语气简洁、分点，重要结论放前面。"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] （笔记：{chunk['title']}）\n{chunk['text']}")
    return "\n\n".join(parts)


def extract_citations(text: str, chunks: list[dict]) -> list[dict]:
    """解析回答里的 [n]，返回真正被引用到的片段。"""
    used = {int(n) for n in re.findall(r"\[(\d+)\]", text or "")}
    return [chunks[i - 1] for i in sorted(used) if 1 <= i <= len(chunks)]


def answer(llm, question: str, chunks: list[dict], min_score: float) -> dict:
    if not chunks:
        return {
            "answer": "我的知识库里还没有相关笔记，先往里添加一些内容吧。",
            "low_confidence": True,
            "citations": [],
        }

    top_score = chunks[0].get("final_score", 0.0)
    if top_score < min_score:
        return {
            "answer": (f"我在知识库里没有找到足够相关的笔记（最高相关度 {top_score:.2f}）。"
                       "换个说法，或者先补充一些笔记再问我。"),
            "low_confidence": True,
            "citations": [],
        }

    prompt = (f"笔记片段：\n\n{build_context(chunks)}\n\n"
              f"问题：{question}\n\n"
              "请严格基于上面的笔记回答，并用 [编号] 标注引用。")
    text = llm.chat(SYSTEM, prompt)
    citations = extract_citations(text, chunks)
    refused = any(marker in text for marker in NO_INFO_MARKERS)
    return {
        "answer": text.strip(),
        # 模型明确说「没有」或没引用任何片段，都视为低置信
        "low_confidence": refused or not citations,
        "citations": citations,
    }
