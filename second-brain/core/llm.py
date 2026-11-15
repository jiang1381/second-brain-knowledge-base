"""大模型客户端：标准库 urllib 手写调用，带重试。

对外只暴露 chat(system, user) -> str，简化上层逻辑。
"""
from __future__ import annotations

import json
import time
import urllib.request


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, settings):
        if settings.offline:
            raise LLMError("未配置 LLM_API_KEY，请在 second-brain/.env 中填写。")
        self.settings = settings

    def chat(self, system: str, user: str, temperature: float | None = None) -> str:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (temperature if temperature is not None
                            else self.settings.temperature),
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err: Exception | None = None

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.settings.llm_api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"调用大模型失败：{last_err}")
