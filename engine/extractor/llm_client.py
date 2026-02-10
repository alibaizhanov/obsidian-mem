"""
LLM Client — вызов LLM для извлечения знаний.

Поддерживает:
- Anthropic (Claude) — через API
- OpenAI (GPT) — через API
- Ollama — локально, бесплатно
"""

import json
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Абстрактный клиент для LLM"""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Отправить промпт, получить ответ"""
        pass


class AnthropicClient(LLMClient):
    """Claude через Anthropic API"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "You are a knowledge extraction assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIClient(LLMClient):
    """GPT через OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a knowledge extraction assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class OllamaClient(LLMClient):
    """Ollama — полностью локальная LLM (бесплатно)"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str, system: str = "") -> str:
        import urllib.request
        import json

        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system or "You are a knowledge extraction assistant.",
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result["response"]


def create_llm_client(config: dict) -> LLMClient:
    """Создаёт LLM клиент на основе конфига"""
    provider = config.get("provider", "anthropic")

    if provider == "anthropic":
        settings = config.get("anthropic", {})
        return AnthropicClient(
            api_key=settings["api_key"],
            model=settings.get("model", "claude-sonnet-4-20250514"),
        )
    elif provider == "openai":
        settings = config.get("openai", {})
        return OpenAIClient(
            api_key=settings["api_key"],
            model=settings.get("model", "gpt-4o-mini"),
        )
    elif provider == "ollama":
        settings = config.get("ollama", {})
        return OllamaClient(
            base_url=settings.get("base_url", "http://localhost:11434"),
            model=settings.get("model", "llama3.2"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
