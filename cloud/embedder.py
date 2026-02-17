"""
Cloud Embedder — API-based embeddings (no PyTorch).

Uses OpenAI or Anthropic API instead of local sentence-transformers.
Result: ~200 MB Docker image instead of 8.7 GB.
"""

import json
import os
import urllib.request
from typing import Optional


class CloudEmbedder:
    """
    Generates embeddings via API.
    Uses OpenAI text-embedding-3-large with Matryoshka dimensionality reduction (1536D).
    Better quality than text-embedding-3-small at the same dimensions.
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        if provider == "openai":
            self.model = "text-embedding-3-large"
            self.dimensions = 1536
            self.url = "https://api.openai.com/v1/embeddings"
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        data = json.dumps({
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }).encode()

        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())

        # Sort by index to maintain order
        embeddings = sorted(result["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]
