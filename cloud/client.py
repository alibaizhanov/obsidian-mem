"""
Mengram Cloud Client — developer SDK.

Usage:
    from mengram import CloudMemory

    m = CloudMemory(api_key="mg-...")

    # Add memories from conversation
    m.add([
        {"role": "user", "content": "We fixed the OOM with Redis cache. Config: pool-size=20"},
        {"role": "assistant", "content": "Got it, I've noted the HikariCP config change."},
    ], user_id="ali")

    # Search
    results = m.search("database connection issues", user_id="ali")
    for r in results:
        print(f"{r['entity']} (score={r['score']})")

    # Get all
    memories = m.get_all(user_id="ali")

    # Get specific
    entity = m.get("PostgreSQL", user_id="ali")

    # Delete
    m.delete("PostgreSQL", user_id="ali")

    # Stats
    print(m.stats(user_id="ali"))
"""

import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


class CloudMemory:
    """
    Mengram Cloud client.
    
    Drop-in replacement for local Memory class.
    Data stored in cloud PostgreSQL — works from any device.
    """

    DEFAULT_BASE_URL = "https://mengram.io"

    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def _request(self, method: str, path: str, data: dict = None,
                 params: dict = None) -> dict:
        """Make authenticated API request."""
        url = f"{self.base_url}{path}"
        if params:
            query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items() if v is not None)
            if query_string:
                url = f"{url}?{query_string}"
        body = json.dumps(data).encode() if data else None

        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            raise Exception(f"API error {e.code}: {detail}")

    def add(self, messages: list[dict], user_id: str = "default") -> dict:
        """
        Add memories from conversation.
        
        Automatically extracts entities, facts, relations, and knowledge.
        
        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            user_id: User identifier
            
        Returns:
            {"status": "ok", "created": [...], "updated": [...], "knowledge_count": N}
        """
        return self._request("POST", "/v1/add", {
            "messages": messages,
            "user_id": user_id,
        })

    def search(self, query: str, user_id: str = "default",
               limit: int = 5) -> list[dict]:
        """
        Semantic search across memories.
        
        Args:
            query: Natural language query
            user_id: User identifier
            limit: Max results
            
        Returns:
            [{"entity": "...", "type": "...", "score": 0.85, "facts": [...], "knowledge": [...]}]
        """
        result = self._request("POST", "/v1/search", {
            "query": query,
            "user_id": user_id,
            "limit": limit,
        })
        return result.get("results", [])

    def get_all(self, user_id: str = "default") -> list[dict]:
        """Get all memories for user."""
        result = self._request("GET", f"/v1/memories?user_id_param={user_id}")
        return result.get("memories", [])

    def get_all_full(self, user_id: str = "default") -> list[dict]:
        """Get all memories with full details in one request."""
        result = self._request("GET", "/v1/memories/full")
        return result.get("memories", [])

    def get(self, name: str, user_id: str = "default") -> Optional[dict]:
        """Get specific entity details."""
        try:
            return self._request("GET", f"/v1/memory/{name}")
        except Exception:
            return None

    def delete(self, name: str, user_id: str = "default") -> bool:
        """Delete a memory."""
        try:
            self._request("DELETE", f"/v1/memory/{name}")
            return True
        except Exception:
            return False

    def stats(self, user_id: str = "default") -> dict:
        """Get usage statistics."""
        return self._request("GET", "/v1/stats")

    def timeline(self, after: str = None, before: str = None,
                 user_id: str = "default", limit: int = 20) -> list[dict]:
        """Temporal search — facts in a time range."""
        params = {"limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        resp = self._request("GET", "/v1/timeline", params=params)
        return resp.get("results", [])

    def graph(self, user_id: str = "default") -> dict:
        """Get knowledge graph (nodes + edges)."""
        return self._request("GET", "/v1/graph")
