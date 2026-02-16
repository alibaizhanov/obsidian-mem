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

    def add(self, messages: list[dict], user_id: str = "default",
            agent_id: str = None, run_id: str = None, app_id: str = None,
            expiration_date: str = None) -> dict:
        """
        Add memories from conversation.
        
        Automatically extracts entities, facts, relations, and knowledge.
        Returns immediately — processing happens in background.
        
        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            user_id: User identifier
            agent_id: Agent identifier (for multi-agent systems)
            run_id: Run/session identifier
            app_id: Application identifier
            expiration_date: ISO datetime string — facts auto-expire after this date.
                             None = persist forever.
            
        Returns:
            {"status": "accepted", "job_id": "job-...", "message": "..."}
        """
        body = {"messages": messages, "user_id": user_id}
        if agent_id:
            body["agent_id"] = agent_id
        if run_id:
            body["run_id"] = run_id
        if app_id:
            body["app_id"] = app_id
        if expiration_date:
            body["expiration_date"] = expiration_date
        return self._request("POST", "/v1/add", body)

    def search(self, query: str, user_id: str = "default",
               limit: int = 5, agent_id: str = None,
               run_id: str = None, app_id: str = None) -> list[dict]:
        """
        Semantic search across memories.
        
        Args:
            query: Natural language query
            user_id: User identifier
            limit: Max results
            agent_id: Filter by agent
            run_id: Filter by run/session
            app_id: Filter by application
            
        Returns:
            [{"entity": "...", "type": "...", "score": 0.85, "facts": [...], "knowledge": [...]}]
        """
        body = {"query": query, "user_id": user_id, "limit": limit}
        if agent_id:
            body["agent_id"] = agent_id
        if run_id:
            body["run_id"] = run_id
        if app_id:
            body["app_id"] = app_id
        result = self._request("POST", "/v1/search", body)
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

    # ---- Agents ----

    def run_agents(self, agent: str = "all", auto_fix: bool = False,
                   user_id: str = "default") -> dict:
        """
        Run memory agents.
        
        Args:
            agent: "curator", "connector", "digest", or "all"
            auto_fix: Auto-archive low quality and stale facts (curator only)
            
        Returns:
            Agent results with findings, patterns, suggestions
        """
        return self._request("POST", "/v1/agents/run",
                             params={"agent": agent, "auto_fix": str(auto_fix).lower()})

    def agent_history(self, agent: str = None, limit: int = 10,
                      user_id: str = "default") -> list:
        """Get agent run history."""
        params = {"limit": limit}
        if agent:
            params["agent"] = agent
        result = self._request("GET", "/v1/agents/history", params=params)
        return result.get("runs", [])

    def agent_status(self, user_id: str = "default") -> dict:
        """Check which agents are due to run."""
        return self._request("GET", "/v1/agents/status")

    # ---- Insights & Reflections ----

    def insights(self, user_id: str = "default") -> dict:
        """Get AI insights from memory reflections."""
        return self._request("GET", "/v1/insights")

    def reflect(self, user_id: str = "default") -> dict:
        """Trigger memory reflection — generates AI insights from facts."""
        return self._request("POST", "/v1/reflect")

    def reflections(self, scope: str = None, user_id: str = "default") -> list:
        """Get all reflections. Optional scope: entity, cross, temporal."""
        params = {}
        if scope:
            params["scope"] = scope
        result = self._request("GET", "/v1/reflections", params=params)
        return result.get("reflections", [])

    # ---- Webhooks ----

    def create_webhook(self, url: str, name: str = "",
                       event_types: list = None, secret: str = "",
                       user_id: str = "default") -> dict:
        """
        Create a webhook.
        
        Args:
            url: URL to send POST requests to
            name: Human-readable name
            event_types: ["memory_add", "memory_update", "memory_delete"]
            secret: Optional HMAC secret for signature verification
        """
        data = {"url": url, "name": name, "secret": secret}
        if event_types:
            data["event_types"] = event_types
        result = self._request("POST", "/v1/webhooks", data)
        return result.get("webhook", result)

    def get_webhooks(self, user_id: str = "default") -> list:
        """List all webhooks."""
        result = self._request("GET", "/v1/webhooks")
        return result.get("webhooks", [])

    def update_webhook(self, webhook_id: int, url: str = None,
                       name: str = None, event_types: list = None,
                       active: bool = None, user_id: str = "default") -> dict:
        """Update a webhook."""
        data = {}
        if url is not None: data["url"] = url
        if name is not None: data["name"] = name
        if event_types is not None: data["event_types"] = event_types
        if active is not None: data["active"] = active
        return self._request("PUT", f"/v1/webhooks/{webhook_id}", data)

    def delete_webhook(self, webhook_id: int, user_id: str = "default") -> bool:
        """Delete a webhook."""
        try:
            self._request("DELETE", f"/v1/webhooks/{webhook_id}")
            return True
        except Exception:
            return False

    # ---- Teams ----

    def create_team(self, name: str, description: str = "",
                    user_id: str = "default") -> dict:
        """Create a team. Returns team info with invite_code."""
        result = self._request("POST", "/v1/teams", {"name": name, "description": description})
        return result.get("team", result)

    def join_team(self, invite_code: str, user_id: str = "default") -> dict:
        """Join a team via invite code."""
        return self._request("POST", "/v1/teams/join", {"invite_code": invite_code})

    def get_teams(self, user_id: str = "default") -> list:
        """List user's teams."""
        result = self._request("GET", "/v1/teams")
        return result.get("teams", [])

    def share_memory(self, entity_name: str, team_id: int,
                     user_id: str = "default") -> dict:
        """Share a memory entity with a team."""
        return self._request("POST", f"/v1/teams/{team_id}/share", {"entity": entity_name})

    def unshare_memory(self, entity_name: str, team_id: int,
                       user_id: str = "default") -> dict:
        """Make a shared memory personal again."""
        return self._request("POST", f"/v1/teams/{team_id}/unshare", {"entity": entity_name})

    # ---- API Key Management ----

    def list_keys(self) -> list:
        """List all API keys for your account."""
        return self._request("GET", "/v1/keys")["keys"]

    def create_key(self, name: str = "default") -> dict:
        """Create a new API key. Returns raw key (save it!)."""
        return self._request("POST", "/v1/keys", {"name": name})

    def revoke_key(self, key_id: int) -> dict:
        """Revoke a specific API key by ID."""
        return self._request("DELETE", f"/v1/keys/{key_id}")

    # ---- Job Tracking (Async) ----

    def job_status(self, job_id: str) -> dict:
        """Check status of a background job."""
        return self._request("GET", f"/v1/jobs/{job_id}")

    def wait_for_job(self, job_id: str, poll_interval: float = 1.0,
                     max_wait: float = 60.0) -> dict:
        """Wait for a background job to complete.
        
        Args:
            job_id: Job ID from add() response
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait
            
        Returns:
            Job result when completed
        """
        import time as _time
        start = _time.time()
        while _time.time() - start < max_wait:
            job = self.job_status(job_id)
            if job["status"] in ("completed", "failed"):
                return job
            _time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} timed out after {max_wait}s")
