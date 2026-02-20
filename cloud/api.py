"""
Mengram Cloud API Server

Hosted version — PostgreSQL + pgvector backend.
Developers get API key, integrate in 3 lines:

    from cloud.client import CloudMemory
    m = CloudMemory(api_key="om-...")
    m.add(messages)
    results = m.search("database issues")
"""

import os
import sys
import logging
import secrets
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mengram")

from fastapi import FastAPI, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel

from cloud.store import CloudStore


# ---- Config ----

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://localhost:5432/mengram"
)
REDIS_URL = os.environ.get("REDIS_PUBLIC_URL") or os.environ.get("REDIS_URL")

# ---- Models ----

class Message(BaseModel):
    role: str
    content: str

class AddRequest(BaseModel):
    messages: list[Message]
    user_id: str = "default"
    agent_id: str = None
    run_id: str = None
    app_id: str = None
    expiration_date: str = None

class AddTextRequest(BaseModel):
    text: str
    user_id: str = "default"
    agent_id: str = None
    run_id: str = None
    app_id: str = None
    expiration_date: str = None

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    agent_id: str = None
    run_id: str = None
    app_id: str = None
    limit: int = 5
    graph_depth: int = 2  # 0=no graph, 1=1-hop, 2=2-hop (default)

class FeedbackRequest(BaseModel):
    context: str = None         # What went wrong (triggers evolution on failure)
    failed_at_step: int = None  # Which step failed

class SignupRequest(BaseModel):
    email: str

class SignupResponse(BaseModel):
    api_key: str
    message: str

class ResetKeyRequest(BaseModel):
    email: str


# ---- App ----

def create_cloud_api() -> FastAPI:
    app = FastAPI(
        title="Mengram API",
        description="""
## Human-Like Memory for AI — Semantic + Episodic + Procedural

The only AI memory API with 3 memory types. Your AI remembers facts, events, and learned workflows.

### 3 Memory Types
- **Semantic** — facts, preferences, skills (entities, relations, knowledge graph)
- **Episodic** — events, decisions, experiences (what happened, when, outcome)
- **Procedural** — workflows, processes, habits (learned step-by-step procedures)

### Key Features
- **Cognitive Profile** — one API call generates a system prompt from all memory types
- **Unified Search** — search across all 3 types simultaneously
- **Procedure Feedback** — AI learns which workflows succeed
- **Memory Agents** — autonomous cleanup, pattern detection, weekly digests
- **Team Sharing** — shared memory across team members
- **LangChain** — drop-in replacement for ConversationBufferMemory
- **CrewAI** — 5 tools with procedural learning (agents learn optimal workflows)
- **OpenClaw** — skill with 3 memory types across all channels (WhatsApp, Telegram, Discord)

### Authentication
All endpoints require `Authorization: Bearer YOUR_API_KEY` header.

### Quick Start
```python
from cloud.client import CloudMemory
m = CloudMemory(api_key="om-...")
m.add([{"role": "user", "content": "I use Python and Railway"}])
results = m.search_all("deployment")  # semantic + episodic + procedural
profile = m.get_profile()             # instant system prompt
```
        """,
        version="2.11.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Memory", "description": "Store and retrieve semantic memories"},
            {"name": "Episodic Memory", "description": "Events, decisions, experiences — what happened"},
            {"name": "Procedural Memory", "description": "Workflows, processes — how to do things"},
            {"name": "Search", "description": "Semantic and unified search across all memory types"},
            {"name": "Agents", "description": "Autonomous memory agents — Curator, Connector, Digest"},
            {"name": "Teams", "description": "Shared team memory with invite codes"},
            {"name": "Webhooks", "description": "HTTP notifications on memory events"},
            {"name": "Insights", "description": "AI-generated reflections and patterns"},
            {"name": "System", "description": "Health, stats, and account management"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = CloudStore(DATABASE_URL, redis_url=REDIS_URL)

    # LLM client for extraction (shared)
    _llm_client = None
    _extractor = None

    def get_llm():
        nonlocal _llm_client, _extractor
        if _llm_client is None:
            from engine.extractor.llm_client import create_llm_client
            llm_config = {
                "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
                "anthropic": {"api_key": os.environ.get("ANTHROPIC_API_KEY", "")},
                "openai": {"api_key": os.environ.get("OPENAI_API_KEY", "")},
            }
            _llm_client = create_llm_client(llm_config)
            from engine.extractor.conversation_extractor import ConversationExtractor
            _extractor = ConversationExtractor(_llm_client)
        return _extractor

    # Embedder (shared — API-based, no PyTorch)
    _embedder = None

    def get_embedder():
        nonlocal _embedder
        if _embedder is None:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                from cloud.embedder import CloudEmbedder
                _embedder = CloudEmbedder(provider="openai", api_key=openai_key)
        return _embedder

    # ---- Re-ranking (Cohere Rerank → gpt-4o-mini fallback) ----

    def rerank_results(query: str, results: list[dict]) -> list[dict]:
        """Re-rank search results using Cohere Rerank (fast cross-encoder).
        Falls back to gpt-4o-mini if Cohere unavailable."""
        if not results or len(results) <= 1:
            return results

        # Try Cohere Rerank first (cross-encoder, faster than LLM)
        cohere_key = os.environ.get("COHERE_API_KEY", "")
        if cohere_key:
            try:
                import cohere
                co = cohere.ClientV2(api_key=cohere_key)

                # Build document texts for reranking
                documents = []
                for r in results:
                    facts_str = "; ".join(r.get("facts", [])[:5])
                    rels_str = "; ".join(
                        f"{rel.get('type', '')} {rel.get('target', '')}"
                        for rel in r.get("relations", [])[:3]
                    )
                    doc = f"{r['entity']} ({r['type']}): {facts_str}"
                    if rels_str:
                        doc += f" | {rels_str}"
                    documents.append(doc)

                resp = co.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=documents,
                    top_n=len(results),
                )

                # Filter by relevance score and reorder
                reranked = []
                for item in resp.results:
                    if item.relevance_score >= 0.1:
                        reranked.append(results[item.index])
                return reranked if reranked else results

            except Exception as e:
                logger.warning(f"⚠️ Cohere rerank failed, falling back: {e}")

        # Fallback: gpt-4o-mini
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            return results

        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)

            candidates = []
            for i, r in enumerate(results):
                facts_str = "; ".join(r.get("facts", [])[:5])
                rels_str = "; ".join(
                    f"{rel.get('type', '')} {rel.get('target', '')}"
                    for rel in r.get("relations", [])[:3]
                )
                info = f"[{i}] {r['entity']} ({r['type']}): {facts_str}"
                if rels_str:
                    info += f" | relations: {rels_str}"
                candidates.append(info)

            prompt = f"""Given the user's query, select ONLY the entities that are directly relevant.

Query: "{query}"

Candidates:
{chr(10).join(candidates)}

Return ONLY a JSON array of indices of relevant entities, e.g. [0, 2, 4].
If none are relevant, return [].
Be strict — only include entities that directly answer or relate to the query."""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )

            text = resp.choices[0].message.content.strip()
            import json as json_mod
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            indices = json_mod.loads(text)

            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                filtered = [results[i] for i in indices if 0 <= i < len(results)]
                if filtered:
                    return filtered

            return results

        except Exception as e:
            logger.error(f"⚠️ Re-ranking failed, returning raw results: {e}")
            return results

    # ---- Rate Limiting (Redis-shared or in-memory fallback) ----
    _rate_limits = {}  # fallback: user_id -> {"count": N, "window_start": time}
    _rate_lock = __import__('threading').Lock()
    RATE_LIMIT = 120  # requests per minute
    RATE_WINDOW = 60   # seconds

    def _check_rate_limit(user_id: str) -> bool:
        """Returns True if allowed, False if rate limited.
        Uses Redis INCR for cross-worker consistency when available."""
        # Try Redis first (shared across workers)
        redis_client = getattr(store.cache, '_redis', None) if store else None
        if redis_client:
            try:
                key = f"rl:{user_id}"
                count = redis_client.incr(key)
                if count == 1:
                    redis_client.expire(key, RATE_WINDOW)
                return count <= RATE_LIMIT
            except Exception:
                pass  # fall through to in-memory

        # In-memory fallback (per-worker)
        import time as _time
        now = _time.time()
        with _rate_lock:
            entry = _rate_limits.get(user_id)
            if not entry or now - entry["window_start"] >= RATE_WINDOW:
                _rate_limits[user_id] = {"count": 1, "window_start": now}
                return True
            if entry["count"] >= RATE_LIMIT:
                return False
            entry["count"] += 1
            return True

    # ---- Auth middleware ----

    async def auth(authorization: str = Header(...)) -> str:
        """Verify API key, return user_id. Rate limited."""
        key = authorization.replace("Bearer ", "")
        user_id = store.verify_api_key(key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not _check_rate_limit(user_id):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({RATE_LIMIT} requests/min). Retry in 60 seconds."
            )
        return user_id

    # ---- Email helper ----

    def _send_api_key_email(email: str, api_key: str, is_reset: bool = False):
        """Send API key to user via Resend."""
        resend_key = os.environ.get("RESEND_API_KEY")
        if not resend_key:
            logger.info("⚠️  RESEND_API_KEY not set, skipping email")
            return

        try:
            import resend
            resend.api_key = resend_key

            action = "reset" if is_reset else "created"
            subject = f"Your new Mengram API key" if is_reset else "Welcome to Mengram"

            html = f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:520px;margin:0 auto;padding:40px 24px;color:#e8e8f0;background:#0a0a12;border-radius:16px">
                <div style="text-align:center;margin-bottom:32px">
                    <svg width="36" height="36" viewBox="0 0 120 120"><path d="M60 16 Q92 16 96 48 Q100 78 72 88 Q50 96 38 76 Q26 58 46 46 Q62 38 70 52 Q76 64 62 68" fill="none" stroke="#a855f7" stroke-width="8" stroke-linecap="round"/><circle cx="62" cy="68" r="8" fill="#a855f7"/><circle cx="62" cy="68" r="3.5" fill="white"/></svg>
                    <h1 style="font-size:22px;font-weight:700;margin:8px 0 4px;color:#e8e8f0">Mengram</h1>
                    <p style="color:#8888a8;font-size:14px;margin:0">AI memory layer for apps</p>
                </div>
                <p style="font-size:15px;color:#c8c8d8;line-height:1.6">
                    {"Your API key has been reset. Old keys are now deactivated." if is_reset else "Welcome! Your account has been created."}
                </p>
                <div style="background:#12121e;border:1px solid #1a1a2e;border-radius:10px;padding:18px;margin:20px 0;text-align:center">
                    <p style="color:#8888a8;font-size:12px;margin:0 0 8px;text-transform:uppercase;letter-spacing:1px">Your API Key</p>
                    <code style="font-size:14px;color:#a78bfa;word-break:break-all">{api_key}</code>
                </div>
                <p style="font-size:13px;color:#ef4444;font-weight:600">⚠️ Save this key — it won't be shown again.</p>
                <p style="font-size:14px;color:#8888a8;margin-top:24px">
                    Quick start:<br>
                    <code style="color:#22c55e;font-size:13px">pip install mengram-ai</code>
                </p>
                <hr style="border:none;border-top:1px solid #1a1a2e;margin:28px 0">
                <p style="font-size:12px;color:#55556a;text-align:center">
                    <a href="https://mengram.io/dashboard" style="color:#7c3aed;text-decoration:none">Dashboard</a> ·
                    <a href="https://mengram.io/docs" style="color:#7c3aed;text-decoration:none">API Docs</a> ·
                    <a href="https://github.com/alibaizhanov/mengram" style="color:#7c3aed;text-decoration:none">GitHub</a>
                </p>
            </div>
            """

            resend.Emails.send({
                "from": "Mengram <onboarding@resend.dev>",
                "to": [email],
                "subject": subject,
                "html": html,
            })
            logger.info(f"📧 Email sent to {email} (key {action})")
        except Exception as e:
            logger.error(f"⚠️  Email send failed: {e}")

    # ---- Public endpoints ----

    @app.get("/", response_class=HTMLResponse)
    async def landing():
        """Landing page."""
        landing_path = Path(__file__).parent / "landing.html"
        return landing_path.read_text(encoding="utf-8")

    @app.get("/robots.txt", response_class=PlainTextResponse)
    async def robots():
        return "User-agent: *\nAllow: /\nSitemap: https://mengram.io/sitemap.xml"

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Web dashboard."""
        dashboard_path = Path(__file__).parent / "dashboard.html"
        return dashboard_path.read_text(encoding="utf-8")

    @app.get("/extension/download")
    async def download_extension():
        """Download Chrome extension zip."""
        ext_path = Path(__file__).parent / "mengram-chrome-extension.zip"
        if not ext_path.exists():
            raise HTTPException(status_code=404, detail="Extension not available")
        return FileResponse(
            path=str(ext_path),
            filename="mengram-chrome-extension.zip",
            media_type="application/zip"
        )

    @app.post("/v1/signup", tags=["System"], response_model=SignupResponse)
    async def signup(req: SignupRequest):
        """Create account and get API key."""
        existing = store.get_user_by_email(req.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        user_id = store.create_user(req.email)
        api_key = store.create_api_key(user_id)

        # Send key via email
        _send_api_key_email(req.email, api_key, is_reset=False)

        return SignupResponse(
            api_key=api_key,
            message="API key sent to your email. Save it — it won't be shown again."
        )

    @app.post("/v1/reset-key", tags=["System"])
    async def reset_key(req: ResetKeyRequest):
        """Reset API key and send new one to email."""
        user_id = store.get_user_by_email(req.email)
        if not user_id:
            # Don't reveal whether email exists
            return {"message": "If this email is registered, a new API key has been sent."}

        new_key = store.reset_api_key(user_id)
        _send_api_key_email(req.email, new_key, is_reset=True)

        return {"message": "If this email is registered, a new API key has been sent."}

    # ---- API Key Management ----

    @app.get("/v1/keys", tags=["System"])
    async def list_keys(user_id: str = Depends(auth)):
        """List all API keys for your account."""
        keys = store.list_api_keys(user_id)
        return {"keys": keys, "total": len(keys)}

    @app.post("/v1/keys", tags=["System"])
    async def create_key(req: dict, user_id: str = Depends(auth)):
        """Create a new API key with a name."""
        name = req.get("name", "default")
        if len(name) > 50:
            raise HTTPException(status_code=400, detail="Name too long (max 50 chars)")
        raw_key = store.create_api_key(user_id, name=name)
        return {
            "key": raw_key,
            "name": name,
            "message": "Save this key — it won't be shown again."
        }

    @app.delete("/v1/keys/{key_id}", tags=["System"])
    async def revoke_key(key_id: str, user_id: str = Depends(auth)):
        """Revoke a specific API key."""
        # Don't allow revoking the key being used for this request
        keys = store.list_api_keys(user_id)
        active_count = sum(1 for k in keys if k["active"])
        if active_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot revoke your last active key. Create a new one first."
            )
        if store.revoke_api_key(user_id, key_id):
            return {"status": "revoked", "key_id": key_id}
        raise HTTPException(status_code=404, detail="Key not found or already revoked")

    @app.patch("/v1/keys/{key_id}", tags=["System"])
    async def rename_key(key_id: str, req: dict, user_id: str = Depends(auth)):
        """Rename an API key."""
        name = req.get("name", "")
        if not name or len(name) > 50:
            raise HTTPException(status_code=400, detail="Name required (max 50 chars)")
        if store.rename_api_key(user_id, key_id, name):
            return {"status": "renamed", "key_id": key_id, "name": name}
        raise HTTPException(status_code=404, detail="Key not found")

    # ---- OAuth (for ChatGPT Custom GPTs) ----

    @app.get("/oauth/authorize")
    async def oauth_authorize(
        client_id: str = "",
        redirect_uri: str = "",
        state: str = "",
        response_type: str = "code",
    ):
        """OAuth authorize page — shows email login."""
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mengram — Sign In</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,system-ui,sans-serif; background:#0a0a0a; color:#e0e0e0;
         display:flex; align-items:center; justify-content:center; min-height:100vh; }}
  .card {{ background:#141414; border:1px solid #2a2a2a; border-radius:16px; padding:40px;
           max-width:400px; width:100%; }}
  h1 {{ font-size:24px; margin-bottom:8px; }}
  p {{ color:#888; margin-bottom:24px; font-size:14px; }}
  input {{ width:100%; padding:12px 16px; background:#1a1a1a; border:1px solid #333;
           border-radius:8px; color:#e0e0e0; font-size:16px; margin-bottom:12px; outline:none; }}
  input:focus {{ border-color:#646cff; }}
  button {{ width:100%; padding:12px; background:#646cff; color:white; border:none;
            border-radius:8px; font-size:16px; cursor:pointer; }}
  button:hover {{ background:#5558dd; }}
  .step {{ display:none; }}
  .step.active {{ display:block; }}
  .error {{ color:#ff4444; font-size:13px; margin-bottom:12px; display:none; }}
  .logo {{ font-size:32px; margin-bottom:16px; }}
</style>
</head><body>
<div class="card">
  <div class="logo"><svg width='32' height='32' viewBox='0 0 120 120'><path d='M60 16 Q92 16 96 48 Q100 78 72 88 Q50 96 38 76 Q26 58 46 46 Q62 38 70 52 Q76 64 62 68' fill='none' stroke='#a855f7' stroke-width='8' stroke-linecap='round'/><circle cx='62' cy='68' r='8' fill='#a855f7'/><circle cx='62' cy='68' r='3.5' fill='white'/></svg></div>
  <h1>Sign in to Mengram</h1>
  <p>Connect your memory to ChatGPT</p>

  <div id="step1" class="step active">
    <input type="email" id="email" placeholder="your@email.com" autofocus>
    <div class="error" id="err1"></div>
    <button onclick="sendCode()">Send verification code</button>
  </div>

  <div id="step2" class="step">
    <p id="sentMsg" style="color:#888">Code sent to your email</p>
    <input type="text" id="code" placeholder="Enter 6-digit code" maxlength="6">
    <div class="error" id="err2"></div>
    <button onclick="verifyCode()">Verify & Connect</button>
  </div>
</div>

<script>
const redirectUri = "{redirect_uri}";
const state = "{state}";

async function sendCode() {{
  const email = document.getElementById('email').value.trim();
  if (!email) return;
  const res = await fetch('/oauth/send-code', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{email}})
  }});
  const data = await res.json();
  if (data.ok) {{
    document.getElementById('step1').classList.remove('active');
    document.getElementById('step2').classList.add('active');
    document.getElementById('sentMsg').textContent = 'Code sent to ' + email;
  }} else {{
    document.getElementById('err1').textContent = data.error || 'Failed to send code';
    document.getElementById('err1').style.display = 'block';
  }}
}}

async function verifyCode() {{
  const email = document.getElementById('email').value.trim();
  const code = document.getElementById('code').value.trim();
  const res = await fetch('/oauth/verify', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{email, code, redirect_uri: redirectUri, state}})
  }});
  const data = await res.json();
  if (data.redirect) {{
    window.location.href = data.redirect;
  }} else {{
    document.getElementById('err2').textContent = data.error || 'Invalid code';
    document.getElementById('err2').style.display = 'block';
  }}
}}

document.getElementById('email').addEventListener('keydown', e => {{ if(e.key==='Enter') sendCode(); }});
document.getElementById('code').addEventListener('keydown', e => {{ if(e.key==='Enter') verifyCode(); }});
</script>
</body></html>""")

    @app.post("/oauth/send-code")
    async def oauth_send_code(req: dict):
        """Send email verification code for OAuth."""
        email = req.get("email", "").strip().lower()
        if not email:
            return {"ok": False, "error": "Email required"}

        # Check if user exists, if not create
        user_id = store.get_user_by_email(email)
        if not user_id:
            user_id = store.create_user(email)
            store.create_api_key(user_id)

        # Generate and send 6-digit code
        code = f"{secrets.randbelow(900000) + 100000}"
        store.save_email_code(email, code)

        # Send via Resend
        resend_key = os.environ.get("RESEND_API_KEY")
        if resend_key:
            try:
                import resend
                resend.api_key = resend_key
                resend.Emails.send({
                    "from": "Mengram <onboarding@resend.dev>",
                    "to": [email],
                    "subject": "Mengram verification code",
                    "html": f"<h2>Your code: {code}</h2><p>Expires in 10 minutes.</p>",
                })
            except Exception as e:
                logger.error(f"⚠️ Email send failed: {e}")
                return {"ok": False, "error": "Failed to send email"}
        else:
            logger.warning(f"⚠️ No RESEND_API_KEY, code for {email}: {code}")

        return {"ok": True}

    @app.post("/oauth/verify")
    async def oauth_verify(req: dict):
        """Verify email code and create OAuth authorization code."""
        email = req.get("email", "").strip().lower()
        code = req.get("code", "").strip()
        redirect_uri = req.get("redirect_uri", "")
        state = req.get("state", "")

        if not store.verify_email_code(email, code):
            return {"error": "Invalid or expired code"}

        user_id = store.get_user_by_email(email)
        if not user_id:
            return {"error": "User not found"}

        # Create OAuth authorization code
        oauth_code = secrets.token_urlsafe(32)
        store.save_oauth_code(oauth_code, user_id, redirect_uri, state)

        # Build redirect URL
        separator = "&" if "?" in redirect_uri else "?"
        redirect_url = f"{redirect_uri}{separator}code={oauth_code}&state={state}"

        return {"redirect": redirect_url}

    @app.post("/oauth/token")
    async def oauth_token(
        grant_type: str = Form("authorization_code"),
        code: str = Form(""),
        client_id: str = Form(""),
        client_secret: str = Form(""),
        redirect_uri: str = Form(""),
    ):
        """Exchange OAuth code for access token."""
        if grant_type != "authorization_code":
            raise HTTPException(status_code=400, detail="Unsupported grant_type")

        result = store.verify_oauth_code(code)
        if not result:
            raise HTTPException(status_code=400, detail="Invalid or expired code")

        # Get or create API key for this user
        user_id = result["user_id"]
        api_key = store.create_api_key(user_id, name="chatgpt-oauth")

        return {
            "access_token": api_key,
            "token_type": "Bearer",
            "scope": "read write",
        }

    @app.get("/v1/health", tags=["System"])
    async def health():
        cache_stats = store.cache.stats()
        pool_info = {"type": "pool", "max": 10} if store._pool else {"type": "single"}
        return {
            "status": "ok",
            "version": "2.11.0",
            "cache": cache_stats,
            "connection": pool_info,
        }

    # ---- Protected endpoints ----

    @app.post("/v1/add", tags=["Memory"])
    async def add(req: AddRequest, user_id: str = Depends(auth)):
        """
        Add memories from conversation.
        Returns immediately with job_id, processes in background.
        """
        import threading

        job_id = store.create_job(user_id, "add")
        # Build metadata from categories
        metadata = {}
        if req.agent_id:
            metadata["agent_id"] = req.agent_id
        if req.run_id:
            metadata["run_id"] = req.run_id
        if req.app_id:
            metadata["app_id"] = req.app_id

        def process_in_background():
            created = []
            try:
                extractor = get_llm()
                conversation = [{"role": m.role, "content": m.content} for m in req.messages]

                # Get existing entities context for smarter extraction
                existing_context = ""
                try:
                    existing_context = store.get_existing_context(user_id)
                except Exception as e:
                    logger.error(f"⚠️ Context fetch failed: {e}")

                extraction = extractor.extract(conversation, existing_context=existing_context)

                # ---- Phase 1: Parallel conflict resolution (LLM calls) ----
                from concurrent.futures import ThreadPoolExecutor, as_completed

                conflict_tasks = []
                for entity in extraction.entities:
                    if not entity.name:
                        continue
                    existing_id = store.get_entity_id(user_id, entity.name)
                    if existing_id and entity.facts:
                        conflict_tasks.append((entity, existing_id))

                conflict_results = {}  # name -> archived_facts
                if conflict_tasks:
                    def _check_conflicts(entity, existing_id):
                        try:
                            archived = store.archive_contradicted_facts(
                                existing_id, entity.facts, extractor.llm)
                            return entity.name, archived
                        except Exception as e:
                            logger.error(f"⚠️ Conflict check failed for {entity.name}: {e}")
                            return entity.name, []

                    with ThreadPoolExecutor(max_workers=5) as pool:
                        futures = [pool.submit(_check_conflicts, ent, eid)
                                   for ent, eid in conflict_tasks]
                        for future in as_completed(futures):
                            name, archived = future.result()
                            conflict_results[name] = archived

                # ---- Phase 2: Save entities + collect embedding chunks ----
                embedding_queue = []  # [(entity_id, chunks)]
                for entity in extraction.entities:
                    name = entity.name
                    if not name:
                        continue

                    # Build relations
                    entity_relations = []
                    for rel in extraction.relations:
                        if rel.from_entity == name:
                            entity_relations.append({
                                "target": rel.to_entity,
                                "type": rel.relation_type,
                                "description": rel.description,
                                "direction": "outgoing",
                            })
                        elif rel.to_entity == name:
                            entity_relations.append({
                                "target": rel.from_entity,
                                "type": rel.relation_type,
                                "description": rel.description,
                                "direction": "incoming",
                            })

                    # Build knowledge
                    entity_knowledge = []
                    for k in extraction.knowledge:
                        if k.entity == name:
                            entity_knowledge.append({
                                "type": k.knowledge_type,
                                "title": k.title,
                                "content": k.content,
                                "artifact": k.artifact,
                            })

                    # Use pre-computed conflict results
                    archived = conflict_results.get(name)
                    if archived:
                        store.fire_webhooks(user_id, "memory_update", {
                            "entity": name,
                            "archived_facts": archived,
                            "new_facts": entity.facts
                        })

                    entity_id = store.save_entity(
                        user_id=user_id,
                        name=name,
                        type=entity.entity_type,
                        facts=entity.facts,
                        relations=entity_relations,
                        knowledge=entity_knowledge,
                        metadata=metadata if metadata else None,
                        expires_at=req.expiration_date,
                    )
                    created.append(name)

                    # Collect chunks for batched embedding
                    chunks = [name] + entity.facts
                    for r in entity_relations:
                        target = r.get("target", "")
                        rel_type = r.get("type", "")
                        if target and rel_type:
                            chunks.append(f"{name} {rel_type} {target}")
                    for k in entity_knowledge:
                        chunks.append(f"{k['title']} {k['content']}")
                    embedding_queue.append((entity_id, chunks))

                # ---- Phase 3: Batch embeddings across ALL entities (1 API call) ----
                embedder = get_embedder()
                if embedder and embedding_queue:
                    all_chunks = []
                    chunk_map = []  # [(entity_id, chunk_index)]
                    for entity_id, chunks in embedding_queue:
                        store.delete_embeddings(entity_id)
                        for chunk in chunks:
                            chunk_map.append((entity_id, chunk))
                            all_chunks.append(chunk)

                    if all_chunks:
                        all_embeddings = embedder.embed_batch(all_chunks)
                        for (entity_id, chunk_text), emb in zip(chunk_map, all_embeddings):
                            store.save_embedding(entity_id, chunk_text, emb)

                store.log_usage(user_id, "add")

                # ---- Episodic Memory: save episodes ----
                episodes_created = 0
                episodes_linked = 0
                embedder = get_embedder()
                for ep in extraction.episodes:
                    if not ep.summary:
                        continue
                    try:
                        episode_id = store.save_episode(
                            user_id=user_id,
                            summary=ep.summary,
                            context=ep.context,
                            outcome=ep.outcome,
                            participants=ep.participants,
                            emotional_valence=ep.emotional_valence,
                            importance=ep.importance,
                            metadata=metadata if metadata else None,
                            expires_at=req.expiration_date,
                        )
                        # Embed episode (truncate to 2000 chars for embedder safety)
                        ep_embedding = None
                        if embedder:
                            ep_text = f"{ep.summary}. {ep.context or ''} {ep.outcome or ''}"[:2000]
                            ep_embs = embedder.embed_batch([ep_text])
                            if ep_embs:
                                ep_embedding = ep_embs[0]
                                store.save_episode_embedding(episode_id, ep_text, ep_embedding)

                        # ---- Auto-link episode to existing procedure ----
                        if ep_embedding:
                            try:
                                similar_procs = store.search_procedures_vector(
                                    user_id, ep_embedding, top_k=1)
                                if similar_procs and similar_procs[0]["score"] >= 0.75:
                                    matched_proc = similar_procs[0]
                                    # Link episode to procedure
                                    store.link_episodes_to_procedure(
                                        [episode_id], matched_proc["id"])

                                    is_negative = ep.emotional_valence == "negative"
                                    if is_negative:
                                        # Failure → trigger evolution
                                        from cloud.evolution import EvolutionEngine
                                        evo = EvolutionEngine(store, embedder, extractor.llm)
                                        evo_result = evo.evolve_on_failure(
                                            user_id, matched_proc["id"], episode_id,
                                            ep.context or ep.summary)
                                        if evo_result:
                                            logger.info(
                                                f"🔄 Auto-evolved '{matched_proc['name']}' "
                                                f"v{evo_result['old_version']}→v{evo_result['new_version']} "
                                                f"from episode")
                                    else:
                                        # Success → increment success count
                                        store.procedure_feedback(
                                            user_id, matched_proc["id"], success=True)

                                    episodes_linked += 1
                            except Exception as e:
                                logger.error(f"⚠️ Episode auto-link failed: {e}")

                        episodes_created += 1
                    except Exception as e:
                        logger.error(f"⚠️ Episode save failed: {e}")

                # ---- Procedural Memory: save procedures ----
                procedures_created = 0
                for pr in extraction.procedures:
                    if not pr.name or not pr.steps:
                        continue
                    try:
                        proc_id = store.save_procedure(
                            user_id=user_id,
                            name=pr.name,
                            trigger_condition=pr.trigger,
                            steps=pr.steps,
                            entity_names=pr.entities,
                            metadata=metadata if metadata else None,
                            expires_at=req.expiration_date,
                        )
                        # Embed procedure
                        if embedder:
                            steps_summary = "; ".join(
                                s.get("action", "") for s in pr.steps[:10]
                            )
                            pr_text = f"{pr.name}. {pr.trigger or ''}. Steps: {steps_summary}"
                            pr_embs = embedder.embed_batch([pr_text])
                            if pr_embs:
                                store.delete_procedure_embeddings(proc_id)
                                store.save_procedure_embedding(proc_id, pr_text, pr_embs[0])
                        procedures_created += 1
                    except Exception as e:
                        logger.error(f"⚠️ Procedure save failed: {e}")

                # Invalidate search cache — fresh data available
                store.cache.invalidate(f"search:{user_id}")
                store.cache.invalidate(f"searchall:{user_id}")

                logger.info(f"✅ Background add complete for {user_id} "
                           f"(entities={len(created)}, episodes={episodes_created}, "
                           f"procedures={procedures_created}, linked={episodes_linked})")
                store.complete_job(job_id, {
                    "created": created,
                    "count": len(created),
                    "episodes": episodes_created,
                    "procedures": procedures_created,
                    "episodes_linked": episodes_linked,
                })

                # Auto-trigger reflection if needed
                try:
                    if store.should_reflect(user_id):
                        logger.info(f"✨ Auto-reflection triggered for {user_id}")
                        extractor2 = get_llm()
                        store.generate_reflections(user_id, extractor2.llm)
                except Exception as e:
                    logger.error(f"⚠️ Auto-reflection failed: {e}")

                # ---- Smart Triggers: detect reminders, contradictions, patterns ----
                triggers_created = 0
                try:
                    triggers_created += store.detect_reminder_triggers(user_id)
                    for entity in extraction.entities:
                        if entity.name and entity.facts:
                            triggers_created += store.detect_contradiction_triggers(
                                user_id, entity.facts, entity.name
                            )
                    triggers_created += store.detect_pattern_triggers(user_id)
                    if triggers_created > 0:
                        logger.info(f"🧠 Smart triggers created: {triggers_created} for {user_id}")
                except Exception as e:
                    logger.error(f"⚠️ Smart triggers failed: {e}")

                # ---- Experience-Driven Procedures: detect patterns in episodes ----
                if episodes_created > 0:
                    try:
                        from cloud.evolution import EvolutionEngine
                        evo_engine = EvolutionEngine(store, embedder, extractor.llm)
                        evo_result = evo_engine.detect_and_create_from_episodes(user_id)
                        if evo_result:
                            logger.info(f"🔄 Auto-created procedure '{evo_result['name']}' "
                                       f"from {evo_result['source_episode_count']} episodes")
                    except Exception as e:
                        logger.error(f"⚠️ Experience-driven procedure detection failed: {e}")
            except Exception as e:
                logger.error(f"❌ Background add failed: {e}")
                store.fail_job(job_id, str(e))

        threading.Thread(target=process_in_background, daemon=True).start()

        return {
            "status": "accepted",
            "message": "Processing in background. Memories will appear shortly.",
            "job_id": job_id,
        }

    @app.get("/v1/jobs/{job_id}", tags=["System"])
    async def job_status(job_id: str, user_id: str = Depends(auth)):
        """Check status of a background job."""
        job = store.get_job(job_id, user_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/v1/search", tags=["Search"])
    async def search(req: SearchRequest, user_id: str = Depends(auth)):
        """Semantic search across memories with LLM re-ranking."""
        import hashlib as _hashlib

        # ---- Redis cache: same query → instant response ----
        cache_key = f"search:{user_id}:{_hashlib.md5(f'{req.query}:{req.limit}:{req.graph_depth}'.encode()).hexdigest()}"
        cached = store.cache.get(cache_key)
        if cached:
            store.log_usage(user_id, "search")
            return {"results": cached}

        embedder = get_embedder()

        # Search with more candidates for re-ranking
        search_limit = max(req.limit * 2, 10)

        if embedder:
            emb = embedder.embed(req.query)
            results = store.search_vector_with_teams(user_id, emb, top_k=search_limit,
                                          query_text=req.query, graph_depth=req.graph_depth)
            # Fallback: if nothing found, retry with lower threshold
            if not results:
                results = store.search_vector_with_teams(user_id, emb, top_k=search_limit,
                                              min_score=0.15, query_text=req.query,
                                              graph_depth=req.graph_depth)
        else:
            results = store.search_text(user_id, req.query, top_k=search_limit)

        # Split direct matches from graph-expanded entities
        direct = [r for r in results if not r.get("_graph")]
        graph = [r for r in results if r.get("_graph")]

        # LLM re-ranking: only rerank direct matches (graph entities are logically relevant)
        if direct and len(direct) > 3:
            direct = rerank_results(req.query, direct)

        # Merge: direct first, then graph-expanded
        results = direct + graph

        # Limit to requested count
        results = results[:req.limit]

        # Clean up internal flag
        for r in results:
            r.pop("_graph", None)

        # Prepend matching reflections for richer context
        reflections = store.get_reflections(user_id)
        if reflections:
            query_lower = req.query.lower()
            matching = [r for r in reflections if
                       query_lower in r["content"].lower() or
                       query_lower in r["title"].lower() or
                       any(w in r["content"].lower() for w in query_lower.split() if len(w) > 3)]
            if matching:
                # Add as a special "reflection" result at the top
                top_reflection = matching[0]
                results.insert(0, {
                    "entity": f"✨ Insight: {top_reflection['title']}",
                    "type": "reflection",
                    "scope": top_reflection["scope"],
                    "score": top_reflection["confidence"],
                    "facts": [top_reflection["content"]],
                    "relations": [],
                    "knowledge": [],
                })

        # Cache results in Redis (TTL 30s)
        store.cache.set(cache_key, results, ttl=30)
        store.log_usage(user_id, "search")

        return {"results": results}

    @app.get("/v1/memories", tags=["Memory"])
    async def get_all(user_id_param: str = "default",
                      user_id: str = Depends(auth)):
        """Get all memories (entities)."""
        entities = store.get_all_entities(user_id)
        store.log_usage(user_id, "get_all")
        return {"memories": entities}

    @app.post("/v1/reindex", tags=["Memory"])
    async def reindex(user_id: str = Depends(auth)):
        """Re-generate all embeddings (includes relations now)."""
        embedder = get_embedder()
        if not embedder:
            raise HTTPException(status_code=500, detail="No embedder configured")

        entities = store.get_all_entities_full(user_id)
        count = 0
        for entity in entities:
            name = entity["entity"]
            entity_id = store.get_entity_id(user_id, name)
            if not entity_id:
                continue

            chunks = [name] + entity.get("facts", [])
            for r in entity.get("relations", []):
                target = r.get("target", "")
                rel_type = r.get("type", "")
                if target and rel_type:
                    chunks.append(f"{name} {rel_type} {target}")
            for k in entity.get("knowledge", []):
                chunks.append(f"{k.get('title', '')} {k.get('content', '')}")

            store.delete_embeddings(entity_id)
            embeddings = embedder.embed_batch(chunks)
            for chunk, emb in zip(chunks, embeddings):
                store.save_embedding(entity_id, chunk, emb)
            count += 1

        return {"reindexed": count}

    @app.post("/v1/dedup", tags=["Memory"])
    async def dedup(user_id: str = Depends(auth)):
        """Find and merge duplicate entities."""
        entities = store.get_all_entities(user_id)
        names = [(e["name"], e.get("type", "unknown")) for e in entities]
        merged = []

        # Compare all pairs — find word-boundary matches (e.g. "Ali" + "Ali Baizhanov")
        processed = set()
        for i, (name_a, _) in enumerate(names):
            if name_a in processed:
                continue
            for j, (name_b, _) in enumerate(names):
                if i >= j or name_b in processed:
                    continue
                a_lower = name_a.strip().lower()
                b_lower = name_b.strip().lower()
                # One must start with the other + space, or be equal
                is_match = (
                    b_lower.startswith(a_lower + " ") or
                    a_lower.startswith(b_lower + " ") or
                    a_lower == b_lower
                )
                if is_match:
                    # Merge shorter into longer
                    canonical = name_a if len(name_a) >= len(name_b) else name_b
                    shorter = name_b if canonical == name_a else name_a
                    canon_id = store.get_entity_id(user_id, canonical)
                    short_id = store.get_entity_id(user_id, shorter)
                    if canon_id and short_id and canon_id != short_id:
                        store.merge_entities(user_id, short_id, canon_id, canonical)
                        merged.append(f"{shorter} → {canonical}")
                        processed.add(shorter)

        return {"merged": merged, "count": len(merged)}

    @app.delete("/v1/entity/{name}", tags=["Memory"])
    async def delete_entity(name: str, user_id: str = Depends(auth)):
        """Delete an entity and all its facts, relations, knowledge, embeddings."""
        entity_id = store.get_entity_id(user_id, name)
        if not entity_id:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        with store.conn.cursor() as cur:
            cur.execute("DELETE FROM embeddings WHERE entity_id = %s", (entity_id,))
            cur.execute("DELETE FROM knowledge WHERE entity_id = %s", (entity_id,))
            cur.execute("DELETE FROM facts WHERE entity_id = %s", (entity_id,))
            cur.execute("DELETE FROM relations WHERE source_id = %s OR target_id = %s", (entity_id, entity_id))
            cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
        store.fire_webhooks(user_id, "memory_delete", {"entity": name})
        return {"deleted": name}

    @app.post("/v1/merge_user", tags=["Memory"])
    async def merge_user_entity(user_id: str = Depends(auth)):
        """Merge 'User' entity into the primary person entity (e.g. 'Ali Baizhanov')."""
        user_entity_id = store.get_entity_id(user_id, "User")
        if not user_entity_id:
            return {"status": "skip", "message": "No 'User' entity found"}

        primary = store._find_primary_person(user_id)
        if not primary:
            return {"status": "skip", "message": "No primary person entity to merge into"}

        target_id, target_name = primary
        if user_entity_id == target_id:
            return {"status": "skip", "message": "User IS the primary entity"}

        store.merge_entities(user_id, user_entity_id, target_id, target_name)
        return {"status": "merged", "from": "User", "into": target_name, "target_id": target_id}

    @app.post("/v1/merge", tags=["Memory"])
    async def merge_entities_endpoint(source: str, target: str, user_id: str = Depends(auth)):
        """Merge source entity into target. Source gets deleted, all data moves to target."""
        source_id = store.get_entity_id(user_id, source)
        if not source_id:
            raise HTTPException(status_code=404, detail=f"Source entity '{source}' not found")
        target_id = store.get_entity_id(user_id, target)
        if not target_id:
            raise HTTPException(status_code=404, detail=f"Target entity '{target}' not found")
        if source_id == target_id:
            return {"status": "skip", "message": "Same entity"}
        store.merge_entities(user_id, source_id, target_id, target)
        return {"status": "merged", "from": source, "into": target}

    @app.patch("/v1/entity/{name}/type")
    async def fix_entity_type(name: str, new_type: str, user_id: str = Depends(auth)):
        """Fix entity type (e.g. 'company' → 'technology')."""
        valid_types = {"person", "project", "technology", "company", "concept", "unknown"}
        if new_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid type. Must be one of: {valid_types}")
        entity_id = store.get_entity_id(user_id, name)
        if not entity_id:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        with store.conn.cursor() as cur:
            cur.execute("UPDATE entities SET type = %s WHERE id = %s", (new_type, entity_id))
        return {"entity": name, "new_type": new_type}

    @app.post("/v1/entity/{name}/dedup", tags=["Memory"])
    async def dedup_entity(name: str, user_id: str = Depends(auth)):
        """Use LLM to deduplicate facts on an entity. Keeps best version, archives redundant ones."""
        entity_id = store.get_entity_id(user_id, name)
        if not entity_id:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        extractor = get_llm()
        result = store.dedup_entity_facts(entity_id, name, extractor.llm)
        return result

    @app.post("/v1/dedup_all", tags=["Memory"])
    async def dedup_all_entities(user_id: str = Depends(auth)):
        """Deduplicate facts across ALL entities for this user."""
        entities = store.get_all_entities(user_id)
        extractor = get_llm()
        total_archived = 0
        results = []
        for e in entities:
            entity_id = store.get_entity_id(user_id, e["name"])
            if not entity_id:
                continue
            r = store.dedup_entity_facts(entity_id, e["name"], extractor.llm)
            if r["archived"]:
                total_archived += len(r["archived"])
                results.append({"entity": e["name"], "archived": len(r["archived"])})
        return {"total_archived": total_archived, "entities": results}

    # ---- Reflection ----

    @app.post("/v1/reflect", tags=["Insights"])
    async def trigger_reflection(user_id: str = Depends(auth)):
        """Manually trigger memory reflection. Generates AI insights from facts."""
        extractor = get_llm()
        stats = store.get_reflection_stats(user_id)
        result = store.generate_reflections(user_id, extractor.llm)

        entity_count = len(result.get("entity_reflections", []))
        cross_count = len(result.get("cross_entity", []))
        temporal_count = len(result.get("temporal", []))

        return {
            "status": "reflected",
            "generated": {
                "entity_reflections": entity_count,
                "cross_entity": cross_count,
                "temporal": temporal_count,
            },
            "stats_before": stats,
        }

    @app.get("/v1/reflections", tags=["Insights"])
    async def get_reflections(scope: str = None, user_id: str = Depends(auth)):
        """Get all reflections. Optional ?scope=entity|cross|temporal"""
        return {"reflections": store.get_reflections(user_id, scope=scope)}

    @app.get("/v1/insights", tags=["Insights"])
    async def get_insights(user_id: str = Depends(auth)):
        """Get formatted AI insights for dashboard."""
        return store.get_insights(user_id)

    # =====================================================
    # MEMORY AGENTS v2.0
    # =====================================================

    @app.post("/v1/agents/run", tags=["Agents"])
    async def run_agents(
        agent: str = "all",
        auto_fix: bool = False,
        user_id: str = Depends(auth)
    ):
        """Run memory agents. 
        ?agent=curator|connector|digest|all
        ?auto_fix=true — auto-archive low quality and stale facts (curator only)
        """
        llm = get_llm()

        if agent == "all":
            result = store.run_all_agents(user_id, llm.llm, auto_fix=auto_fix)
            return {"status": "completed", "agents": result}
        elif agent == "curator":
            result = store.run_curator_agent(user_id, llm.llm, auto_fix=auto_fix)
            return {"status": "completed", "agent": "curator", "result": result}
        elif agent == "connector":
            result = store.run_connector_agent(user_id, llm.llm)
            return {"status": "completed", "agent": "connector", "result": result}
        elif agent == "digest":
            result = store.run_digest_agent(user_id, llm.llm)
            return {"status": "completed", "agent": "digest", "result": result}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}. Use: curator, connector, digest, all")

    @app.get("/v1/agents/history", tags=["Agents"])
    async def agent_history(
        agent: str = None,
        limit: int = 10,
        user_id: str = Depends(auth)
    ):
        """Get agent run history. Optional ?agent=curator|connector|digest"""
        runs = store.get_agent_history(user_id, agent_type=agent, limit=limit)
        return {"runs": runs, "total": len(runs)}

    @app.get("/v1/agents/status", tags=["Agents"])
    async def agent_status(user_id: str = Depends(auth)):
        """Check which agents are due to run."""
        due = store.should_run_agents(user_id)
        history = store.get_agent_history(user_id, limit=3)
        return {
            "due": due,
            "last_runs": history
        }

    # =====================================================
    # WEBHOOKS
    # =====================================================

    @app.post("/v1/webhooks", tags=["Webhooks"])
    async def create_webhook(req: dict, user_id: str = Depends(auth)):
        """Create a webhook.
        Body: {"url": "https://...", "name": "My Hook", "event_types": ["memory_add"], "secret": "optional"}
        """
        url = req.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        try:
            hook = store.create_webhook(
                user_id=user_id,
                url=url,
                name=req.get("name", ""),
                event_types=req.get("event_types"),
                secret=req.get("secret", "")
            )
            return {"status": "created", "webhook": hook}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/webhooks", tags=["Webhooks"])
    async def list_webhooks(user_id: str = Depends(auth)):
        """List all webhooks."""
        hooks = store.get_webhooks(user_id)
        return {"webhooks": hooks, "total": len(hooks)}

    @app.put("/v1/webhooks/{webhook_id}", tags=["Webhooks"])
    async def update_webhook(webhook_id: int, req: dict, user_id: str = Depends(auth)):
        """Update a webhook. Body: any of {url, name, event_types, active}"""
        result = store.update_webhook(
            user_id=user_id,
            webhook_id=webhook_id,
            url=req.get("url"),
            name=req.get("name"),
            event_types=req.get("event_types"),
            active=req.get("active")
        )
        return result

    @app.delete("/v1/webhooks/{webhook_id}", tags=["Webhooks"])
    async def delete_webhook(webhook_id: int, user_id: str = Depends(auth)):
        """Delete a webhook."""
        deleted = store.delete_webhook(user_id, webhook_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"status": "deleted", "id": webhook_id}

    # =====================================================
    # TEAMS — SHARED MEMORY
    # =====================================================

    @app.post("/v1/teams", tags=["Teams"])
    async def create_team(req: dict, user_id: str = Depends(auth)):
        """Create a team. Body: {"name": "My Team", "description": "optional"}"""
        name = req.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        team = store.create_team(user_id, name, req.get("description", ""))
        return {"status": "created", "team": team}

    @app.get("/v1/teams", tags=["Teams"])
    async def list_teams(user_id: str = Depends(auth)):
        """List user's teams."""
        teams = store.get_user_teams(user_id)
        return {"teams": teams, "total": len(teams)}

    @app.post("/v1/teams/join", tags=["Teams"])
    async def join_team(req: dict, user_id: str = Depends(auth)):
        """Join a team. Body: {"invite_code": "abc123"}"""
        code = req.get("invite_code")
        if not code:
            raise HTTPException(status_code=400, detail="invite_code is required")
        try:
            result = store.join_team(user_id, code)
            return {"status": "joined", **result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/teams/{team_id}/members", tags=["Teams"])
    async def team_members(team_id: int, user_id: str = Depends(auth)):
        """Get team members."""
        try:
            members = store.get_team_members(user_id, team_id)
            return {"members": members, "total": len(members)}
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))

    @app.post("/v1/teams/{team_id}/share", tags=["Teams"])
    async def share_entity(team_id: int, req: dict, user_id: str = Depends(auth)):
        """Share a memory with team. Body: {"entity": "Redis"}"""
        entity_name = req.get("entity")
        if not entity_name:
            raise HTTPException(status_code=400, detail="entity name is required")
        try:
            return store.share_entity(user_id, entity_name, team_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/teams/{team_id}/unshare", tags=["Teams"])
    async def unshare_entity(team_id: int, req: dict, user_id: str = Depends(auth)):
        """Make a shared memory personal again. Body: {"entity": "Redis"}"""
        entity_name = req.get("entity")
        if not entity_name:
            raise HTTPException(status_code=400, detail="entity name is required")
        return store.unshare_entity(user_id, entity_name)

    @app.post("/v1/teams/{team_id}/leave", tags=["Teams"])
    async def leave_team(team_id: int, user_id: str = Depends(auth)):
        """Leave a team."""
        if store.leave_team(user_id, team_id):
            return {"status": "left"}
        raise HTTPException(status_code=400, detail="Cannot leave (owner or not a member)")

    @app.delete("/v1/teams/{team_id}", tags=["Teams"])
    async def delete_team(team_id: int, user_id: str = Depends(auth)):
        """Delete a team (owner only)."""
        try:
            store.delete_team(user_id, team_id)
            return {"status": "deleted"}
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))

    @app.post("/v1/archive_fact", tags=["Memory"])
    async def archive_fact(
        req: dict,
        user_id: str = Depends(auth)
    ):
        """Manually archive a wrong fact."""
        entity_name = req.get("entity_name")
        fact = req.get("fact_content") or req.get("fact")
        if not entity_name or not fact:
            raise HTTPException(status_code=400, detail="entity_name and fact_content required")
        entity_id = store.get_entity_id(user_id, entity_name)
        if not entity_id:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        with store.conn.cursor() as cur:
            cur.execute(
                """UPDATE facts SET archived = TRUE, superseded_by = 'manually archived'
                   WHERE entity_id = %s AND content = %s AND archived = FALSE""",
                (entity_id, fact)
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Fact not found")
        return {"archived": fact, "entity": entity_name}

    @app.get("/v1/timeline", tags=["Memory"])
    async def timeline(
        after: str = None, before: str = None,
        limit: int = 20,
        user_id: str = Depends(auth)
    ):
        """Temporal search — what happened in a time range?
        after/before: ISO datetime strings (e.g. 2025-02-01T00:00:00Z)"""
        results = store.search_temporal(user_id, after=after, before=before, top_k=limit)
        return {"results": results}

    @app.get("/v1/memories/full", tags=["Memory"])
    async def get_all_full(user_id: str = Depends(auth)):
        """Get all memories with full facts, relations, knowledge. Single query."""
        entities = store.get_all_entities_full(user_id)
        store.log_usage(user_id, "get_all")
        return {"memories": entities}

    @app.get("/v1/memory/{name}", tags=["Memory"])
    async def get_memory(name: str, user_id: str = Depends(auth)):
        """Get specific entity details."""
        entity = store.get_entity(user_id, name)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        return {
            "entity": entity.name,
            "type": entity.type,
            "facts": entity.facts,
            "relations": entity.relations,
            "knowledge": entity.knowledge,
        }

    @app.delete("/v1/memory/{name}", tags=["Memory"])
    async def delete_memory(name: str, user_id: str = Depends(auth)):
        """Delete a memory."""
        deleted = store.delete_entity(user_id, name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        return {"status": "deleted", "entity": name}

    @app.get("/v1/stats", tags=["System"])
    async def stats(user_id: str = Depends(auth)):
        """Usage statistics."""
        return store.get_stats(user_id)

    @app.get("/v1/graph", tags=["Memory"])
    async def graph(user_id: str = Depends(auth)):
        """Knowledge graph for visualization."""
        return store.get_graph(user_id)

    @app.get("/v1/feed", tags=["Memory"])
    async def feed(limit: int = 50, user_id: str = Depends(auth)):
        """Memory feed — recent facts with timestamps for dashboard."""
        return store.get_feed(user_id, limit=min(limit, 100))

    @app.get("/v1/profile/{target_user_id}", tags=["Memory"])
    async def get_profile(target_user_id: str, force: bool = False, user_id: str = Depends(auth)):
        """Cognitive Profile — generates a ready-to-use system prompt from user memory.
        
        Returns a personalization prompt that can be inserted into any LLM.
        Cached for 1 hour. Use force=true to regenerate."""
        return store.get_profile(target_user_id, force=force)

    @app.get("/v1/profile", tags=["Memory"])
    async def get_own_profile(force: bool = False, user_id: str = Depends(auth)):
        """Cognitive Profile for the authenticated user."""
        return store.get_profile(user_id, force=force)

    # ---- Episodic Memory ----

    @app.get("/v1/episodes", tags=["Episodic Memory"])
    async def list_episodes(
        limit: int = 20, after: str = None, before: str = None,
        user_id: str = Depends(auth)
    ):
        """List episodic memories (events, interactions, experiences)."""
        episodes = store.get_episodes(user_id, limit=min(limit, 100),
                                       after=after, before=before)
        return {"episodes": episodes, "count": len(episodes)}

    @app.get("/v1/episodes/search", tags=["Episodic Memory"])
    async def search_episodes(
        query: str, limit: int = 5,
        after: str = None, before: str = None,
        user_id: str = Depends(auth)
    ):
        """Semantic search over episodic memories."""
        embedder = get_embedder()
        if embedder:
            emb = embedder.embed(query)
            results = store.search_episodes_vector(
                user_id, emb, top_k=limit, after=after, before=before)
        else:
            results = store.search_episodes_text(user_id, query, top_k=limit)
        return {"results": results}

    # ---- Procedural Memory ----

    @app.get("/v1/procedures", tags=["Procedural Memory"])
    async def list_procedures(
        limit: int = 20,
        user_id: str = Depends(auth)
    ):
        """List procedural memories (learned workflows, skills)."""
        procedures = store.get_procedures(user_id, limit=min(limit, 100))
        return {"procedures": procedures, "count": len(procedures)}

    @app.get("/v1/procedures/search", tags=["Procedural Memory"])
    async def search_procedures(
        query: str, limit: int = 5,
        user_id: str = Depends(auth)
    ):
        """Semantic search over procedural memories."""
        embedder = get_embedder()
        if embedder:
            emb = embedder.embed(query)
            results = store.search_procedures_vector(user_id, emb, top_k=limit)
        else:
            results = store.search_procedures_text(user_id, query, top_k=limit)
        return {"results": results}

    @app.patch("/v1/procedures/{procedure_id}/feedback", tags=["Procedural Memory"])
    async def procedure_feedback(
        procedure_id: str, success: bool = True,
        body: FeedbackRequest = None,
        user_id: str = Depends(auth)
    ):
        """Record success/failure feedback for a procedure.

        On failure with context, triggers experience-driven evolution:
        creates a linked failure episode and evolves the procedure to a new version.
        """
        result = store.procedure_feedback(user_id, procedure_id, success)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        # Experience-driven evolution: on failure with context, evolve the procedure
        evolution_triggered = False
        if not success and body and body.context:
            import threading

            def evolve_in_background():
                try:
                    # 1. Create a linked failure episode
                    episode_id = store.save_episode(
                        user_id=user_id,
                        summary=f"Procedure '{result['name']}' failed: {body.context[:100]}",
                        context=body.context,
                        outcome="failure",
                        emotional_valence="negative",
                        importance=0.7,
                        linked_procedure_id=procedure_id,
                        failed_at_step=body.failed_at_step,
                    )
                    # Embed the failure episode
                    embedder = get_embedder()
                    if embedder:
                        ep_text = f"Procedure {result['name']} failed. {body.context}"[:2000]
                        ep_embs = embedder.embed_batch([ep_text])
                        if ep_embs:
                            store.save_episode_embedding(episode_id, ep_text, ep_embs[0])

                    # 2. Trigger evolution
                    from cloud.evolution import EvolutionEngine
                    extractor = get_llm()
                    engine = EvolutionEngine(store, embedder, extractor.llm)
                    engine.evolve_on_failure(user_id, procedure_id, episode_id, body.context)
                except Exception as e:
                    logger.error(f"⚠️ Procedure evolution failed: {e}")

            threading.Thread(target=evolve_in_background, daemon=True).start()
            evolution_triggered = True

        result["evolution_triggered"] = evolution_triggered
        return result

    @app.get("/v1/procedures/{procedure_id}/history", tags=["Procedural Memory"])
    async def procedure_history(procedure_id: str, user_id: str = Depends(auth)):
        """Get version history for a procedure. Shows how it evolved over time."""
        history = store.get_procedure_history(user_id, procedure_id)
        if not history:
            raise HTTPException(status_code=404, detail="procedure not found")
        evolution = store.get_procedure_evolution(user_id, procedure_id)
        return {"versions": history, "evolution_log": evolution}

    @app.get("/v1/procedures/{procedure_id}/evolution", tags=["Procedural Memory"])
    async def procedure_evolution(procedure_id: str, user_id: str = Depends(auth)):
        """Get the evolution log for a procedure — what changed and why."""
        evolution = store.get_procedure_evolution(user_id, procedure_id)
        return {"evolution": evolution}

    # ---- Unified Search (all 3 memory types) ----

    @app.post("/v1/search/all", tags=["Search"])
    async def search_all(req: SearchRequest, user_id: str = Depends(auth)):
        """Search across all memory types: semantic, episodic, and procedural.
        Returns categorized results from each memory system."""
        import hashlib as _hashlib

        # ---- Redis cache ----
        cache_key = f"searchall:{user_id}:{_hashlib.md5(f'{req.query}:{req.limit}:{req.graph_depth}'.encode()).hexdigest()}"
        cached = store.cache.get(cache_key)
        if cached:
            store.log_usage(user_id, "search_all")
            return cached

        embedder = get_embedder()
        ep_limit = max(req.limit // 2, 3)
        proc_limit = max(req.limit // 2, 3)

        # Semantic (existing search)
        search_limit = max(req.limit * 2, 10)
        if embedder:
            emb = embedder.embed(req.query)
            semantic = store.search_vector_with_teams(
                user_id, emb, top_k=search_limit, query_text=req.query,
                graph_depth=req.graph_depth)
            if not semantic:
                semantic = store.search_vector_with_teams(
                    user_id, emb, top_k=search_limit, min_score=0.15,
                    query_text=req.query, graph_depth=req.graph_depth)
            # Episodic
            episodic = store.search_episodes_vector(
                user_id, emb, top_k=ep_limit)
            # Procedural
            procedural = store.search_procedures_vector(
                user_id, emb, top_k=proc_limit)
        else:
            semantic = store.search_text(user_id, req.query, top_k=search_limit)
            episodic = store.search_episodes_text(
                user_id, req.query, top_k=ep_limit)
            procedural = store.search_procedures_text(
                user_id, req.query, top_k=proc_limit)

        # Split direct from graph-expanded, rerank only direct
        direct_sem = [r for r in semantic if not r.get("_graph")]
        graph_sem = [r for r in semantic if r.get("_graph")]
        if direct_sem and len(direct_sem) > 3:
            direct_sem = rerank_results(req.query, direct_sem)
        semantic = (direct_sem + graph_sem)[:req.limit]
        for r in semantic:
            r.pop("_graph", None)

        result = {
            "semantic": semantic,
            "episodic": episodic,
            "procedural": procedural,
        }

        # Cache in Redis (TTL 30s)
        store.cache.set(cache_key, result, ttl=30)
        store.log_usage(user_id, "search_all")
        return result

    # ============================================
    # Smart Memory Triggers (v2.6)
    # ============================================

    @app.get("/v1/triggers", tags=["Smart Triggers"])
    async def get_own_triggers(include_fired: bool = False,
                               limit: int = 50, user_id: str = Depends(auth)):
        """Get smart triggers for the authenticated user."""
        triggers = store.get_triggers(user_id, include_fired=include_fired, limit=limit)
        for t in triggers:
            for key in ("fire_at", "fired_at", "created_at"):
                if t.get(key) and hasattr(t[key], "isoformat"):
                    t[key] = t[key].isoformat()
        return {"triggers": triggers, "count": len(triggers)}

    @app.get("/v1/triggers/{target_user_id}", tags=["Smart Triggers"])
    async def get_triggers(target_user_id: str, include_fired: bool = False,
                           limit: int = 50, user_id: str = Depends(auth)):
        """Get smart triggers for a specific user."""
        triggers = store.get_triggers(target_user_id, include_fired=include_fired, limit=limit)
        for t in triggers:
            for key in ("fire_at", "fired_at", "created_at"):
                if t.get(key) and hasattr(t[key], "isoformat"):
                    t[key] = t[key].isoformat()
        return {"triggers": triggers, "count": len(triggers)}

    @app.post("/v1/triggers/process", tags=["Smart Triggers"])
    async def process_triggers(user_id: str = Depends(auth)):
        """Manually process all pending triggers (fire those that are due)."""
        result = store.process_all_triggers()
        return result

    @app.delete("/v1/triggers/{trigger_id}", tags=["Smart Triggers"])
    async def dismiss_trigger(trigger_id: int, user_id: str = Depends(auth)):
        """Dismiss (mark as fired) a specific trigger without sending webhook."""
        store.ensure_triggers_table()
        with store._cursor() as cur:
            cur.execute("""
                UPDATE memory_triggers SET fired = TRUE, fired_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (trigger_id,))
            row = cur.fetchone()
        if row:
            return {"status": "dismissed", "id": trigger_id}
        raise HTTPException(status_code=404, detail="Trigger not found")

    @app.post("/v1/triggers/detect/{target_user_id}", tags=["Smart Triggers"])
    async def detect_triggers_debug(target_user_id: str, user_id: str = Depends(auth)):
        """Manually run trigger detection for a user. Returns detailed results."""
        results = {"reminders": 0, "contradictions": 0, "patterns": 0, "errors": []}
        try:
            results["reminders"] = store.detect_reminder_triggers(target_user_id)
        except Exception as e:
            results["errors"].append(f"reminders: {e}")
        try:
            results["patterns"] = store.detect_pattern_triggers(target_user_id)
        except Exception as e:
            results["errors"].append(f"patterns: {e}")
        triggers = store.get_triggers(target_user_id)
        results["total_pending"] = len(triggers)
        results["triggers"] = triggers
        # Serialize datetimes
        for t in results["triggers"]:
            for key in ("fire_at", "fired_at", "created_at"):
                if t.get(key) and hasattr(t[key], "isoformat"):
                    t[key] = t[key].isoformat()
        return results

    # ---- Background trigger processing (cron) ----
    import threading, time as _time

    def _trigger_cron_loop():
        """Background thread that processes triggers every 5 minutes."""
        _time.sleep(30)  # Initial delay to let server start
        while True:
            try:
                result = store.process_all_triggers()
                if result["fired"] > 0:
                    logger.info(f"🧠 Trigger cron: fired {result['fired']} triggers")
            except Exception as e:
                logger.error(f"⚠️ Trigger cron error: {e}")
            _time.sleep(300)  # Every 5 minutes

    _cron_thread = threading.Thread(target=_trigger_cron_loop, daemon=True)
    _cron_thread.start()
    logger.info("🧠 Smart trigger cron started (every 5 min)")

    return app


# ---- Module-level app for gunicorn ----
# gunicorn cloud.api:app -w 4 -k uvicorn.workers.UvicornWorker
app = create_cloud_api()


# ---- Entry point (local dev) ----

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8420))

    logger.info(f"🧠 Mengram Cloud API")
    logger.info(f"   http://0.0.0.0:{port}")
    logger.info(f"   Docs: http://localhost:{port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
