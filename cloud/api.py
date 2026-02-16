"""
Mengram Cloud API Server

Hosted version — PostgreSQL + pgvector backend.
Developers get API key, integrate in 3 lines:

    from mengram import CloudMemory
    m = CloudMemory(api_key="mg-...")
    m.add(messages, user_id="alice")
    results = m.search("database issues", user_id="alice")
"""

import os
import sys
import secrets
from typing import Optional
from pathlib import Path

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

# ---- Models ----

class Message(BaseModel):
    role: str
    content: str

class AddRequest(BaseModel):
    messages: list[Message]
    user_id: str = "default"

class AddTextRequest(BaseModel):
    text: str
    user_id: str = "default"

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 5

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
        title="Mengram Cloud API",
        description="Memory layer for AI apps — hosted",
        version="1.6.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = CloudStore(DATABASE_URL)

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

    # ---- LLM Re-ranking ----

    def rerank_results(query: str, results: list[dict]) -> list[dict]:
        """Use LLM to filter search results for relevance."""
        if not results or len(results) <= 1:
            return results

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            return results

        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)

            # Build candidate list for LLM
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
            # Parse JSON array from response
            import json as json_mod
            # Handle possible markdown wrapping
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            indices = json_mod.loads(text)

            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                filtered = [results[i] for i in indices if 0 <= i < len(results)]
                if filtered:
                    return filtered

            return results  # fallback if parsing fails

        except Exception as e:
            print(f"⚠️ Re-ranking failed, returning raw results: {e}", file=sys.stderr)
            return results

    # ---- Auth middleware ----

    async def auth(authorization: str = Header(...)) -> str:
        """Verify API key, return user_id."""
        key = authorization.replace("Bearer ", "")
        user_id = store.verify_api_key(key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user_id

    # ---- Email helper ----

    def _send_api_key_email(email: str, api_key: str, is_reset: bool = False):
        """Send API key to user via Resend."""
        resend_key = os.environ.get("RESEND_API_KEY")
        if not resend_key:
            print("⚠️  RESEND_API_KEY not set, skipping email", file=sys.stderr)
            return

        try:
            import resend
            resend.api_key = resend_key

            action = "reset" if is_reset else "created"
            subject = f"Your new Mengram API key" if is_reset else "Welcome to Mengram 🧠"

            html = f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:520px;margin:0 auto;padding:40px 24px;color:#e8e8f0;background:#0a0a12;border-radius:16px">
                <div style="text-align:center;margin-bottom:32px">
                    <span style="font-size:36px">🧠</span>
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
            print(f"📧 Email sent to {email} (key {action})", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Email send failed: {e}", file=sys.stderr)

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

    @app.post("/v1/signup", response_model=SignupResponse)
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

    @app.post("/v1/reset-key")
    async def reset_key(req: ResetKeyRequest):
        """Reset API key and send new one to email."""
        user_id = store.get_user_by_email(req.email)
        if not user_id:
            # Don't reveal whether email exists
            return {"message": "If this email is registered, a new API key has been sent."}

        new_key = store.reset_api_key(user_id)
        _send_api_key_email(req.email, new_key, is_reset=True)

        return {"message": "If this email is registered, a new API key has been sent."}

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
  <div class="logo">🧠</div>
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
                print(f"⚠️ Email send failed: {e}", file=sys.stderr)
                return {"ok": False, "error": "Failed to send email"}
        else:
            print(f"⚠️ No RESEND_API_KEY, code for {email}: {code}", file=sys.stderr)

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

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "version": "1.6.0"}

    # ---- Protected endpoints ----

    @app.post("/v1/add")
    async def add(req: AddRequest, user_id: str = Depends(auth)):
        """
        Add memories from conversation.
        Returns immediately, processes in background.
        """
        import threading

        def process_in_background():
            try:
                extractor = get_llm()
                conversation = [{"role": m.role, "content": m.content} for m in req.messages]

                # Get existing entities context for smarter extraction
                existing_context = ""
                try:
                    existing_context = store.get_existing_context(user_id)
                except Exception as e:
                    print(f"⚠️ Context fetch failed: {e}", file=sys.stderr)

                extraction = extractor.extract(conversation, existing_context=existing_context)

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

                    # Conflict resolution — archive contradicted facts
                    existing_id = store.get_entity_id(user_id, name)
                    if existing_id and entity.facts:
                        try:
                            archived = store.archive_contradicted_facts(
                                existing_id, entity.facts, extractor.llm
                            )
                        except Exception as e:
                            print(f"⚠️ Conflict check failed: {e}", file=sys.stderr)

                    entity_id = store.save_entity(
                        user_id=user_id,
                        name=name,
                        type=entity.entity_type,
                        facts=entity.facts,
                        relations=entity_relations,
                        knowledge=entity_knowledge,
                    )

                    # Batch embeddings — one API call instead of N
                    embedder = get_embedder()
                    if embedder:
                        chunks = [name] + entity.facts
                        for r in entity_relations:
                            target = r.get("target", "")
                            rel_type = r.get("type", "")
                            if target and rel_type:
                                chunks.append(f"{name} {rel_type} {target}")
                        for k in entity_knowledge:
                            chunks.append(f"{k['title']} {k['content']}")

                        store.delete_embeddings(entity_id)
                        embeddings = embedder.embed_batch(chunks)
                        for chunk, emb in zip(chunks, embeddings):
                            store.save_embedding(entity_id, chunk, emb)

                store.log_usage(user_id, "add")
                print(f"✅ Background add complete for {user_id}", file=sys.stderr)
            except Exception as e:
                print(f"❌ Background add failed: {e}", file=sys.stderr)

        threading.Thread(target=process_in_background, daemon=True).start()

        return {
            "status": "accepted",
            "message": "Processing in background. Memories will appear shortly.",
        }

    @app.post("/v1/search")
    async def search(req: SearchRequest, user_id: str = Depends(auth)):
        """Semantic search across memories with LLM re-ranking."""
        embedder = get_embedder()

        # Search with more candidates for re-ranking
        search_limit = max(req.limit * 2, 10)

        if embedder:
            emb = embedder.embed(req.query)
            results = store.search_vector(user_id, emb, top_k=search_limit,
                                          query_text=req.query)
            # Fallback: if nothing found, retry with lower threshold
            if not results:
                results = store.search_vector(user_id, emb, top_k=search_limit,
                                              min_score=0.15, query_text=req.query)
        else:
            results = store.search_text(user_id, req.query, top_k=search_limit)

        # LLM re-ranking: filter to only relevant results
        if results and len(results) > 1:
            results = rerank_results(req.query, results)

        # Limit to requested count
        results = results[:req.limit]

        store.log_usage(user_id, "search")

        return {"results": results}

    @app.get("/v1/memories")
    async def get_all(user_id_param: str = "default",
                      user_id: str = Depends(auth)):
        """Get all memories (entities)."""
        entities = store.get_all_entities(user_id)
        store.log_usage(user_id, "get_all")
        return {"memories": entities}

    @app.post("/v1/reindex")
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

    @app.post("/v1/dedup")
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

    @app.delete("/v1/entity/{name}")
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
        return {"deleted": name}

    @app.post("/v1/archive_fact")
    async def archive_fact(
        entity_name: str,
        fact: str,
        user_id: str = Depends(auth)
    ):
        """Manually archive a wrong fact."""
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

    @app.get("/v1/timeline")
    async def timeline(
        after: str = None, before: str = None,
        limit: int = 20,
        user_id: str = Depends(auth)
    ):
        """Temporal search — what happened in a time range?
        after/before: ISO datetime strings (e.g. 2025-02-01T00:00:00Z)"""
        results = store.search_temporal(user_id, after=after, before=before, top_k=limit)
        return {"results": results}

    @app.get("/v1/memories/full")
    async def get_all_full(user_id: str = Depends(auth)):
        """Get all memories with full facts, relations, knowledge. Single query."""
        entities = store.get_all_entities_full(user_id)
        store.log_usage(user_id, "get_all")
        return {"memories": entities}

    @app.get("/v1/memory/{name}")
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

    @app.delete("/v1/memory/{name}")
    async def delete_memory(name: str, user_id: str = Depends(auth)):
        """Delete a memory."""
        deleted = store.delete_entity(user_id, name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        return {"status": "deleted", "entity": name}

    @app.get("/v1/stats")
    async def stats(user_id: str = Depends(auth)):
        """Usage statistics."""
        return store.get_stats(user_id)

    @app.get("/v1/graph")
    async def graph(user_id: str = Depends(auth)):
        """Knowledge graph for visualization."""
        return store.get_graph(user_id)

    @app.get("/v1/feed")
    async def feed(limit: int = 50, user_id: str = Depends(auth)):
        """Memory feed — recent facts with timestamps for dashboard."""
        return store.get_feed(user_id, limit=min(limit, 100))

    return app


# ---- Entry point ----

def main():
    import uvicorn
    app = create_cloud_api()
    port = int(os.environ.get("PORT", 8420))

    print(f"🧠 Mengram Cloud API", file=sys.stderr)
    print(f"   http://0.0.0.0:{port}", file=sys.stderr)
    print(f"   Docs: http://localhost:{port}/docs", file=sys.stderr)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
