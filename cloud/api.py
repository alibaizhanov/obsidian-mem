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
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
        version="1.0.0",
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

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Web dashboard."""
        dashboard_path = Path(__file__).parent / "dashboard.html"
        return dashboard_path.read_text(encoding="utf-8")

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

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

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
                extraction = extractor.extract(conversation)

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
        """Semantic search across memories."""
        embedder = get_embedder()

        if embedder:
            emb = embedder.embed(req.query)
            results = store.search_vector(user_id, emb, top_k=req.limit)
            # Fallback: if nothing found, retry with lower threshold
            if not results:
                results = store.search_vector(user_id, emb, top_k=req.limit, min_score=0.25)
        else:
            results = store.search_text(user_id, req.query, top_k=req.limit)

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
