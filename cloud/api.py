"""
Mengram Cloud API Server

Hosted version — PostgreSQL + pgvector backend.
Developers get API key, integrate in 3 lines:

    from mengram import CloudMemory
    m = CloudMemory(api_key="om-...")
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

    # ---- Public endpoints ----

    @app.get("/", response_class=HTMLResponse)
    async def landing():
        """Landing page."""
        landing_path = Path(__file__).parent / "landing.html"
        return landing_path.read_text(encoding="utf-8")

    @app.post("/v1/signup", response_model=SignupResponse)
    async def signup(req: SignupRequest):
        """Create account and get API key."""
        existing = store.get_user_by_email(req.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        user_id = store.create_user(req.email)
        api_key = store.create_api_key(user_id)

        return SignupResponse(
            api_key=api_key,
            message="Save this key — it won't be shown again."
        )

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    # ---- Protected endpoints ----

    @app.post("/v1/add")
    async def add(req: AddRequest, user_id: str = Depends(auth)):
        """
        Add memories from conversation.
        Extracts entities, facts, relations, knowledge automatically.
        """
        extractor = get_llm()

        conversation = [{"role": m.role, "content": m.content} for m in req.messages]
        extraction = extractor.extract(conversation)

        created = []
        updated = []
        knowledge_count = 0

        for entity in extraction.entities:
            name = entity.name
            if not name:
                continue

            # Check if exists
            existing = store.get_entity(user_id, name)

            # Build relations list for store
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

            # Build knowledge list for store
            entity_knowledge = []
            for k in extraction.knowledge:
                if k.entity == name:
                    entity_knowledge.append({
                        "type": k.knowledge_type,
                        "title": k.title,
                        "content": k.content,
                        "artifact": k.artifact,
                    })

            entity_id = store.save_entity(
                user_id=user_id,
                name=name,
                type=entity.entity_type,
                facts=entity.facts,
                relations=entity_relations,
                knowledge=entity_knowledge,
            )

            if existing:
                updated.append(name)
            else:
                created.append(name)

            knowledge_count += len(entity_knowledge)

            # Generate embeddings
            embedder = get_embedder()
            if embedder:
                chunks = [name] + entity.facts
                for k in entity_knowledge:
                    chunks.append(f"{k['title']} {k['content']}")

                store.delete_embeddings(entity_id)
                for chunk in chunks:
                    emb = embedder.embed(chunk)
                    store.save_embedding(entity_id, chunk, emb)

        store.log_usage(user_id, "add")

        return {
            "status": "ok",
            "created": created,
            "updated": updated,
            "knowledge_count": knowledge_count,
        }

    @app.post("/v1/search")
    async def search(req: SearchRequest, user_id: str = Depends(auth)):
        """Semantic search across memories."""
        embedder = get_embedder()

        if embedder:
            emb = embedder.embed(req.query)
            results = store.search_vector(user_id, emb, top_k=req.limit)
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
