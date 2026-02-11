"""
Mengram REST API v1.0

FastAPI server for Mengram — integrate with any app, LLM, or frontend.

Endpoints:
  POST /api/remember          — Save knowledge from conversation
  POST /api/remember/text     — Save knowledge from text
  POST /api/recall            — Semantic search (rich context)
  POST /api/search            — Structured search (JSON results)
  GET  /api/recall/all        — Full vault overview
  GET  /api/profile           — User knowledge profile
  GET  /api/knowledge/recent  — Recent knowledge entries
  GET  /api/entity/{name}     — Specific entity details
  GET  /api/stats             — Vault statistics
  GET  /api/graph             — Knowledge graph (nodes + edges)

Usage:
  mengram api                          # start on :8420
  mengram api --port 3000              # custom port
  python -m api.rest_server config.yaml     # direct
"""

import sys
import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from engine.brain import MengramBrain, create_brain, load_config


# --- Request/Response Models ---

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class RememberRequest(BaseModel):
    conversation: list[Message]

class RememberTextRequest(BaseModel):
    text: str

class RecallRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    messages: list[Message]
    system: str = ""
    auto_remember: bool = True

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class RememberResponse(BaseModel):
    status: str
    created: list[str]
    updated: list[str]
    knowledge_count: int

class RecallResponse(BaseModel):
    query: str
    context: str

class SearchResult(BaseModel):
    entity: str
    type: str
    score: float
    facts: list[str]
    relations: list[dict]
    knowledge: list[dict]

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class EntityResponse(BaseModel):
    entity: str
    type: str
    facts: list[str]
    relations: list[dict]
    knowledge: list[dict]

class StatsResponse(BaseModel):
    vault: dict
    graph: dict

class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    facts_count: int
    knowledge_count: int

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    description: str = ""

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# --- API Factory ---

def create_rest_api(brain: MengramBrain) -> "FastAPI":
    """Create FastAPI app with all endpoints."""

    app = FastAPI(
        title="Mengram API",
        description="AI memory as a typed knowledge graph in Obsidian",
        version="1.0.0",
    )

    # CORS — allow all origins for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Chat (main feature for Web UI) ---

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        """
        Chat with LLM + automatic memory.
        
        1. Recalls context from vault based on user's message
        2. Sends to LLM with context in system prompt
        3. Remembers the conversation (extracts entities/knowledge)
        4. Returns response + graph updates
        """
        from engine.extractor.conversation_extractor import MockLLMClient

        if isinstance(brain.llm_client, MockLLMClient):
            raise HTTPException(
                status_code=503,
                detail="LLM not configured. Set provider/api_key in config.yaml"
            )

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        # 1. Get last user message for recall
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        # 2. Recall relevant context
        context = ""
        if last_user:
            try:
                context = brain.recall(last_user, top_k=5)
            except Exception:
                pass

        # 3. Build system prompt with context
        system = req.system or "You are a helpful assistant with access to the user's knowledge base."
        if context:
            system += f"\n\n<memory_context>\n{context}\n</memory_context>"

        # 4. Call LLM
        try:
            response = brain.llm_client.chat(messages, system=system)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

        # 5. Remember (async-ish, non-blocking for response)
        new_entities = []
        if req.auto_remember and len(messages) >= 2:
            try:
                result = brain.remember(messages + [{"role": "assistant", "content": response}])
                new_entities = result.get("created", []) + result.get("updated", [])
            except Exception as e:
                print(f"⚠️  Remember error: {e}", file=sys.stderr)

        return {
            "response": response,
            "context_used": bool(context),
            "new_entities": new_entities,
        }

    # --- Remember ---

    @app.post("/api/remember", response_model=RememberResponse)
    async def remember(req: RememberRequest):
        """Save knowledge from a conversation."""
        conversation = [{"role": m.role, "content": m.content} for m in req.conversation]
        result = brain.remember(conversation)
        return RememberResponse(
            status="ok",
            created=result.get("created", []),
            updated=result.get("updated", []),
            knowledge_count=result.get("knowledge_count", 0),
        )

    @app.post("/api/remember/text", response_model=RememberResponse)
    async def remember_text(req: RememberTextRequest):
        """Save knowledge from plain text."""
        result = brain.remember_text(req.text)
        return RememberResponse(
            status="ok",
            created=result.get("created", []),
            updated=result.get("updated", []),
            knowledge_count=result.get("knowledge_count", 0),
        )

    # --- Recall ---

    @app.post("/api/recall", response_model=RecallResponse)
    async def recall(req: RecallRequest):
        """Semantic search — returns rich context with facts, relations, knowledge."""
        context = brain.recall(req.query, top_k=req.top_k)
        return RecallResponse(query=req.query, context=context)

    @app.get("/api/recall/all")
    async def recall_all():
        """Full vault overview with all entities and knowledge."""
        return {"content": brain.recall_all()}

    # --- Search ---

    @app.post("/api/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        """Structured semantic search — returns JSON results with scores."""
        results = brain.search(req.query, top_k=req.top_k)
        search_results = []
        for r in results:
            search_results.append(SearchResult(
                entity=r.get("entity", ""),
                type=r.get("type", ""),
                score=r.get("score", 0.0),
                facts=r.get("facts", []),
                relations=r.get("relations", []),
                knowledge=r.get("knowledge", []),
            ))
        return SearchResponse(query=req.query, results=search_results)

    # --- Entity ---

    @app.get("/api/entity/{name}", response_model=EntityResponse)
    async def get_entity(name: str):
        """Get specific entity details."""
        data = brain._get_entity_data(name)
        if not data["facts"] and not data["relations"]:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        return EntityResponse(
            entity=data["entity"],
            type=data.get("type", "unknown"),
            facts=data.get("facts", []),
            relations=data.get("relations", []),
            knowledge=data.get("knowledge", []),
        )

    # --- Profile ---

    @app.get("/api/profile")
    async def get_profile():
        """Full user knowledge profile."""
        return {"profile": brain.get_profile()}

    @app.get("/api/knowledge/recent")
    async def get_recent_knowledge(limit: int = 10):
        """Recent knowledge entries across all entities."""
        return {"knowledge": brain.get_recent_knowledge(limit=limit)}

    # --- Stats ---

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """Vault statistics."""
        stats = brain.get_stats()
        return StatsResponse(
            vault=stats.get("vault", {}),
            graph=stats.get("graph", {}),
        )

    # --- Graph ---

    @app.get("/api/graph", response_model=GraphResponse)
    async def get_graph():
        """Knowledge graph — nodes and edges for visualization."""
        vault = Path(brain.vault_path)
        files = list(vault.glob("*.md"))

        nodes = []
        edges = []
        seen_edges = set()

        for f in files:
            data = brain._get_entity_data(f.stem)
            nodes.append(GraphNode(
                id=f.stem,
                name=f.stem,
                type=data.get("type", "unknown"),
                facts_count=len(data.get("facts", [])),
                knowledge_count=len(data.get("knowledge", [])),
            ))

            for rel in data.get("relations", []):
                target = rel.get("target", "")
                rel_type = rel.get("type", "related_to")
                direction = rel.get("direction", "outgoing")

                if direction == "outgoing":
                    edge_key = f"{f.stem}:{rel_type}:{target}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(GraphEdge(
                            source=f.stem,
                            target=target,
                            type=rel_type,
                            description=rel.get("description", ""),
                        ))

        return GraphResponse(nodes=nodes, edges=edges)

    # --- Health ---

    @app.get("/api/health")
    async def health():
        """Health check."""
        return {"status": "ok", "version": "0.6.0"}

    # --- Static files for Web UI ---
    from fastapi.responses import FileResponse

    web_dir = Path(__file__).parent.parent / "web"

    @app.get("/")
    async def index():
        index_file = web_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Mengram API. Web UI not found. Docs at /docs"}

    @app.get("/static/{filepath:path}")
    async def static_files(filepath: str):
        file = web_dir / "static" / filepath
        if file.exists() and file.is_file():
            return FileResponse(file)
        raise HTTPException(status_code=404)

    return app


# --- CLI Entry ---

def main():
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not installed: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Mengram REST API")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8420, help="Port (default: 8420)")
    args = parser.parse_args()

    brain = create_brain(args.config)

    # Warmup vector store
    if brain.use_vectors:
        _ = brain.vector_store

    app = create_rest_api(brain)

    print(f"🧠 Mengram REST API", file=sys.stderr)
    print(f"   http://{args.host}:{args.port}", file=sys.stderr)
    print(f"   Docs: http://localhost:{args.port}/docs", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
