"""
Mengram Brain — главный оркестратор.

Объединяет все компоненты:
1. Conversation Extractor → извлекает знания из разговоров
2. Vault Manager → записывает знания в .md файлы
3. Vector Store → семантический поиск (embeddings)
4. Knowledge Graph → граф связей + graph expansion
5. Hybrid Retrieval → vector + graph = лучший recall

Два основных действия:
- remember(conversation) → извлечь, сохранить, индексировать
- recall(query) → семантический поиск + граф → контекст
"""

import re
import yaml
import sys
from pathlib import Path
from typing import Optional

from engine.extractor.llm_client import LLMClient, create_llm_client
from engine.extractor.conversation_extractor import ConversationExtractor, MockLLMClient
from engine.vault_manager.vault_manager import VaultManager
from engine.graph.knowledge_graph import build_graph_from_vault, KnowledgeGraph
from engine.parser.markdown_parser import parse_vault


class MengramBrain:
    """
    Главный класс — "мозг".

    brain.remember(conversation) → извлекает знания → пишет в vault → индексирует
    brain.recall(query) → семантический поиск + граф → контекст для LLM
    """

    def __init__(self, vault_path: str, llm_client: Optional[LLMClient] = None,
                 use_vectors: bool = True, vector_db_path: Optional[str] = None):
        self.vault_path = vault_path
        self.vault_manager = VaultManager(vault_path)
        self.llm_client = llm_client or MockLLMClient()
        self.extractor = ConversationExtractor(self.llm_client)
        self.use_vectors = use_vectors

        # Graph — ленивая загрузка
        self._graph: Optional[KnowledgeGraph] = None

        # Vector Store — ленивая загрузка
        self._vector_store = None
        self._vector_db_path = vector_db_path or str(Path(vault_path) / ".vectors.db")

    @property
    def graph(self) -> KnowledgeGraph:
        if self._graph is None:
            self._rebuild_graph()
        return self._graph

    @property
    def vector_store(self):
        if self._vector_store is None and self.use_vectors:
            self._init_vector_store()
        return self._vector_store

    def _init_vector_store(self):
        """Инициализация vector store с embeddings"""
        try:
            from engine.vector.embedder import Embedder
            from engine.vector.vector_store import VectorStore

            print("🧠 Инициализирую semantic search...", file=sys.stderr)
            embedder = Embedder()
            self._vector_store = VectorStore(
                db_path=self._vector_db_path,
                embedder=embedder,
            )

            # Auto-sync: index only new/missing entities
            stats = self._vector_store.stats()
            vault_notes = list(Path(self.vault_path).glob("*.md"))
            indexed_entities = stats.get("total_entities", 0)

            if vault_notes and stats["total_chunks"] == 0:
                print("📝 Первичная индексация vault...", file=sys.stderr)
                self._reindex_vault()
            elif len(vault_notes) > indexed_entities:
                # Find which entities are missing
                indexed_ids = set()
                try:
                    rows = self._vector_store.conn.execute(
                        "SELECT DISTINCT entity_name FROM chunks"
                    ).fetchall()
                    indexed_ids = {r[0] for r in rows}
                except Exception:
                    pass
                missing = [f.stem for f in vault_notes if f.stem not in indexed_ids]
                if missing:
                    print(f"📝 Индексирую {len(missing)} новых заметок...", file=sys.stderr)
                    self._index_entities(missing)
                    stats = self._vector_store.stats()
                    print(f"✅ Semantic search готов ({stats['total_chunks']} chunks)", file=sys.stderr)
                else:
                    print(f"✅ Semantic search готов ({stats['total_chunks']} chunks)", file=sys.stderr)
            else:
                print(f"✅ Semantic search готов ({stats['total_chunks']} chunks)", file=sys.stderr)

        except ImportError as e:
            print(f"⚠️  sentence-transformers не установлен: {e}", file=sys.stderr)
            print("   pip install sentence-transformers", file=sys.stderr)
            self.use_vectors = False
            self._vector_store = None

    def remember(self, conversation: list[dict]) -> dict:
        """
        Запомнить знания из разговора.

        1. Извлекает entities/facts/relations через LLM
        2. Записывает в vault (.md файлы)
        3. Индексирует новые данные для semantic search
        """
        print("🧠 Извлекаю знания из разговора...", file=sys.stderr)

        # 1. Извлекаем через LLM
        extraction = self.extractor.extract(conversation)
        print(f"   📊 Найдено: {len(extraction.entities)} entities, {len(extraction.relations)} relations, {len(extraction.knowledge)} knowledge", file=sys.stderr)

        # 2. Записываем в vault
        stats = self.vault_manager.process_extraction(extraction)
        print(f"   📝 Создано: {stats['created']}", file=sys.stderr)
        print(f"   📝 Обновлено: {stats['updated']}", file=sys.stderr)

        # 3. Инвалидируем граф
        self._graph = None

        # 4. Обновляем vector index
        changed = stats["created"] + stats["updated"]
        if changed and self.use_vectors:
            self._index_entities(changed)

        return {
            "entities_created": stats["created"],
            "entities_updated": stats["updated"],
            "extraction": extraction,
        }

    def remember_text(self, text: str) -> dict:
        conversation = [{"role": "user", "content": text}]
        return self.remember(conversation)

    def recall(self, query: str, top_k: int = 5) -> str:
        """
        Вспомнить контекст по запросу.

        Hybrid strategy:
        1. Semantic search → top-K чанков по смыслу
        2. Graph expansion → связанные entities
        3. Fallback → graph text search → raw text search
        """
        contexts = []

        # === 1. SEMANTIC SEARCH ===
        if self.use_vectors and self.vector_store:
            try:
                results = self.vector_store.search(query, top_k=top_k, min_score=0.25)
                if results:
                    seen = set()
                    for r in results:
                        if r.entity_name not in seen:
                            ctx = self._build_rich_context(r.entity_name, r.score)
                            if ctx:
                                contexts.append(ctx)
                                seen.add(r.entity_name)

                    # Graph expansion от top результата
                    if results and self._graph is not None:
                        expanded = self._expand_via_graph(results[0].entity_name, seen)
                        contexts.extend(expanded)

                    if contexts:
                        return self._assemble_context(query, contexts)
            except Exception as e:
                print(f"⚠️  Vector search error: {e}", file=sys.stderr)

        # === 2. GRAPH SEARCH ===
        graph = self.graph

        entity = graph.find_entity(query)
        if entity:
            ctx = self._build_entity_context(entity.id)
            if ctx:
                return ctx

        entities = graph.search_entities(query)
        if entities:
            for e in entities[:top_k]:
                ctx = self._build_entity_context(e.id)
                if ctx:
                    contexts.append(ctx)
            if contexts:
                return "\n\n---\n\n".join(contexts)

        # === 3. TEXT SEARCH ===
        notes = parse_vault(self.vault_path)
        query_lower = query.lower()
        for note in notes:
            if query_lower in note.raw_content.lower():
                contexts.append(f"**{note.title}**:\n{note.raw_content[:500]}")

        if contexts:
            return "\n\n---\n\n".join(contexts[:top_k])

        return f"Ничего не найдено по запросу: '{query}'"

    def recall_all(self) -> str:
        """Full vault overview with knowledge entries."""
        vault = Path(self.vault_path)
        files = sorted(vault.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return "Vault is empty. No knowledge saved yet."

        lines = [f"# Knowledge vault ({len(files)} entities)\n"]
        for f in files:
            data = self._get_entity_data(f.stem)
            lines.append(f"## {f.stem} ({data.get('type', 'unknown')})")

            if data["facts"]:
                for fact in data["facts"][:5]:
                    lines.append(f"- {fact}")

            if data["relations"]:
                for r in data["relations"][:5]:
                    arrow = "→" if r["direction"] == "outgoing" else "←"
                    lines.append(f"- {arrow} {r['type']}: {r['target']}")

            if data["knowledge"]:
                lines.append("\nKnowledge:")
                for k in data["knowledge"]:
                    lines.append(f"  **[{k['type']}] {k['title']}**")
                    lines.append(f"  {k['content'][:300]}")
                    if k.get("artifact"):
                        lines.append(f"  ```\n  {k['artifact'][:500]}\n  ```")

            lines.append("")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search — структурированные результаты для SDK.
        
        Returns:
            [{"entity": "...", "type": "...", "score": 0.85, "facts": [...], "relations": [...]}]
        """
        results = []

        if self.use_vectors and self.vector_store:
            try:
                vresults = self.vector_store.search(query, top_k=top_k, min_score=0.2)
                seen = set()
                for vr in vresults:
                    if vr.entity_name in seen:
                        continue
                    seen.add(vr.entity_name)
                    data = self._get_entity_data(vr.entity_name)
                    data["score"] = round(vr.score, 3)
                    results.append(data)
            except Exception as e:
                print(f"⚠️  Search error: {e}", file=sys.stderr)

        if not results:
            graph = self.graph
            entities = graph.search_entities(query)
            for e in entities[:top_k]:
                data = self._get_entity_data(e.name)
                data["score"] = 0.5
                results.append(data)

        return results

    def get_profile(self) -> str:
        """
        Generate comprehensive user profile from vault.
        Used as MCP resource for proactive context.
        """
        vault = Path(self.vault_path)
        files = sorted(vault.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return "Memory vault is empty. No user context available yet."

        sections = []
        entities_by_type = {}

        for f in files:
            data = self._get_entity_data(f.stem)
            etype = data.get("type", "unknown")
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(data)

        # Build profile
        sections.append("# User Knowledge Profile\n")

        for etype in ["person", "company", "project", "technology", "concept"]:
            entities = entities_by_type.get(etype, [])
            if not entities:
                continue
            plural = {"person": "People", "company": "Companies", "project": "Projects",
                      "technology": "Technologies", "concept": "Concepts"}.get(etype, etype.title() + "s")
            sections.append(f"\n## {plural}")
            for e in entities[:10]:
                name = e["entity"]
                facts = e.get("facts", [])[:5]
                knowledge = e.get("knowledge", [])
                rels = e.get("relations", [])[:5]

                lines = [f"\n### {name}"]
                if facts:
                    for fact in facts:
                        lines.append(f"- {fact}")
                if knowledge:
                    for k in knowledge[:3]:
                        lines.append(f"- [{k['type']}] {k['title']}: {k['content'][:150]}")
                        if k.get("artifact"):
                            lines.append(f"  ```{k['artifact'][:200]}```")
                if rels:
                    for r in rels:
                        arrow = "→" if r["direction"] == "outgoing" else "←"
                        lines.append(f"- {arrow} {r['type']}: {r['target']}")
                sections.append("\n".join(lines))

        return "\n".join(sections)

    def get_recent_knowledge(self, limit: int = 10) -> str:
        """Get most recent knowledge entries across all entities."""
        vault = Path(self.vault_path)
        files = sorted(vault.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)

        all_knowledge = []
        for f in files[:20]:
            data = self._get_entity_data(f.stem)
            for k in data.get("knowledge", []):
                k["_entity"] = f.stem
                all_knowledge.append(k)

        if not all_knowledge:
            return "No knowledge entries yet."

        lines = ["# Recent Knowledge\n"]
        for k in all_knowledge[:limit]:
            lines.append(f"**[{k['type']}] {k['title']}** → {k['_entity']}")
            lines.append(k['content'][:200])
            if k.get("artifact"):
                lines.append(f"```{k['artifact'][:300]}```")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        vault_stats = self.vault_manager.get_vault_stats()
        graph_stats = self.graph.stats() if self._graph else {"total_entities": "?", "total_relations": "?"}
        stats = {"vault": vault_stats, "graph": graph_stats}
        if self.use_vectors and self._vector_store:
            stats["vectors"] = self._vector_store.stats()
        return stats

    # --- Internal ---

    def _get_entity_data(self, entity_name: str) -> dict:
        data = {"entity": entity_name, "type": "unknown", "facts": [], "relations": [], "knowledge": []}
        file_path = Path(self.vault_path) / f"{entity_name}.md"
        if not file_path.exists():
            return data

        content = file_path.read_text(encoding="utf-8")
        body = content

        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if fm_match:
            try:
                fm = yaml.safe_load(fm_match.group(1)) or {}
                data["type"] = fm.get("type", "unknown")
            except:
                pass
            body = content[fm_match.end():]

        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("- ") and "**" not in line:
                fact = re.sub(r"\[\[([^\]]+)\]\]", r"\1", line[2:])
                data["facts"].append(fact)

        for line in body.split("\n"):
            line = line.strip()
            if ("→ **" in line or "← **" in line) and "[[" in line:
                rel_match = re.search(r"(→|←)\s+\*\*(\w+)\*\*\s+\[\[([^\]]+)\]\]", line)
                if rel_match:
                    direction, rel_type, target = rel_match.groups()
                    data["relations"].append({
                        "type": rel_type,
                        "target": target,
                        "direction": "outgoing" if direction == "→" else "incoming",
                    })

        # Extract knowledge entries
        knowledge_matches = re.findall(
            r"\*\*\[(\w+)\]\s+(.+?)\*\*.*?\n(.*?)(?=\n\*\*\[|\n## |\Z)",
            body, re.DOTALL
        )
        for k_type, k_title, k_body in knowledge_matches:
            k_body = k_body.strip()
            # Separate content from artifact (code block)
            artifact = None
            code_match = re.search(r"```\w*\n(.*?)```", k_body, re.DOTALL)
            if code_match:
                artifact = code_match.group(1).strip()
                k_content = k_body[:code_match.start()].strip()
            else:
                k_content = k_body
            k_content = re.sub(r"\[\[([^\]]+)\]\]", r"\1", k_content)
            data["knowledge"].append({
                "type": k_type,
                "title": k_title,
                "content": k_content,
                "artifact": artifact,
            })

        return data

    def _build_rich_context(self, entity_name: str, score: float = 0.0) -> Optional[str]:
        data = self._get_entity_data(entity_name)
        if not data["facts"] and not data["relations"] and not data["knowledge"]:
            return None

        lines = [f"## {entity_name} ({data['type']}) [relevance: {score:.2f}]"]
        for fact in data["facts"][:10]:
            lines.append(f"- {fact}")
        if data["relations"]:
            lines.append("\nRelations:")
            for rel in data["relations"][:8]:
                arrow = "→" if rel["direction"] == "outgoing" else "←"
                lines.append(f"  {arrow} {rel['type']}: {rel['target']}")
        if data["knowledge"]:
            lines.append("\nKnowledge:")
            for k in data["knowledge"][:5]:
                lines.append(f"  [{k['type']}] {k['title']}: {k['content'][:200]}")
                if k.get("artifact"):
                    # Include artifact truncated
                    artifact = k["artifact"][:300]
                    lines.append(f"    ```{artifact}```")
        return "\n".join(lines)

    def _expand_via_graph(self, entity_name: str, seen: set) -> list[str]:
        expanded = []
        try:
            entity = self.graph.find_entity(entity_name)
            if not entity:
                return []
            neighbors = self.graph.get_neighbors(entity.id, depth=1)
            for n in neighbors:
                name = n["entity"].name
                if name not in seen and n["entity"].entity_type != "tag":
                    ctx = self._build_rich_context(name, score=0.0)
                    if ctx:
                        expanded.append(ctx)
                        seen.add(name)
                        if len(expanded) >= 3:
                            break
        except Exception:
            pass
        return expanded

    def _assemble_context(self, query: str, contexts: list[str]) -> str:
        header = f"# Контекст из памяти (запрос: '{query}')\n"
        return header + "\n\n---\n\n".join(contexts)

    def _build_entity_context(self, entity_id: str) -> str:
        entity = self.graph.get_entity(entity_id)
        if not entity:
            return ""

        lines = [f"## {entity.name} ({entity.entity_type})"]
        neighbors = self.graph.get_neighbors(entity_id, depth=1)
        if neighbors:
            lines.append("\nСвязи:")
            for n in neighbors:
                if n["entity"].entity_type != "tag":
                    lines.append(f"  → {n['relation_type']}: {n['entity'].name}")

        if entity.source_file:
            try:
                content = Path(entity.source_file).read_text(encoding="utf-8")
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        content = parts[2]
                lines.append(f"\nЗаметка:\n{content.strip()[:500]}")
            except Exception:
                pass
        return "\n".join(lines)

    def _rebuild_graph(self):
        print("🔄 Обновляю граф знаний...", file=sys.stderr)
        self._graph = build_graph_from_vault(self.vault_path)
        stats = self._graph.stats()
        print(f"   ✅ {stats['total_entities']} entities, {stats['total_relations']} relations", file=sys.stderr)

    def _reindex_vault(self):
        if not self._vector_store:
            return
        notes = parse_vault(self.vault_path)
        if not notes:
            return

        print(f"📝 Индексирую {len(notes)} заметок...", file=sys.stderr)
        all_chunks = []
        for note in notes:
            entity_id = note.name.lower().replace(" ", "_")
            for chunk in note.chunks:
                all_chunks.append({
                    "chunk_id": f"{entity_id}:{chunk.position}",
                    "entity_id": entity_id,
                    "entity_name": note.name,
                    "section": chunk.section,
                    "content": chunk.content,
                    "position": chunk.position,
                })
        if all_chunks:
            self._vector_store.add_chunks_batch(all_chunks)
            stats = self._vector_store.stats()
            print(f"✅ Indexed: {stats['total_chunks']} chunks", file=sys.stderr)

    def _index_entities(self, entity_names: list[str]):
        if not self._vector_store:
            self._init_vector_store()
            return

        from engine.parser.markdown_parser import parse_note as parse_note_file

        chunks = []
        for name in entity_names:
            file_path = Path(self.vault_path) / f"{name}.md"
            if not file_path.exists():
                continue
            try:
                note = parse_note_file(str(file_path))
                if not note:
                    continue
                entity_id = note.name.lower().replace(" ", "_")
                self._vector_store.conn.execute(
                    "DELETE FROM chunks WHERE entity_id = ?", (entity_id,)
                )
                for chunk in note.chunks:
                    chunks.append({
                        "chunk_id": f"{entity_id}:{chunk.position}",
                        "entity_id": entity_id,
                        "entity_name": note.name,
                        "section": chunk.section,
                        "content": chunk.content,
                        "position": chunk.position,
                    })
            except Exception as e:
                print(f"⚠️  Error indexing {name}: {e}", file=sys.stderr)

        if chunks:
            self._vector_store.add_chunks_batch(chunks)
            print(f"   🔍 Indexed {len(chunks)} chunks for {len(entity_names)} entities", file=sys.stderr)


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {"vault_path": "./vault", "llm": {"provider": "mock"}}
    with open(path) as f:
        return yaml.safe_load(f)


def create_brain(config_path: str = "config.yaml") -> MengramBrain:
    config = load_config(config_path)
    vault_path = config.get("vault_path", "./vault")

    llm_config = config.get("llm", {})
    if llm_config.get("provider") == "mock":
        llm_client = MockLLMClient()
    else:
        llm_client = create_llm_client(llm_config)

    use_vectors = config.get("semantic_search", {}).get("enabled", True)

    return MengramBrain(
        vault_path=vault_path,
        llm_client=llm_client,
        use_vectors=use_vectors,
    )
