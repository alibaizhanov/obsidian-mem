"""
Mengram Evolution Engine — Experience-Driven Procedures (v2.7)

Closed feedback loop between episodic and procedural memory:
- Failure cycle: procedure fails → episode created → LLM analyzes → procedure evolves
- Success cycle: 3+ similar positive episodes → LLM extracts pattern → auto-create procedure
"""

import json
import logging

logger = logging.getLogger("mengram")


# ---- LLM Prompts ----

EVOLVE_ON_FAILURE_PROMPT = """You are a procedure improvement assistant. A user followed a procedure but it failed.

PROCEDURE: {procedure_name}
TRIGGER: {trigger_condition}
CURRENT STEPS:
{steps_text}

FAILURE EPISODE:
- Summary: {episode_summary}
- Context: {episode_context}
- Outcome: {episode_outcome}
- Failed at step: {failed_at_step}

Analyze what went wrong and produce an improved version of the procedure.
You may add steps, remove steps, reorder steps, or modify existing steps.
Keep the procedure practical and concise.

Return ONLY valid JSON (no markdown fences):
{{
  "new_steps": [
    {{"step": 1, "action": "...", "detail": "..."}},
    {{"step": 2, "action": "...", "detail": "..."}}
  ],
  "new_trigger": "updated trigger condition or null if unchanged",
  "change_type": "step_added|step_removed|step_modified|step_reordered",
  "change_description": "Brief description of what changed and why",
  "diff": {{
    "added": ["description of added steps"],
    "removed": ["description of removed steps"],
    "modified": ["description of modified steps"]
  }}
}}"""

DETECT_PATTERN_PROMPT = """You are a workflow extraction assistant. Analyze these successful episodes and extract a common repeatable procedure if one exists.

EPISODES:
{episodes_text}

Rules:
- Only extract a procedure if there is a clear repeatable pattern across 3+ episodes
- The procedure must have 2+ concrete steps
- Name it descriptively based on what the user does
- If no clear pattern exists, return {{"procedure": null}}

Return ONLY valid JSON (no markdown fences):
{{
  "procedure": {{
    "name": "Short descriptive name",
    "trigger": "When to use this procedure",
    "steps": [
      {{"step": 1, "action": "...", "detail": "..."}},
      {{"step": 2, "action": "...", "detail": "..."}}
    ],
    "entities": ["related entity names"]
  }}
}}

If no clear pattern: {{"procedure": null}}"""


class EvolutionEngine:
    """Drives experience-driven procedure evolution.

    Stateless — receives store, embedder, and llm_client as dependencies.
    All methods are designed to run in background threads.
    """

    def __init__(self, store, embedder, llm_client):
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client

    def evolve_on_failure(self, user_id: str, procedure_id: str,
                          episode_id: str, failure_context: str = "") -> dict | None:
        """Analyze a procedure failure and create an improved version.

        Args:
            user_id: The user who owns the procedure.
            procedure_id: ID of the failed procedure (current version).
            episode_id: ID of the failure episode.
            failure_context: Additional context about what went wrong.

        Returns:
            Dict with evolution result, or None if evolution failed.
        """
        # 1. Fetch current procedure
        proc = self.store.get_procedure_by_id(user_id, procedure_id)
        if not proc:
            logger.error(f"Evolution failed: procedure {procedure_id} not found")
            return None

        # 2. Fetch the failure episode
        episode = None
        for ep in self.store.get_episodes(user_id, limit=50):
            if ep["id"] == episode_id:
                episode = ep
                break
        if not episode:
            logger.error(f"Evolution failed: episode {episode_id} not found")
            return None

        # 3. Build LLM prompt
        steps_text = "\n".join(
            f"  Step {s.get('step', i+1)}: {s.get('action', '')} — {s.get('detail', '')}"
            for i, s in enumerate(proc["steps"])
        )
        failed_step = episode.get("failed_at_step")
        if failed_step is None and failure_context:
            failed_step = "unknown"

        prompt = EVOLVE_ON_FAILURE_PROMPT.format(
            procedure_name=proc["name"],
            trigger_condition=proc["trigger_condition"] or "N/A",
            steps_text=steps_text or "(no steps)",
            episode_summary=episode["summary"],
            episode_context=episode.get("context") or failure_context or "N/A",
            episode_outcome=episode.get("outcome") or "failure",
            failed_at_step=failed_step or "unknown",
        )

        # 4. Call LLM
        try:
            raw = self.llm_client.complete(prompt)
            result = self._parse_json(raw)
            if not result or not result.get("new_steps"):
                logger.warning("Evolution LLM returned no new steps")
                return None
        except Exception as e:
            logger.error(f"Evolution LLM call failed: {e}")
            return None

        # 5. Create evolved procedure
        try:
            new_proc_id = self.store.evolve_procedure(
                user_id=user_id,
                procedure_id=procedure_id,
                new_steps=result["new_steps"],
                new_trigger=result.get("new_trigger"),
                episode_id=episode_id,
                change_type=result.get("change_type", "step_modified"),
                diff=result.get("diff", {}),
            )

            # 6. Re-embed the new version
            if self.embedder:
                steps_summary = "; ".join(
                    s.get("action", "") for s in result["new_steps"][:10]
                )
                text = f"{proc['name']}. {result.get('new_trigger') or proc['trigger_condition'] or ''}. Steps: {steps_summary}"
                embs = self.embedder.embed_batch([text])
                if embs:
                    self.store.delete_procedure_embeddings(new_proc_id)
                    self.store.save_procedure_embedding(new_proc_id, text, embs[0])

            logger.info(f"✅ Procedure evolved: {proc['name']} v{proc['version']} → v{proc['version'] + 1}")
            return {
                "new_procedure_id": new_proc_id,
                "old_version": proc["version"],
                "new_version": proc["version"] + 1,
                "change_type": result.get("change_type", "step_modified"),
                "change_description": result.get("change_description", ""),
            }

        except Exception as e:
            logger.error(f"Evolution procedure creation failed: {e}")
            return None

    def detect_and_create_from_episodes(self, user_id: str) -> dict | None:
        """Find clusters of similar positive episodes and auto-create procedures.

        Looks for 3+ positive episodes that aren't linked to any procedure,
        uses embedding similarity to cluster them, then asks LLM to extract
        a common workflow.

        Returns:
            Dict with created procedure info, or None if no pattern found.
        """
        # 1. Get unlinked positive episodes
        episodes = self.store.get_unlinked_positive_episodes(user_id, limit=30)
        if len(episodes) < 3:
            return None

        # 2. Try to find clusters using embeddings
        if self.embedder:
            clusters = self._cluster_episodes_by_embedding(episodes)
        else:
            # Fallback: treat all episodes as one group
            clusters = [episodes] if len(episodes) >= 3 else []

        # 3. For each cluster >= 3, try to extract a procedure
        for cluster in clusters:
            if len(cluster) < 3:
                continue

            episodes_text = "\n\n".join(
                f"Episode {i+1}:\n"
                f"  Summary: {ep['summary']}\n"
                f"  Context: {ep.get('context') or 'N/A'}\n"
                f"  Outcome: {ep.get('outcome') or 'N/A'}"
                for i, ep in enumerate(cluster[:8])  # Limit to 8 to keep prompt manageable
            )

            prompt = DETECT_PATTERN_PROMPT.format(episodes_text=episodes_text)

            try:
                raw = self.llm_client.complete(prompt)
                result = self._parse_json(raw)
                if not result or not result.get("procedure"):
                    continue

                proc_data = result["procedure"]
                if not proc_data.get("name") or not proc_data.get("steps"):
                    continue

                # 4. Create the procedure
                episode_ids = [ep["id"] for ep in cluster]
                proc_id = self.store.save_procedure(
                    user_id=user_id,
                    name=proc_data["name"],
                    trigger_condition=proc_data.get("trigger"),
                    steps=proc_data["steps"],
                    entity_names=proc_data.get("entities", []),
                    source_episode_ids=episode_ids,
                )

                # 5. Embed the new procedure
                if self.embedder:
                    steps_summary = "; ".join(
                        s.get("action", "") for s in proc_data["steps"][:10]
                    )
                    text = f"{proc_data['name']}. {proc_data.get('trigger', '')}. Steps: {steps_summary}"
                    embs = self.embedder.embed_batch([text])
                    if embs:
                        self.store.save_procedure_embedding(proc_id, text, embs[0])

                # 6. Link episodes to the new procedure
                self.store.link_episodes_to_procedure(episode_ids, proc_id)

                # 7. Log evolution
                with self.store._cursor() as cur:
                    cur.execute(
                        """INSERT INTO procedure_evolution
                           (procedure_id, change_type, diff, version_before, version_after)
                           VALUES (%s, %s, %s::jsonb, %s, %s)""",
                        (proc_id, "auto_created",
                         json.dumps({"source_episodes": len(episode_ids)}),
                         0, 1)
                    )

                logger.info(f"🆕 Auto-created procedure from {len(episode_ids)} episodes: {proc_data['name']}")
                return {
                    "procedure_id": proc_id,
                    "name": proc_data["name"],
                    "source_episode_count": len(episode_ids),
                    "steps_count": len(proc_data["steps"]),
                }

            except Exception as e:
                logger.error(f"Pattern detection failed: {e}")
                continue

        return None

    def _cluster_episodes_by_embedding(self, episodes: list[dict],
                                       similarity_threshold: float = 0.75) -> list[list[dict]]:
        """Cluster episodes by embedding similarity using a simple greedy approach.

        For each episode, compute embedding and group with the most similar existing cluster.
        """
        if not episodes:
            return []

        # Embed all episode summaries
        texts = [
            f"{ep['summary']}. {ep.get('context') or ''}"[:500]
            for ep in episodes
        ]
        embeddings = self.embedder.embed_batch(texts)
        if not embeddings or len(embeddings) != len(episodes):
            return [episodes]  # Fallback: single cluster

        # Greedy clustering
        clusters = []  # List of (centroid_embedding, episodes_list)

        for ep, emb in zip(episodes, embeddings):
            best_cluster = None
            best_sim = -1

            for i, (centroid, _) in enumerate(clusters):
                sim = self._cosine_similarity(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = i

            if best_cluster is not None and best_sim >= similarity_threshold:
                clusters[best_cluster][1].append(ep)
            else:
                clusters.append((emb, [ep]))

        return [eps for _, eps in clusters]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse evolution LLM response: {text[:200]}")
            return None
