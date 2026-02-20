"""
Mengram Importer — load existing knowledge from external sources.

Supports:
  - ChatGPT export (ZIP with conversations.json)
  - Obsidian vault (directory of .md files)
  - Plain text/markdown files

All importers accept an `add_fn` callable so they work with both
local brain.remember() and cloud CloudMemory.add().

Usage (CLI):
    mengram import chatgpt ~/Downloads/chatgpt-export.zip
    mengram import obsidian ~/Documents/MyVault
    mengram import files notes/*.md

Usage (Python):
    from importer import import_chatgpt, import_obsidian, import_files
    result = import_chatgpt("export.zip", add_fn=brain.remember)
"""

import os
import json
import time
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ImportResult:
    """Result of an import operation."""
    conversations_found: int = 0
    chunks_sent: int = 0
    entities_created: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    duration_seconds: float = 0.0


class RateLimiter:
    """Simple rate limiter — tracks call timestamps."""

    def __init__(self, max_per_minute: int = 100):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []

    def wait_if_needed(self):
        """Block until we're under the rate limit."""
        now = time.time()
        # Remove timestamps older than 60s
        self._timestamps = [t for t in self._timestamps if now - t < 60]

        if len(self._timestamps) >= self.max_per_minute:
            # Wait until the oldest timestamp expires
            sleep_time = 60 - (now - self._timestamps[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._timestamps.append(time.time())


# ============================================================
# ChatGPT Export Parser
# ============================================================

def _walk_chatgpt_tree(mapping: dict) -> list[dict]:
    """
    Reconstruct message order from ChatGPT's tree structure.

    Each node in `mapping` has:
      - "parent": parent node ID or None
      - "message": {"role": "user"|"assistant"|"system", "content": {...}}
      - "children": list of child node IDs

    We find the root (parent=None), then walk depth-first
    following the first child at each level (main conversation thread).
    """
    if not mapping:
        return []

    # Find root node (parent is None or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        return []

    # Walk the tree depth-first, following first child
    messages = []
    current_id = root_id

    while current_id:
        node = mapping.get(current_id)
        if not node:
            break

        msg = node.get("message")
        if msg and msg.get("content"):
            role = msg.get("author", {}).get("role", "")
            # Extract text content
            content_data = msg.get("content", {})
            if isinstance(content_data, dict):
                parts = content_data.get("parts", [])
                text = ""
                for part in parts:
                    if isinstance(part, str):
                        text += part
                    elif isinstance(part, dict) and "text" in part:
                        text += part["text"]
            elif isinstance(content_data, str):
                text = content_data
            else:
                text = ""

            text = text.strip()
            if text and role in ("user", "assistant"):
                messages.append({"role": role, "content": text})

        # Move to first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def parse_chatgpt_zip(zip_path: str) -> list[list[dict]]:
    """
    Parse ChatGPT export ZIP file.

    Returns list of conversations, each a list of {"role", "content"} dicts.
    """
    conversations = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find conversations.json
        json_files = [n for n in zf.namelist() if n.endswith("conversations.json")]
        if not json_files:
            raise ValueError("No conversations.json found in ZIP file")

        data = json.loads(zf.read(json_files[0]))

        if not isinstance(data, list):
            raise ValueError("conversations.json should contain a list")

        for conv in data:
            mapping = conv.get("mapping", {})
            messages = _walk_chatgpt_tree(mapping)
            if messages:
                conversations.append(messages)

    return conversations


# ============================================================
# Chunking Utilities
# ============================================================

def chunk_messages(messages: list[dict], chunk_size: int = 20) -> list[list[dict]]:
    """Split a conversation into chunks of `chunk_size` messages."""
    if not messages:
        return []
    if len(messages) <= chunk_size:
        return [messages]

    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text(text: str, chunk_chars: int = 4000) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.
    Each chunk is at most `chunk_chars` characters.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If single paragraph exceeds chunk_chars, split by lines
        if len(para) > chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # Split long paragraph by newlines
            lines = para.split("\n")
            for line in lines:
                if len(current) + len(line) + 1 > chunk_chars and current:
                    chunks.append(current.strip())
                    current = ""
                current += line + "\n"
            if current:
                chunks.append(current.strip())
                current = ""
            continue

        if len(current) + len(para) + 2 > chunk_chars and current:
            chunks.append(current.strip())
            current = ""

        current += para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ============================================================
# Importers
# ============================================================

def import_chatgpt(
    zip_path: str,
    add_fn: Callable,
    chunk_size: int = 20,
    on_progress: Optional[Callable] = None,
) -> ImportResult:
    """
    Import ChatGPT export ZIP into memory.

    Args:
        zip_path: Path to ChatGPT export ZIP
        add_fn: Callable that takes list[dict] messages → dict result
        chunk_size: Max messages per chunk (default 20)
        on_progress: Optional callback(current, total, title)
    """
    start = time.time()
    result = ImportResult()

    try:
        conversations = parse_chatgpt_zip(zip_path)
    except Exception as e:
        result.errors.append(f"Failed to parse ZIP: {e}")
        result.duration_seconds = time.time() - start
        return result

    result.conversations_found = len(conversations)
    total_chunks = sum(
        len(chunk_messages(conv, chunk_size)) for conv in conversations
    )

    chunk_idx = 0
    for i, conv in enumerate(conversations):
        chunks = chunk_messages(conv, chunk_size)
        for chunk in chunks:
            try:
                resp = add_fn(chunk)
                result.chunks_sent += 1
                chunk_idx += 1

                # Collect created entities if available
                for key in ("entities_created", "entities_updated"):
                    if isinstance(resp, dict) and key in resp:
                        result.entities_created.extend(resp[key])

                if on_progress:
                    on_progress(chunk_idx, total_chunks, f"conversation {i + 1}/{len(conversations)}")

            except Exception as e:
                result.errors.append(f"Conversation {i + 1}, chunk: {e}")

    result.entities_created = list(set(result.entities_created))
    result.duration_seconds = time.time() - start
    return result


def import_obsidian(
    vault_path: str,
    add_fn: Callable,
    chunk_chars: int = 4000,
    on_progress: Optional[Callable] = None,
) -> ImportResult:
    """
    Import Obsidian vault into memory.

    Args:
        vault_path: Path to Obsidian vault directory
        add_fn: Callable that takes list[dict] messages → dict result
        chunk_chars: Max characters per text chunk (default 4000)
        on_progress: Optional callback(current, total, title)
    """
    start = time.time()
    result = ImportResult()

    vault = Path(vault_path)
    if not vault.is_dir():
        result.errors.append(f"Not a directory: {vault_path}")
        result.duration_seconds = time.time() - start
        return result

    # Collect .md files, skip dotfiles and Obsidian internals
    md_files = []
    for f in sorted(vault.rglob("*.md")):
        rel = f.relative_to(vault)
        parts = rel.parts
        # Skip hidden dirs/files and .obsidian/, .trash/
        if any(p.startswith(".") for p in parts):
            continue
        if any(p in ("node_modules", "__pycache__") for p in parts):
            continue
        md_files.append(f)

    result.conversations_found = len(md_files)

    # Pre-count total chunks
    total_chunks = 0
    file_chunks = []
    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(content, chunk_chars)
            file_chunks.append((f, chunks))
            total_chunks += len(chunks) if chunks else 1
        except Exception:
            file_chunks.append((f, []))
            total_chunks += 1

    chunk_idx = 0
    for f, chunks in file_chunks:
        title = f.stem
        if not chunks:
            chunk_idx += 1
            continue

        for chunk in chunks:
            messages = [{"role": "user", "content": f"Note: {title}\n\n{chunk}"}]
            try:
                resp = add_fn(messages)
                result.chunks_sent += 1
                chunk_idx += 1

                for key in ("entities_created", "entities_updated"):
                    if isinstance(resp, dict) and key in resp:
                        result.entities_created.extend(resp[key])

                if on_progress:
                    on_progress(chunk_idx, total_chunks, title)

            except Exception as e:
                result.errors.append(f"{title}: {e}")
                chunk_idx += 1

    result.entities_created = list(set(result.entities_created))
    result.duration_seconds = time.time() - start
    return result


def import_files(
    paths: list[str],
    add_fn: Callable,
    chunk_chars: int = 4000,
    on_progress: Optional[Callable] = None,
) -> ImportResult:
    """
    Import plain text/markdown files into memory.

    Args:
        paths: List of file paths
        add_fn: Callable that takes list[dict] messages → dict result
        chunk_chars: Max characters per text chunk (default 4000)
        on_progress: Optional callback(current, total, title)
    """
    start = time.time()
    result = ImportResult()

    # Resolve paths
    resolved = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            # Import all .md and .txt from directory
            for ext in ("*.md", "*.txt"):
                resolved.extend(sorted(path.rglob(ext)))

    result.conversations_found = len(resolved)

    # Pre-count chunks
    total_chunks = 0
    file_chunks = []
    for f in resolved:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(content, chunk_chars)
            file_chunks.append((f, chunks))
            total_chunks += len(chunks) if chunks else 1
        except Exception:
            file_chunks.append((f, []))
            total_chunks += 1

    chunk_idx = 0
    for f, chunks in file_chunks:
        title = f.stem
        if not chunks:
            chunk_idx += 1
            continue

        for chunk in chunks:
            messages = [{"role": "user", "content": f"Note: {title}\n\n{chunk}"}]
            try:
                resp = add_fn(messages)
                result.chunks_sent += 1
                chunk_idx += 1

                for key in ("entities_created", "entities_updated"):
                    if isinstance(resp, dict) and key in resp:
                        result.entities_created.extend(resp[key])

                if on_progress:
                    on_progress(chunk_idx, total_chunks, title)

            except Exception as e:
                result.errors.append(f"{title}: {e}")
                chunk_idx += 1

    result.entities_created = list(set(result.entities_created))
    result.duration_seconds = time.time() - start
    return result
