"""Tests for importer.py — ChatGPT, Obsidian, and text file importers."""

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from importer import (
    ImportResult,
    RateLimiter,
    _walk_chatgpt_tree,
    chunk_messages,
    chunk_text,
    import_chatgpt,
    import_files,
    import_obsidian,
    parse_chatgpt_zip,
)


# ============================================================
# Helpers
# ============================================================

def make_mock_add_fn():
    """Create a mock add_fn that records calls."""
    calls = []

    def add_fn(messages):
        calls.append(messages)
        return {"entities_created": ["TestEntity"], "entities_updated": []}

    return add_fn, calls


def make_chatgpt_zip(conversations: list[dict], tmpdir: str) -> str:
    """Create a minimal ChatGPT export ZIP."""
    zip_path = os.path.join(tmpdir, "chatgpt-export.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    return zip_path


def make_chatgpt_conversation(messages: list[tuple[str, str]], title: str = "Test") -> dict:
    """
    Build a ChatGPT conversation dict with mapping tree.

    messages: list of (role, content) tuples
    """
    mapping = {}
    # Root node (system)
    root_id = "root-0"
    mapping[root_id] = {
        "id": root_id,
        "parent": None,
        "children": [],
        "message": None,
    }

    prev_id = root_id
    for i, (role, content) in enumerate(messages):
        node_id = f"msg-{i}"
        mapping[node_id] = {
            "id": node_id,
            "parent": prev_id,
            "children": [],
            "message": {
                "author": {"role": role},
                "content": {"parts": [content]},
            },
        }
        mapping[prev_id]["children"].append(node_id)
        prev_id = node_id

    return {"title": title, "mapping": mapping}


# ============================================================
# Tests: _walk_chatgpt_tree
# ============================================================

class TestWalkChatgptTree:
    def test_empty_mapping(self):
        assert _walk_chatgpt_tree({}) == []

    def test_single_message(self):
        conv = make_chatgpt_conversation([("user", "Hello")])
        messages = _walk_chatgpt_tree(conv["mapping"])
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_conversation_pair(self):
        conv = make_chatgpt_conversation([
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language."),
        ])
        messages = _walk_chatgpt_tree(conv["mapping"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_skips_system_messages(self):
        conv = make_chatgpt_conversation([
            ("system", "You are a helpful assistant."),
            ("user", "Hi"),
            ("assistant", "Hello!"),
        ])
        messages = _walk_chatgpt_tree(conv["mapping"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_skips_empty_content(self):
        mapping = {}
        root_id = "root"
        mapping[root_id] = {
            "id": root_id, "parent": None, "children": ["msg-0"],
            "message": None,
        }
        mapping["msg-0"] = {
            "id": "msg-0", "parent": root_id, "children": ["msg-1"],
            "message": {"author": {"role": "user"}, "content": {"parts": [""]}},
        }
        mapping["msg-1"] = {
            "id": "msg-1", "parent": "msg-0", "children": [],
            "message": {"author": {"role": "assistant"}, "content": {"parts": ["Real answer"]}},
        }
        messages = _walk_chatgpt_tree(mapping)
        assert len(messages) == 1
        assert messages[0]["content"] == "Real answer"


# ============================================================
# Tests: parse_chatgpt_zip
# ============================================================

class TestParseChatgptZip:
    def test_valid_zip(self, tmp_path):
        conv = make_chatgpt_conversation([
            ("user", "Hello"),
            ("assistant", "Hi!"),
        ])
        zip_path = make_chatgpt_zip([conv], str(tmp_path))
        conversations = parse_chatgpt_zip(zip_path)
        assert len(conversations) == 1
        assert len(conversations[0]) == 2

    def test_multiple_conversations(self, tmp_path):
        convs = [
            make_chatgpt_conversation([("user", "First")]),
            make_chatgpt_conversation([("user", "Second"), ("assistant", "Got it")]),
        ]
        zip_path = make_chatgpt_zip(convs, str(tmp_path))
        conversations = parse_chatgpt_zip(zip_path)
        assert len(conversations) == 2

    def test_missing_conversations_json(self, tmp_path):
        zip_path = os.path.join(str(tmp_path), "bad.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("other.txt", "nothing")
        with pytest.raises(ValueError, match="No conversations.json"):
            parse_chatgpt_zip(zip_path)

    def test_empty_conversations(self, tmp_path):
        zip_path = make_chatgpt_zip([], str(tmp_path))
        conversations = parse_chatgpt_zip(zip_path)
        assert conversations == []


# ============================================================
# Tests: chunk_messages
# ============================================================

class TestChunkMessages:
    def test_empty(self):
        assert chunk_messages([]) == []

    def test_under_limit(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        chunks = chunk_messages(msgs, chunk_size=20)
        assert len(chunks) == 1
        assert chunks[0] == msgs

    def test_exact_limit(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        chunks = chunk_messages(msgs, chunk_size=20)
        assert len(chunks) == 1

    def test_over_limit(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
        chunks = chunk_messages(msgs, chunk_size=10)
        assert len(chunks) == 3
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5


# ============================================================
# Tests: chunk_text
# ============================================================

class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        chunks = chunk_text("Hello world", chunk_chars=100)
        assert chunks == ["Hello world"]

    def test_paragraph_splitting(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_chars=25)
        assert len(chunks) >= 2
        # All content is preserved
        combined = " ".join(chunks)
        assert "Para one" in combined
        assert "Para three" in combined

    def test_long_single_paragraph(self):
        text = "x" * 100
        chunks = chunk_text(text, chunk_chars=30)
        assert len(chunks) >= 1
        # All content preserved
        assert sum(len(c) for c in chunks) >= 100


# ============================================================
# Tests: import_chatgpt
# ============================================================

class TestImportChatgpt:
    def test_basic_import(self, tmp_path):
        conv = make_chatgpt_conversation([
            ("user", "I work at Google on Kubernetes"),
            ("assistant", "Noted! You work at Google on K8s."),
        ])
        zip_path = make_chatgpt_zip([conv], str(tmp_path))

        add_fn, calls = make_mock_add_fn()
        result = import_chatgpt(zip_path, add_fn)

        assert result.conversations_found == 1
        assert result.chunks_sent == 1
        assert len(calls) == 1
        assert len(result.errors) == 0

    def test_chunked_import(self, tmp_path):
        # 30 messages → should become 2 chunks with chunk_size=20
        messages = []
        for i in range(15):
            messages.append(("user", f"Question {i}"))
            messages.append(("assistant", f"Answer {i}"))

        conv = make_chatgpt_conversation(messages)
        zip_path = make_chatgpt_zip([conv], str(tmp_path))

        add_fn, calls = make_mock_add_fn()
        result = import_chatgpt(zip_path, add_fn, chunk_size=20)

        assert result.conversations_found == 1
        assert result.chunks_sent == 2
        assert len(calls) == 2

    def test_progress_callback(self, tmp_path):
        conv = make_chatgpt_conversation([("user", "Hi"), ("assistant", "Hello")])
        zip_path = make_chatgpt_zip([conv], str(tmp_path))

        add_fn, _ = make_mock_add_fn()
        progress_calls = []

        def on_progress(current, total, title):
            progress_calls.append((current, total, title))

        import_chatgpt(zip_path, add_fn, on_progress=on_progress)
        assert len(progress_calls) == 1

    def test_bad_zip(self, tmp_path):
        bad_path = os.path.join(str(tmp_path), "bad.txt")
        with open(bad_path, "w") as f:
            f.write("not a zip")

        add_fn, calls = make_mock_add_fn()
        result = import_chatgpt(bad_path, add_fn)

        assert result.conversations_found == 0
        assert len(result.errors) == 1
        assert len(calls) == 0


# ============================================================
# Tests: import_obsidian
# ============================================================

class TestImportObsidian:
    def test_basic_import(self, tmp_path):
        # Create fake vault
        (tmp_path / "Note1.md").write_text("# My Note\n\nSome content here.")
        (tmp_path / "Note2.md").write_text("# Another\n\nMore content.")

        add_fn, calls = make_mock_add_fn()
        result = import_obsidian(str(tmp_path), add_fn)

        assert result.conversations_found == 2
        assert result.chunks_sent == 2
        assert len(calls) == 2
        # Check messages format
        assert calls[0][0]["role"] == "user"
        assert "Note:" in calls[0][0]["content"]

    def test_skips_dotfiles(self, tmp_path):
        (tmp_path / "visible.md").write_text("Content")
        dot_dir = tmp_path / ".obsidian"
        dot_dir.mkdir()
        (dot_dir / "config.md").write_text("Config")

        add_fn, calls = make_mock_add_fn()
        result = import_obsidian(str(tmp_path), add_fn)

        assert result.conversations_found == 1

    def test_not_a_directory(self, tmp_path):
        add_fn, calls = make_mock_add_fn()
        result = import_obsidian("/nonexistent/path", add_fn)

        assert len(result.errors) == 1
        assert len(calls) == 0


# ============================================================
# Tests: import_files
# ============================================================

class TestImportFiles:
    def test_basic_import(self, tmp_path):
        f1 = tmp_path / "notes.txt"
        f1.write_text("Some important notes here.")

        add_fn, calls = make_mock_add_fn()
        result = import_files([str(f1)], add_fn)

        assert result.conversations_found == 1
        assert result.chunks_sent == 1

    def test_directory_input(self, tmp_path):
        (tmp_path / "a.md").write_text("Content A")
        (tmp_path / "b.txt").write_text("Content B")
        (tmp_path / "c.py").write_text("# not imported")

        add_fn, calls = make_mock_add_fn()
        result = import_files([str(tmp_path)], add_fn)

        # Only .md and .txt files
        assert result.conversations_found == 2
        assert result.chunks_sent == 2


# ============================================================
# Tests: RateLimiter
# ============================================================

class TestRateLimiter:
    def test_under_limit(self):
        limiter = RateLimiter(max_per_minute=100)
        # Should not block for first few calls
        import time
        start = time.time()
        for _ in range(5):
            limiter.wait_if_needed()
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should be near-instant


# ============================================================
# Tests: ImportResult
# ============================================================

class TestImportResult:
    def test_defaults(self):
        r = ImportResult()
        assert r.conversations_found == 0
        assert r.chunks_sent == 0
        assert r.entities_created == []
        assert r.errors == []
        assert r.duration_seconds == 0.0
