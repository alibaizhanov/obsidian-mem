"""Тесты для Markdown Parser"""

import pytest
from engine.parser.markdown_parser import (
    parse_frontmatter,
    extract_wikilinks,
    extract_tags,
    extract_sections,
    create_chunks,
    parse_note,
    parse_vault,
)


class TestFrontmatter:
    def test_parses_yaml(self):
        content = "---\ntags: [a, b]\nstatus: active\n---\n# Hello"
        meta, body = parse_frontmatter(content)
        assert meta["tags"] == ["a", "b"]
        assert meta["status"] == "active"
        assert body.strip() == "# Hello"

    def test_no_frontmatter(self):
        content = "# Just a note\nSome text"
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: [[\n---\nBody"
        meta, body = parse_frontmatter(content)
        assert meta == {}


class TestWikilinks:
    def test_simple_link(self):
        links = extract_wikilinks("Check out [[PostgreSQL]] for details")
        assert len(links) == 1
        assert links[0].target == "PostgreSQL"

    def test_alias_link(self):
        links = extract_wikilinks("See [[Проект Alpha|main project]]")
        assert links[0].target == "Проект Alpha"
        assert links[0].alias == "main project"

    def test_multiple_links(self):
        text = "Uses [[Kafka]] and [[Redis]] together"
        links = extract_wikilinks(text)
        assert len(links) == 2
        targets = {l.target for l in links}
        assert targets == {"Kafka", "Redis"}

    def test_no_links(self):
        links = extract_wikilinks("No links here")
        assert links == []


class TestTags:
    def test_frontmatter_tags(self):
        tags = extract_tags("Some text", {"tags": ["project", "backend"]})
        assert "project" in tags
        assert "backend" in tags

    def test_inline_tags(self):
        tags = extract_tags("This is #important and #urgent", {})
        assert "important" in tags
        assert "urgent" in tags

    def test_combined(self):
        tags = extract_tags("#inline tag", {"tags": ["frontmatter"]})
        assert "inline" in tags
        assert "frontmatter" in tags


class TestSections:
    def test_basic_sections(self):
        content = "# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"
        sections = extract_sections(content)
        assert len(sections) == 3
        assert sections[0].title == "Title"
        assert sections[1].title == "Section A"
        assert sections[2].title == "Section B"

    def test_no_headings(self):
        content = "Just plain text here"
        sections = extract_sections(content)
        assert len(sections) == 1
        assert sections[0].title == "(root)"


class TestParseVault:
    def test_parse_test_vault(self):
        notes = parse_vault("test_vault")
        assert len(notes) == 7

        # Проверяем что все имена файлов стали title
        titles = {n.title for n in notes}
        assert "Проект Alpha" in titles
        assert "PostgreSQL" in titles
        assert "Ali" in titles

    def test_graph_connections(self):
        notes = parse_vault("test_vault")
        alpha = next(n for n in notes if n.title == "Проект Alpha")

        link_targets = {l.target for l in alpha.wikilinks}
        assert "PostgreSQL" in link_targets
        assert "Kafka" in link_targets
        assert "Ali" in link_targets
        assert "Doston" in link_targets

    def test_frontmatter_extracted(self):
        notes = parse_vault("test_vault")
        alpha = next(n for n in notes if n.title == "Проект Alpha")

        assert alpha.frontmatter["status"] == "active"
        assert "Ali" in alpha.frontmatter["team"]
        assert alpha.frontmatter["priority"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
