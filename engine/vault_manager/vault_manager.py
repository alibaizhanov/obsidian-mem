"""
Vault Manager — создаёт и обновляет .md файлы в Obsidian vault.

Берёт ExtractionResult (entities + relations) и:
1. Создаёт .md файл для каждой новой entity
2. Обновляет существующие файлы новыми фактами
3. Добавляет [[wikilinks]] для связей
4. Поддерживает YAML frontmatter
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

from engine.extractor.conversation_extractor import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
)


class VaultManager:
    """Управляет Obsidian vault — создаёт/обновляет .md файлы"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Vault: {self.vault_path.absolute()}")

    def process_extraction(self, extraction: ExtractionResult) -> dict:
        """
        Главный метод — обрабатывает результат извлечения.
        Создаёт/обновляет файлы в vault.

        Returns:
            {"created": [...], "updated": [...]}
        """
        stats = {"created": [], "updated": []}

        # 1. Обрабатываем каждую entity
        for entity in extraction.entities:
            file_path = self._entity_file_path(entity.name)

            # Находим связи для этой entity
            entity_relations = [
                r for r in extraction.relations
                if r.from_entity == entity.name or r.to_entity == entity.name
            ]

            if file_path.exists():
                self._update_note(file_path, entity, entity_relations)
                stats["updated"].append(entity.name)
            else:
                self._create_note(file_path, entity, entity_relations)
                stats["created"].append(entity.name)

        # 2. Создаём файлы для entities упомянутых только в relations
        all_entity_names = {e.name for e in extraction.entities}
        for rel in extraction.relations:
            for name in (rel.from_entity, rel.to_entity):
                if name not in all_entity_names:
                    file_path = self._entity_file_path(name)
                    if not file_path.exists():
                        stub = ExtractedEntity(name=name, entity_type="concept", facts=[])
                        self._create_note(file_path, stub, [])
                        stats["created"].append(name)
                        all_entity_names.add(name)

        return stats

    def _create_note(self, file_path: Path, entity: ExtractedEntity,
                     relations: list[ExtractedRelation]):
        """Создаёт новый .md файл"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Frontmatter
        frontmatter = {
            "type": entity.entity_type,
            "created": now,
            "updated": now,
            "tags": [entity.entity_type],
        }

        # Контент
        lines = []
        lines.append(f"# {entity.name}\n")

        # Факты
        if entity.facts:
            lines.append("## Факты\n")
            for fact in entity.facts:
                # Превращаем упоминания других entities в [[links]]
                linked_fact = self._add_wikilinks(fact, entity.name)
                lines.append(f"- {linked_fact}")
            lines.append("")

        # Связи
        if relations:
            lines.append("## Связи\n")
            for rel in relations:
                other = rel.to_entity if rel.from_entity == entity.name else rel.from_entity
                direction = "→" if rel.from_entity == entity.name else "←"
                desc = f": {rel.description}" if rel.description else ""
                lines.append(f"- {direction} **{rel.relation_type}** [[{other}]]{desc}")
            lines.append("")

        # Записываем файл
        content = self._format_with_frontmatter(frontmatter, "\n".join(lines))
        file_path.write_text(content, encoding="utf-8")

    def _update_note(self, file_path: Path, entity: ExtractedEntity,
                     relations: list[ExtractedRelation]):
        """Обновляет существующий .md файл — добавляет новые факты"""
        content = file_path.read_text(encoding="utf-8")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Парсим frontmatter
        frontmatter, body = self._parse_frontmatter(content)
        frontmatter["updated"] = now

        # Получаем существующие факты
        existing_facts = self._extract_existing_facts(body)

        # Добавляем новые факты (только уникальные)
        new_facts = []
        for fact in entity.facts:
            # Проверяем что факт не дубликат (нечёткое сравнение)
            if not self._fact_exists(fact, existing_facts):
                new_facts.append(fact)

        if not new_facts and not relations:
            return  # Нечего обновлять

        # Добавляем новые факты в секцию "Факты"
        if new_facts:
            facts_section = "\n## Обновления\n\n"
            facts_section += f"*{now}*\n\n"
            for fact in new_facts:
                linked_fact = self._add_wikilinks(fact, entity.name)
                facts_section += f"- {linked_fact}\n"
            body = body.rstrip() + "\n" + facts_section

        # Добавляем новые связи
        new_relations = []
        existing_links = set(re.findall(r"\[\[([^\]]+)\]\]", body))
        for rel in relations:
            other = rel.to_entity if rel.from_entity == entity.name else rel.from_entity
            if other not in existing_links:
                new_relations.append(rel)

        if new_relations:
            rel_section = "\n### Новые связи\n\n"
            for rel in new_relations:
                other = rel.to_entity if rel.from_entity == entity.name else rel.from_entity
                direction = "→" if rel.from_entity == entity.name else "←"
                desc = f": {rel.description}" if rel.description else ""
                rel_section += f"- {direction} **{rel.relation_type}** [[{other}]]{desc}\n"
            body = body.rstrip() + "\n" + rel_section

        # Записываем обновлённый файл
        content = self._format_with_frontmatter(frontmatter, body)
        file_path.write_text(content, encoding="utf-8")

    def _add_wikilinks(self, text: str, current_entity: str) -> str:
        """
        Находит упоминания entities в тексте и оборачивает в [[wikilinks]].
        Ищет по существующим файлам в vault.
        """
        existing_notes = {
            p.stem for p in self.vault_path.glob("*.md")
        }

        for note_name in existing_notes:
            if note_name == current_entity:
                continue
            # Case-insensitive замена, но сохраняем оригинальный регистр
            pattern = re.compile(re.escape(note_name), re.IGNORECASE)
            # Не оборачиваем если уже в [[ ]]
            if f"[[{note_name}]]" not in text:
                text = pattern.sub(f"[[{note_name}]]", text, count=1)

        return text

    def _entity_file_path(self, entity_name: str) -> Path:
        """Путь к .md файлу для entity"""
        # Убираем символы недопустимые в именах файлов
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', entity_name)
        return self.vault_path / f"{safe_name}.md"

    def _format_with_frontmatter(self, frontmatter: dict, body: str) -> str:
        """Собирает файл: frontmatter + body"""
        fm_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False).strip()
        return f"---\n{fm_str}\n---\n\n{body.strip()}\n"

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Парсит frontmatter из файла"""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return {}, content
        try:
            fm = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            fm = {}
        body = content[match.end():]
        return fm, body

    def _extract_existing_facts(self, body: str) -> list[str]:
        """Извлекает существующие факты из body"""
        facts = []
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Убираем [[links]] для сравнения
                clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", line[2:])
                facts.append(clean.lower().strip())
        return facts

    def _fact_exists(self, new_fact: str, existing_facts: list[str]) -> bool:
        """Проверяет есть ли уже такой факт (нечёткое сравнение)"""
        new_clean = new_fact.lower().strip()
        for existing in existing_facts:
            # Простое сравнение: если >70% слов совпадают
            new_words = set(new_clean.split())
            existing_words = set(existing.split())
            if not new_words:
                continue
            overlap = len(new_words & existing_words) / len(new_words)
            if overlap > 0.7:
                return True
        return False

    def get_vault_stats(self) -> dict:
        """Статистика vault"""
        files = list(self.vault_path.glob("*.md"))
        types = {}
        for f in files:
            content = f.read_text(encoding="utf-8")
            fm, _ = self._parse_frontmatter(content)
            t = fm.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        return {"total_notes": len(files), "by_type": types}

    def list_notes(self) -> list[str]:
        """Список всех заметок"""
        return sorted([p.stem for p in self.vault_path.glob("*.md")])


if __name__ == "__main__":
    from engine.extractor.conversation_extractor import (
        ConversationExtractor, MockLLMClient
    )

    print("🧪 Тест Vault Manager\n")

    # 1. Извлекаем знания (mock)
    extractor = ConversationExtractor(MockLLMClient())
    conversation = [
        {"role": "user", "content": "Я работаю в Uzum Bank, backend разработчик. Делаю микросервисы на Spring Boot."},
        {"role": "assistant", "content": "Какие БД используете?"},
        {"role": "user", "content": "PostgreSQL 15. Проблема с connection pool в Проект Alpha."},
    ]
    extraction = extractor.extract(conversation)

    # 2. Записываем в vault
    vault = VaultManager("./test_vault_auto")
    stats = vault.process_extraction(extraction)

    print(f"\n📊 Результат:")
    print(f"   Создано: {stats['created']}")
    print(f"   Обновлено: {stats['updated']}")

    # 3. Показываем что создалось
    print(f"\n📁 Файлы в vault:")
    for note in vault.list_notes():
        print(f"   📄 {note}.md")

    print(f"\n📈 Статистика: {vault.get_vault_stats()}")

    # 4. Показываем содержимое одного файла
    user_file = vault.vault_path / "User.md"
    if user_file.exists():
        print(f"\n{'='*50}")
        print(f"📄 User.md:")
        print(f"{'='*50}")
        print(user_file.read_text())
