"""
Auto-Memory Middleware — автоматическая память для любого LLM.

Как Mem0 proxy: оборачивает любой вызов LLM.
- ПЕРЕД ответом: recall → добавляет контекст из vault
- ПОСЛЕ ответа: remember → извлекает и сохраняет новые знания

Использование:
    from mengram import Memory
    from mengram_middleware import AutoMemory

    m = Memory(vault_path="./vault", llm_provider="anthropic", api_key="...")
    auto = AutoMemory(memory=m, user_id="ali")

    # Просто общайся — память работает автоматически
    response = auto.chat("У нас проблема с Kafka consumer lag")
    # → Автоматически: recall контекст → LLM ответ → remember новые знания

    # Или используй с OpenAI-совместимым API
    response = auto.chat_with_history([
        {"role": "user", "content": "Помоги с PostgreSQL"},
    ])
"""

from typing import Optional

from mengram import Memory
from engine.extractor.llm_client import LLMClient


class AutoMemory:
    """
    Автоматическая память для LLM.

    Оборачивает каждый вызов:
    1. recall → ищет контекст в vault
    2. Добавляет контекст в system prompt
    3. Вызывает LLM
    4. remember → извлекает знания из разговора
    5. Возвращает ответ
    """

    def __init__(
        self,
        memory: Memory,
        user_id: str = "default",
        auto_remember: bool = True,
        auto_recall: bool = True,
        system_prompt: str = "",
    ):
        self.memory = memory
        self.user_id = user_id
        self.auto_remember = auto_remember
        self.auto_recall = auto_recall
        self.base_system_prompt = system_prompt or (
            "Ты полезный AI-ассистент. Используй контекст из памяти "
            "чтобы давать персонализированные ответы."
        )
        self.conversation_history: list[dict] = []

    def chat(self, message: str) -> str:
        """
        Отправить сообщение с автоматической памятью.

        Args:
            message: Сообщение пользователя

        Returns:
            Ответ LLM (с учётом контекста из vault)
        """
        # Step 1: Recall — ищем контекст
        context = ""
        if self.auto_recall:
            brain = self.memory._get_brain(self.user_id)
            context = brain.recall(message)
            if context and context != f"Ничего не найдено по запросу: '{message}'":
                print(f"🔍 Recall: найден контекст ({len(context)} chars)")
            else:
                context = ""

        # Step 2: Собираем system prompt с контекстом
        system = self.base_system_prompt
        if context:
            system += f"\n\n## Контекст из памяти пользователя:\n{context}"

        # Step 3: Вызываем LLM
        self.conversation_history.append({"role": "user", "content": message})

        # Форматируем историю в один промпт
        conv_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.conversation_history[-10:]  # Последние 10 сообщений
        )

        response = self.memory.llm.complete(
            prompt=conv_text,
            system=system,
        )

        self.conversation_history.append({"role": "assistant", "content": response})

        # Step 4: Remember — извлекаем знания
        if self.auto_remember:
            try:
                # Берём последние 2 сообщения (user + assistant)
                recent = self.conversation_history[-2:]
                result = self.memory.add(recent, user_id=self.user_id)
                created = result.get("entities_created", [])
                updated = result.get("entities_updated", [])
                if created or updated:
                    print(f"💾 Remember: +{len(created)} created, ~{len(updated)} updated")
            except Exception as e:
                print(f"⚠️ Remember failed: {e}")

        return response

    def chat_with_history(self, messages: list[dict]) -> str:
        """
        Вызов с полной историей сообщений (OpenAI-style).

        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Ответ LLM
        """
        if not messages:
            return ""

        # Берём последнее сообщение как запрос
        last_message = messages[-1]["content"]
        self.conversation_history = messages[:-1]

        return self.chat(last_message)

    def reset(self):
        """Сбросить историю разговора (память в vault остаётся)"""
        self.conversation_history = []


# ==========================================
# Обёртка для OpenAI-совместимого API
# ==========================================

class MemoryOpenAIWrapper:
    """
    Drop-in замена для OpenAI client с автоматической памятью.

    Использование:
        from openai import OpenAI
        from mengram_middleware import MemoryOpenAIWrapper

        client = MemoryOpenAIWrapper(
            openai_client=OpenAI(),
            memory=Memory(vault_path="./vault", ...),
            user_id="ali",
        )

        # Используй как обычный OpenAI client
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Помоги с проектом"}],
        )
    """

    def __init__(self, openai_client, memory: Memory, user_id: str = "default"):
        self._client = openai_client
        self.memory = memory
        self.user_id = user_id
        self.chat = self._ChatCompletions(self)

    class _ChatCompletions:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.completions = self

        def create(self, model: str, messages: list[dict], **kwargs):
            # 1. Recall
            last_msg = messages[-1]["content"] if messages else ""
            brain = self.wrapper.memory._get_brain(self.wrapper.user_id)
            context = brain.recall(last_msg)

            # 2. Inject context into system message
            enhanced_messages = list(messages)
            if context and "Ничего не найдено" not in context:
                system_msg = {
                    "role": "system",
                    "content": f"User memory context:\n{context}",
                }
                enhanced_messages.insert(0, system_msg)

            # 3. Call original OpenAI
            response = self.wrapper._client.chat.completions.create(
                model=model,
                messages=enhanced_messages,
                **kwargs,
            )

            # 4. Remember
            try:
                full_conv = messages + [
                    {"role": "assistant", "content": response.choices[0].message.content}
                ]
                self.wrapper.memory.add(full_conv[-4:], user_id=self.wrapper.user_id)
            except Exception:
                pass

            return response


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Auto-Memory Middleware — Demo")
    print("=" * 60)

    # Mock для теста
    m = Memory(vault_path="./demo_auto_vault", llm_provider="mock")
    auto = AutoMemory(memory=m, user_id="ali")

    print("\n💬 Chat 1:")
    resp = auto.chat("Я работаю в Uzum Bank, backend на Spring Boot")
    print(f"   Response: {resp[:100]}...")

    print(f"\n📁 Vault: {m.get_all(user_id='ali')}")

    print("\n💬 Chat 2:")
    resp = auto.chat("У нас проблема с PostgreSQL connection pool")
    print(f"   Response: {resp[:100]}...")

    print(f"\n📁 Vault now: {m.get_all(user_id='ali')}")
