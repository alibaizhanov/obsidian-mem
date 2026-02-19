"""
Mengram × LangChain Integration

Drop-in replacement for LangChain's memory and retrieval components.
Uses Mengram's 3 memory types (semantic, episodic, procedural) as backend.

Quick Start:
    pip install mengram-ai langchain-core

Usage with RunnableWithMessageHistory (modern):
    from integrations.langchain import MengramChatMessageHistory

    history = MengramChatMessageHistory(api_key="om-...")
    chain_with_history = RunnableWithMessageHistory(chain, lambda sid: history)

Usage as Retriever (RAG replacement):
    from integrations.langchain import MengramRetriever

    retriever = MengramRetriever(api_key="om-...")
    docs = retriever.invoke("deployment issues")
    # → Documents from all 3 memory types

Usage with Cognitive Profile:
    from integrations.langchain import get_mengram_profile_prompt

    system = get_mengram_profile_prompt(api_key="om-...")
    chain = prompt | llm  # system prompt auto-personalized
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger("mengram.langchain")


def _check_langchain():
    try:
        import langchain_core
        return True
    except ImportError:
        raise ImportError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install langchain-core"
        )


def _get_client(api_key: str, url: str = "https://mengram.io"):
    from cloud.client import CloudMemory
    return CloudMemory(api_key=api_key, base_url=url)


# =====================================================
# 1. Chat Message History
# =====================================================

class MengramChatMessageHistory:
    """
    LangChain-compatible chat message history backed by Mengram.
    
    Every time messages are added, Mengram's LLM extraction runs in the 
    background — automatically extracting facts, events, and workflows 
    into semantic, episodic, and procedural memory.
    
    Usage:
        from integrations.langchain import MengramChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory

        def get_history(session_id: str):
            return MengramChatMessageHistory(
                api_key="om-...",
                session_id=session_id,
            )
        
        chain_with_history = RunnableWithMessageHistory(
            chain, get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    """

    def __init__(
        self,
        api_key: str,
        user_id: str = "default",
        url: str = "https://mengram.io",
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        _check_langchain()
        self.client = _get_client(api_key, url)
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_id = app_id
        self.run_id = run_id
        self._messages: list = []

    @property
    def messages(self):
        from langchain_core.messages import HumanMessage, AIMessage
        return list(self._messages)

    def add_message(self, message) -> None:
        self._messages.append(message)

    def add_messages(self, messages: Sequence) -> None:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        self._messages.extend(messages)

        # Convert to Mengram format and send for extraction
        mengram_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                mengram_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                mengram_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                continue
            else:
                mengram_messages.append({"role": "user", "content": str(msg.content)})

        if mengram_messages:
            try:
                self.client.add(
                    mengram_messages,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    app_id=self.app_id,
                    run_id=self.run_id,
                )
                logger.debug(f"Sent {len(mengram_messages)} messages to Mengram")
            except Exception as e:
                logger.warning(f"Failed to send to Mengram: {e}")

    def add_user_message(self, message: str) -> None:
        from langchain_core.messages import HumanMessage
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        from langchain_core.messages import AIMessage
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        self._messages.clear()


# =====================================================
# 2. Retriever — searches all 3 memory types
# =====================================================

class MengramRetriever:
    """
    LangChain-compatible retriever that searches across all 3 Mengram memory types.
    
    Returns Documents with metadata indicating memory_type (semantic/episodic/procedural).
    Drop-in replacement for any LangChain retriever — replaces RAG with memory.
    
    Usage:
        from integrations.langchain import MengramRetriever

        retriever = MengramRetriever(api_key="om-...")
        
        # Use in a chain
        from langchain_core.runnables import RunnablePassthrough
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        # Or use directly
        docs = retriever.invoke("deployment problems")
        for doc in docs:
            print(doc.metadata["memory_type"], doc.page_content)
    """

    def __init__(
        self,
        api_key: str,
        user_id: str = "default",
        url: str = "https://mengram.io",
        top_k: int = 5,
        memory_types: Optional[list[str]] = None,
    ):
        """
        Args:
            api_key: Mengram API key
            user_id: User to search memories for
            url: Mengram API URL
            top_k: Max results per memory type
            memory_types: Which types to search. Default: all 3.
                Options: ["semantic", "episodic", "procedural"]
        """
        _check_langchain()
        self.client = _get_client(api_key, url)
        self.user_id = user_id
        self.top_k = top_k
        self.memory_types = memory_types or ["semantic", "episodic", "procedural"]

    def invoke(self, query: str, **kwargs) -> list:
        """Search Mengram and return LangChain Documents."""
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str) -> list:
        from langchain_core.documents import Document

        docs = []

        try:
            results = self.client.search_all(query, limit=self.top_k)
        except Exception as e:
            logger.warning(f"Mengram search failed: {e}")
            return docs

        # Semantic results
        if "semantic" in self.memory_types:
            for r in results.get("semantic", []):
                facts = r.get("facts", [])
                knowledge = r.get("knowledge", [])
                
                content_parts = []
                if facts:
                    content_parts.append(f"{r.get('entity', 'Unknown')}: {'; '.join(facts)}")
                for k in knowledge:
                    content_parts.append(
                        f"[{k.get('type', '')}] {k.get('title', '')}: {k.get('content', '')}"
                    )
                
                if content_parts:
                    docs.append(Document(
                        page_content="\n".join(content_parts),
                        metadata={
                            "memory_type": "semantic",
                            "entity": r.get("entity", ""),
                            "entity_type": r.get("type", ""),
                            "score": r.get("score", 0),
                            "source": "mengram",
                        }
                    ))

        # Episodic results
        if "episodic" in self.memory_types:
            for ep in results.get("episodic", []):
                content = f"Event: {ep.get('summary', '')}"
                if ep.get("context"):
                    content += f"\nDetails: {ep['context']}"
                if ep.get("outcome"):
                    content += f"\nOutcome: {ep['outcome']}"

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "memory_type": "episodic",
                        "participants": ep.get("participants", []),
                        "emotional_valence": ep.get("emotional_valence", "neutral"),
                        "importance": ep.get("importance", 0.5),
                        "score": ep.get("score", 0),
                        "created_at": ep.get("created_at", ""),
                        "source": "mengram",
                    }
                ))

        # Procedural results
        if "procedural" in self.memory_types:
            for pr in results.get("procedural", []):
                steps_text = "\n".join(
                    f"  {s.get('step', i+1)}. {s.get('action', '')} — {s.get('detail', '')}"
                    for i, s in enumerate(pr.get("steps", []))
                )
                content = f"Procedure: {pr.get('name', '')}"
                if pr.get("trigger_condition"):
                    content += f"\nWhen: {pr['trigger_condition']}"
                if steps_text:
                    content += f"\nSteps:\n{steps_text}"

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "memory_type": "procedural",
                        "procedure_name": pr.get("name", ""),
                        "success_count": pr.get("success_count", 0),
                        "fail_count": pr.get("fail_count", 0),
                        "score": pr.get("score", 0),
                        "source": "mengram",
                    }
                ))

        docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        return docs


# =====================================================
# 3. Cognitive Profile — instant system prompt
# =====================================================

def get_mengram_profile_prompt(
    api_key: str,
    user_id: str = "default",
    url: str = "https://mengram.io",
    force: bool = False,
) -> str:
    """
    Get a Cognitive Profile as a system prompt string.
    
    One API call generates a ready-to-use system prompt from all 3 memory types.
    Cached for 1 hour on the server side.
    
    Usage:
        from integrations.langchain import get_mengram_profile_prompt
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = get_mengram_profile_prompt(api_key="om-...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\\n\\n{instructions}"),
            ("human", "{input}"),
        ])
        
        chain = prompt | llm
        chain.invoke({"instructions": "Be helpful", "input": "What should I work on?"})
    
    Returns:
        System prompt string, or empty string if no data.
    """
    client = _get_client(api_key, url)
    try:
        profile = client.get_profile(user_id, force=force)
        return profile.get("system_prompt", "")
    except Exception as e:
        logger.warning(f"Failed to get Mengram profile: {e}")
        return ""


def create_mengram_profile_prompt(
    api_key: str,
    user_id: str = "default",
    url: str = "https://mengram.io",
    additional_instructions: str = "",
):
    """
    Create a LangChain ChatPromptTemplate with Cognitive Profile baked in.
    
    Usage:
        from integrations.langchain import create_mengram_profile_prompt

        prompt = create_mengram_profile_prompt(
            api_key="om-...",
            additional_instructions="Be concise. Focus on actionable advice.",
        )
        
        chain = prompt | llm
        chain.invoke({"input": "What should I prioritize this week?"})
    """
    _check_langchain()
    from langchain_core.prompts import ChatPromptTemplate

    profile_text = get_mengram_profile_prompt(api_key, user_id, url)
    
    system_parts = []
    if profile_text:
        system_parts.append(profile_text)
    if additional_instructions:
        system_parts.append(additional_instructions)
    
    system_message = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{history}"),
        ("human", "{input}"),
    ])


# =====================================================
# 4. All-in-one: memory + retrieval + profile
# =====================================================

def create_mengram_chain(
    llm,
    api_key: str,
    user_id: str = "default",
    url: str = "https://mengram.io",
    additional_instructions: str = "",
):
    """
    Create a fully personalized LangChain chain with:
    - Cognitive Profile as system prompt
    - Mengram retriever for context (all 3 memory types)
    - Automatic memory formatting
    
    Usage:
        from integrations.langchain import create_mengram_chain
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini")
        chain = create_mengram_chain(llm, api_key="om-...")
        
        response = chain.invoke({"input": "How do I deploy?"})
        # → Uses Cognitive Profile + searches episodic/procedural memory
        #   for deployment-related events and workflows
    """
    _check_langchain()
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    profile_text = get_mengram_profile_prompt(api_key, user_id, url)
    system = profile_text or "You are a helpful assistant."
    if additional_instructions:
        system += f"\n\n{additional_instructions}"

    retriever = MengramRetriever(api_key=api_key, user_id=user_id, url=url, top_k=3)

    def format_context(docs):
        if not docs:
            return "No relevant memories found."
        parts = []
        for doc in docs:
            mtype = doc.metadata.get("memory_type", "unknown")
            parts.append(f"[{mtype}] {doc.page_content}")
        return "\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system + "\n\nRelevant memory context:\n{context}"),
        ("human", "{input}"),
    ])

    chain = (
        {
            "context": (lambda x: x["input"]) | retriever | RunnableLambda(format_context),
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
