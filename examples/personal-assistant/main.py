#!/usr/bin/env python3
"""
Personal Assistant — LangChain + Mengram

A personal assistant that knows you — preferences, relationships, habits.
It retrieves context from all 3 memory types and saves every conversation
so it gets smarter over time.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from integrations.langchain import (
    MengramChatMessageHistory,
    MengramRetriever,
    get_mengram_profile_prompt,
)

# Terminal colors
CYAN, GREEN, YELLOW, RED = "\033[96m", "\033[92m", "\033[93m", "\033[91m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def main():
    api_key = os.environ.get("MENGRAM_API_KEY")
    if not api_key:
        print(f"{RED}Error: MENGRAM_API_KEY not set.{RESET}")
        print(f"  Get your key at https://mengram.io/dashboard")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{RED}Error: OPENAI_API_KEY not set.{RESET}")
        sys.exit(1)

    # Initialize LangChain components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    history = MengramChatMessageHistory(api_key=api_key)
    retriever = MengramRetriever(api_key=api_key, top_k=5)

    # Load cognitive profile (AI-generated summary of what Mengram knows about you)
    print(f"\n{DIM}Loading cognitive profile...{RESET}")
    profile = get_mengram_profile_prompt(api_key=api_key)
    system_prompt = profile if profile else "You are a helpful personal assistant."

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  Personal Assistant — LangChain + Mengram")
    print(f"{'=' * 60}{RESET}")

    if profile:
        print(f"\n  {GREEN}Cognitive profile loaded — the assistant knows you.{RESET}")
    else:
        print(f"\n  {YELLOW}No profile yet. Chat to build one!{RESET}")

    print(f"  {DIM}Every message is saved to Mengram and enriches your profile.{RESET}")
    print(f"  {DIM}Type 'quit' to exit.{RESET}\n")

    while True:
        try:
            message = input(f"{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not message or message.lower() in ("quit", "exit", "q"):
            break

        # Retrieve relevant memories (semantic + episodic + procedural)
        docs = retriever.invoke(message)

        context = ""
        if docs:
            context = "\n\nRelevant memories:\n"
            for doc in docs:
                mem_type = doc.metadata.get("memory_type", "unknown")
                context += f"[{mem_type}] {doc.page_content}\n"

        # Generate response with profile + memory context
        response = llm.invoke([
            SystemMessage(content=system_prompt + context),
            HumanMessage(content=message),
        ])

        print(f"\n{BOLD}{CYAN}Assistant:{RESET} {response.content}\n")

        # Save conversation to Mengram (auto-extracts entities, facts, relationships)
        history.add_user_message(message)
        history.add_ai_message(response.content)

    print(f"\n{DIM}Session saved. Next time, the assistant will remember everything.{RESET}\n")


if __name__ == "__main__":
    main()
