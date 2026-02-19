#!/usr/bin/env python3
"""
Customer Support Agent — CrewAI + Mengram

A support agent that remembers customer history, preferences, and past tickets.
Run it twice to see the difference between a "blank slate" and a "returning customer".
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew
from integrations.crewai import create_mengram_tools

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
        print(f"{RED}Error: OPENAI_API_KEY not set (required by CrewAI).{RESET}")
        sys.exit(1)

    tools = create_mengram_tools(api_key=api_key)

    agent = Agent(
        role="Customer Support Specialist",
        goal="Help customers by remembering their history and providing personalized support",
        backstory=(
            "You are an experienced customer support specialist with access to a "
            "memory system. ALWAYS search memory first to check for customer history "
            "before responding. When you learn something new about the customer "
            "(preferences, issues, personal details), save it to memory. Use the "
            "cognitive profile to understand the customer's communication style."
        ),
        tools=tools,
        verbose=True,
    )

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  Customer Support Agent — CrewAI + Mengram")
    print(f"{'=' * 60}{RESET}")
    print(f"\n  {DIM}The agent has 5 Mengram tools: search, remember, profile,")
    print(f"  save workflow, and workflow feedback.{RESET}")
    print(f"  {DIM}Watch CrewAI's verbose output to see the agent's reasoning.{RESET}")
    print(f"  {DIM}Type 'quit' to exit.{RESET}\n")

    while True:
        try:
            message = input(f"{GREEN}Customer:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not message or message.lower() in ("quit", "exit", "q"):
            break

        task = Task(
            description=(
                f'A customer says: "{message}"\n\n'
                "Steps:\n"
                "1. Search memory for any relevant history about this customer\n"
                "2. Check the customer's cognitive profile for communication preferences\n"
                "3. Respond helpfully using what you know about them\n"
                "4. Save any new information from this interaction to memory"
            ),
            expected_output="A helpful, personalized response to the customer",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n{BOLD}{CYAN}Agent:{RESET} {result}\n")

    print(f"\n{DIM}Session ended. Run again — the agent will remember this customer.{RESET}\n")


if __name__ == "__main__":
    main()
