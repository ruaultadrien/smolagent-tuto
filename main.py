"""Sandbox for using the smolagents library."""

import os

from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool


# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


def get_agent() -> CodeAgent:
    """Returns the tuto agent."""
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), suggest_menu],
        model=InferenceClientModel(),
        additional_authorized_imports=["datetime"],
    )
    return agent


def main():
    print("Enter application")
    # Assuming HUGGINGFACE_TOKEN is already set in the environment
    login(token=os.environ["HF_TOKEN"])

    agent = get_agent()

    # Pushing to hub
    agent.push_to_hub("ruaultadrienperso/AlfredAgent")

    # Pulling from hub
    agent = agent.pull_from_hub("ruaultadrienperso/AlfredAgent")

    prep_time_res = agent.run("""
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """)
    print("Party preparation time:", prep_time_res)


if __name__ == "__main__":
    main()
