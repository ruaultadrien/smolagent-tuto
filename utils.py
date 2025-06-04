"""Sandbox for using the smolagents library."""

from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, Model


# Tool to list the available occasions
@tool
def list_occasions() -> str:
    """
    Lists the available occasions.
    """
    return "casual, formal, superhero, custom"


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


def get_agent(model: Model) -> CodeAgent:
    """Returns the tuto agent."""
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), suggest_menu, list_occasions],
        model=model,
        additional_authorized_imports=["datetime"],
    )
    return agent
