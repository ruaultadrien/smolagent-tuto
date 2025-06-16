"""Tools for the agent."""

from typing import Callable

from langchain_community.agent_toolkits.load_tools import load_tools
from smolagents import Tool, tool


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


# function that fetches the highest-rated catering services.
@tool
def catering_service_tool() -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service


class SuperheroPartyThemeTool(Tool):
    """Super Hero Party Theme Generator - Tool for agent."""

    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": """
            The type of superhero party
            (e.g., 'classic heroes','villain masquerade', 'futuristic Gotham').
            """,
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": """
            Justice League Gala: Guests come dressed as their favorite DC heroes
            with themed cocktails like 'The Kryptonite Punch'.
            """,
            "villain masquerade": """
            Gotham Rogues' Ball: A mysterious masquerade
            where guests dress as classic Batman villains.
            """,
            "futuristic Gotham": """
            Neo-Gotham Night: A cyberpunk-style party inspired
            by Batman Beyond, with neon decorations and futuristic gadgets.
            """,
        }

        return themes.get(
            category.lower(),
            "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.",
        )


# def get_image_generation_tool() -> Callable:
#     return load_tool("m-ric/text-to-image", trust_remote_code=True)


def get_langchain_tool() -> Callable:
    serpapi_search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
    serpapi_search_tool.name = "serpapi_search_tool"
    return serpapi_search_tool


def get_image_generation_tool() -> Callable:
    image_generation_tool = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="Generate an image from a prompt",
    )
    return image_generation_tool
