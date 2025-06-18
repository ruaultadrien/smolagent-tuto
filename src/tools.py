"""Tools for the agent."""

import math
from collections.abc import Callable
from typing import ClassVar

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, tool

from src.documents import get_documents


# Tool to list the available occasions
@tool
def list_occasions() -> str:
    """List the available occasions."""
    return "casual, formal, superhero, custom"


# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggest a menu based on the occasion.

    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.

    """
    return {
        "casual": "Pizza, snacks, and drinks.",
        "formal": "3-course dinner with wine and dessert.",
        "superhero": "Buffet with high-energy and healthy food.",
    }.get(occasion, "Custom menu for the butler.")


# function that fetches the highest-rated catering services.
@tool
def catering_service_tool() -> str:
    """
    Tool that returns the highest-rated catering service in Gotham City.

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
    return max(services, key=services.get)


class SuperheroPartyThemeTool(Tool):
    """Super Hero Party Theme Generator - Tool for agent."""

    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs: ClassVar[dict] = {
        "category": {
            "type": "string",
            "description": """
            The type of superhero party
            (e.g., 'classic heroes','villain masquerade', 'futuristic Gotham').
            """,
        },
    }

    output_type = "string"

    def forward(self, category: str) -> str:
        """Generate a superhero-themed party theme based on the category."""
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
            """
            Themed party idea not found. Try 'classic heroes', 'villain masquerade',
            or 'futuristic Gotham'.
            """,
        )


def get_langchain_serpapi_tool() -> Callable:
    """Return a tool that uses the SerpAPI to search the web."""
    serpapi_search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
    serpapi_search_tool.name = "serpapi_search_tool"
    return serpapi_search_tool


def get_image_generation_tool() -> Callable:
    """Return a tool that generates an image based on a prompt."""
    return Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="Generate an image from a prompt",
    )


class PartyPlanningRetrieverTool(Tool):
    """Retrieve relevant party planning ideas for Alfred's party."""

    name = "party_planning_retriever"
    description = """
        Uses semantic search to retrieve relevant party planning ideas for
        Alfred's superhero-themed party at Wayne Manor.
        """

    inputs: ClassVar[dict] = {
        "query": {
            "type": "string",
            "description": """
                The query to perform. This should be a query related to
                party planning or superhero themes.
            """,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize the retriever tool."""
        super().__init__(**kwargs)
        docs = get_documents()
        self.retriever = BM25Retriever.from_documents(
            docs,
            k=5,  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        """Retrieve relevant party planning ideas for Alfred's party."""
        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {i!s} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ],
        )


@tool
def calculate_cargo_travel_time(
    origin_coords: tuple[float, float],
    destination_coords: tuple[float, float],
    cruising_speed_kmh: float | None = 750.0,  # Average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth.

    It great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point.
        destination_coords: Tuple of (latitude, longitude) for the destination.
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for
            typical cargo planes).

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.878, -87.629), (-33.868, 151.209))

    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    earth_radius_km = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = earth_radius_km * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)
