"""Agents module."""

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    Model,
    VisitWebpageTool,
)

from src.browser_tools import close_popups, go_back, save_screenshot, search_item_ctrl_f
from src.tools import calculate_cargo_travel_time


def get_web_agent(model: Model) -> CodeAgent:
    """Return a Web Agent."""
    return CodeAgent(
        model=model,
        tools=[
            GoogleSearchTool("serper"),
            VisitWebpageTool(),
            calculate_cargo_travel_time,
        ],
        name="web_agent",
        description="A web agent that can search the web and visit webpages.",
        max_steps=10,
        verbosity_level=0,
        add_base_tools=False,
    )


def get_browser_agent(model: Model) -> CodeAgent:
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )
