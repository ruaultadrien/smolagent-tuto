"""Agents module."""

from smolagents import CodeAgent, GoogleSearchTool, Model, VisitWebpageTool

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
