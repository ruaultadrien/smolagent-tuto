"""Define the Gradio interface for the agent."""

import os

import gradio as gr
from huggingface_hub import login
from smolagents import (
    CodeAgent,
    ToolCollection,
)

from mcp import StdioServerParameters
from src.agents import get_web_agent
from src.logger import logger, setup_langfuse
from src.mcp import process_mcp_tools
from src.models import get_model
from src.tools import calculate_cargo_travel_time


def call_agent(task: str) -> tuple[str, str]:
    """Get the agent and call it with the prompt."""
    setup_langfuse()

    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    server_parameters = StdioServerParameters(
        command="uvx",
        args=["--quiet", "pubmedmcp@0.1.3"],
        env={"UV_PYTHON": "3.11", **os.environ},
    )
    with ToolCollection.from_mcp(
        server_parameters,
        trust_remote_code=True,
    ) as tool_collection:
        mcp_tools = process_mcp_tools(tool_collection)
        model = get_model("mistral")
        agent = CodeAgent(
            tools=[calculate_cargo_travel_time, *mcp_tools],
            model=model,
            managed_agents=[get_web_agent(model)],
            add_base_tools=False,
            additional_authorized_imports=[
                "geopandas",
                "plotly",
                "plotly.graph_objects",
                "plotly.express",
                "plotly.express.colors",
                "plotly.express.colors.sequential",
                "plotly.express.graph_objects",
                "matplotlib",
                "matplotlib.pyplot",
                "shapely",
                "json",
                "pandas",
                "numpy",
            ],
            planning_interval=5,
            verbosity_level=2,
            max_steps=20,
        )

        logger.info(
            f"Agent's available tools: {list(agent.tools.keys())}",
        )

        prompt = f"""
            You're an expert analyst. You make comprehensive reports after visiting
            many websites. Don't hesitate to search for many queries at once in a for
            loop. For each data point that you find, visit the source url to confirm
            numbers.

            {task}
            """
        return agent.run(prompt)


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch(share=False)
