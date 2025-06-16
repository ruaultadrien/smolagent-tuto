"""Define the Gradio interface for the agent."""

import logging
import os

import gradio as gr
from huggingface_hub import login
from mcp import StdioServerParameters
from smolagents import DuckDuckGoSearchTool, LiteLLMModel, ToolCollection

from src.agent import get_agent, setup_langfuse
from src.tools import (
    SuperheroPartyThemeTool,
    catering_service_tool,
    get_image_generation_tool,
    get_langchain_tool,
    list_occasions,
    suggest_menu,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def call_agent(prompt: str) -> tuple[str, str]:
    """Get the agent and call it with the prompt."""
    setup_langfuse()

    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    server_parameters = StdioServerParameters(
        command="uvx",
        args=["--quiet", "pubmedmcp@0.1.3"],
        env={"UV_PYTHON": "3.11", **os.environ},
    )
    with ToolCollection.from_mcp(
        server_parameters, trust_remote_code=True
    ) as tool_collection:
        mcp_tool_collection: ToolCollection = tool_collection
        mcp_tools = mcp_tool_collection.tools
        for tool in mcp_tools:
            tool.name = f"mcp_{tool.name}"

        tools = [
            DuckDuckGoSearchTool(),
            suggest_menu,
            list_occasions,
            catering_service_tool,
            SuperheroPartyThemeTool(),
            get_image_generation_tool(),
            get_langchain_tool(),
            *tool_collection.tools,
        ]
        logging.info(f"Available tools: {tools}")
        agent = get_agent(model=model, tools=tools, code_agent=True)

        res = agent.run(prompt)

    return res


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch(share=False)
