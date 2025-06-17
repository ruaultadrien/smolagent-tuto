"""Define the Gradio interface for the agent."""

import os

import gradio as gr
from huggingface_hub import login
from smolagents import CodeAgent, LiteLLMModel, ToolCollection

from mcp import StdioServerParameters
from src.agent import setup_langfuse
from src.logger import logger
from src.mcp import process_mcp_tools
from src.tools import get_party_planner_retriever_tool

# from src.tools import (
#     SuperheroPartyThemeTool,
#     catering_service_tool,
#     get_image_generation_tool,
#     get_langchain_serpapi_tool,
#     list_occasions,
#     suggest_menu,
# )  # noqa: ERA001, RUF100


def call_agent(prompt: str) -> tuple[str, str]:
    """Get the agent and call it with the prompt."""
    setup_langfuse()

    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    model = LiteLLMModel(
        model_id="mistral/mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
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

        party_planning_retriever_tool = get_party_planner_retriever_tool()
        tools = [
            party_planning_retriever_tool,
            # suggest_menu,
            # list_occasions,
            # catering_service_tool,
            # SuperheroPartyThemeTool(),  # noqa: ERA001
            # get_image_generation_tool(),  # noqa: ERA001
            # get_langchain_serpapi_tool(),  # noqa: ERA001
            *mcp_tools,
        ]
        agent = CodeAgent(tools=tools, model=model, add_base_tools=True)

        logger.info(
            f"Agent's available tools: {list(agent.tools.keys())}",
        )

        return agent.run(prompt)


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch(share=False)
