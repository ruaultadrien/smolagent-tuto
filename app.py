"""Define the Gradio interface for the agent."""

import os

import gradio as gr
import smolagents
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


def call_agent(task: str) -> list[gr.ChatMessage]:
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
            stream_outputs=True,
        )

        logger.info(
            f"Agent's available tools: {list(agent.tools.keys())}",
        )

        messages = []
        for step in agent.run(task, stream=True):
            if isinstance(step, (smolagents.ActionStep, smolagents.PlanningStep)):
                memory_step: smolagents.ActionStep | smolagents.PlanningStep = step
                for message in memory_step.to_messages():
                    role = "user" if message["role"] == "user" else "assistant"
                    for content in message["content"]:
                        if content["type"] == "text":
                            metadata = {}
                            if message["role"] == smolagents.MessageRole.TOOL_CALL:
                                metadata = {"title": "🛠️ Tool call"}
                            if message["role"] == smolagents.MessageRole.TOOL_RESPONSE:
                                metadata = {"title": "➡️ Tool response"}
                            if message["role"] == smolagents.MessageRole.SYSTEM:
                                metadata = {"title": "️⚙️ System"}
                            messages.append(
                                gr.ChatMessage(
                                    role=role,
                                    content=content["text"],
                                    metadata=metadata,
                                ),
                            )
            if isinstance(step, smolagents.FinalAnswerStep):
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=f"**Final answer:**\n{step.output}",
                    ),
                )
            yield messages


with gr.Blocks() as app:
    chatbot = gr.Chatbot(type="messages", height=700)
    textbox = gr.Textbox(label="Task", value="")
    submit = gr.Button("Submit")
    submit.click(call_agent, inputs=textbox, outputs=chatbot)

app.launch(share=False)
