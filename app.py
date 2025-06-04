"""Define the Gradio interface for the agent."""

import os
import gradio as gr

from smolagents import CodeAgent, LiteLLMModel
from utils import get_agent, setup_langfuse


def call_agent(prompt: str) -> str:
    """Get the agent and call it with the prompt."""
    setup_langfuse()

    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    agent = get_agent(model=model)

    # Pushing to hub
    agent.push_to_hub("ruaultadrienperso/AlfredAgent")

    # Pulling from hub
    agent = CodeAgent.from_hub("ruaultadrienperso/AlfredAgent", trust_remote_code=True)

    return agent.run(prompt)


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch()
