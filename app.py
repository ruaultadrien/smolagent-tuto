"""Define the Gradio interface for the agent."""

import logging
import os

import gradio as gr
from huggingface_hub import login
from smolagents import CodeAgent, LiteLLMModel

from src.utils import get_agent, setup_langfuse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def call_agent(prompt: str) -> str:
    """Get the agent and call it with the prompt."""
    setup_langfuse()

    # Login to Hugging Face Hub
    login(token=os.getenv("HF_TOKEN"))

    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    agent = get_agent(model=model, code_agent=False)

    # Pushing to hub
    agent.push_to_hub("ruaultadrienperso/AlfredAgent")

    # Pulling from hub
    agent = CodeAgent.from_hub("ruaultadrienperso/AlfredAgent", trust_remote_code=True)

    return agent.run(prompt)


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch(share=False)
