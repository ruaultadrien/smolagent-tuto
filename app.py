"""Define the Gradio interface for the agent."""

import gradio as gr

from main import get_agent


def call_agent(prompt: str) -> str:
    """Get the agent and call it with the prompt."""
    agent = get_agent()
    return agent.run(prompt)


app = gr.Interface(fn=call_agent, inputs="text", outputs="text")
app.launch()
