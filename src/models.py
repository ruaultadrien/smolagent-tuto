"""Models module."""

import os

from smolagents import InferenceClientModel, LiteLLMModel, Model


def get_model(model_name: str) -> Model:
    """Return a model based on the model name passed as argument."""
    if model_name == "mistral":
        return get_mistral_model()
    if model_name == "deepseek":
        return get_deepseek_model()
    msg = f"Model {model_name} not found."
    raise ValueError(msg)


def get_mistral_model() -> Model:
    """Return a Mistral model."""
    return LiteLLMModel(
        model_id="mistral/mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        max_tokens=8096,
    )


def get_deepseek_model() -> Model:
    """Return a DeepSeek model."""
    return InferenceClientModel(
        "deepseek-ai/DeepSeek-R1",
        max_tokens=8096,
    )
