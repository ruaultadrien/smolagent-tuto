"""Sandbox for using the smolagents library."""

import base64
import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, Model
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.smolagents import SmolagentsInstrumentor


# Tool to list the available occasions
@tool
def list_occasions() -> str:
    """
    Lists the available occasions.
    """
    return "casual, formal, superhero, custom"


# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


def get_agent(model: Model) -> CodeAgent:
    """Returns the tuto agent."""
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), suggest_menu, list_occasions],
        model=model,
        additional_authorized_imports=["datetime"],
    )
    return agent


def setup_langfuse():
    """Setup Langfuse."""
    langfuse_auth = base64.b64encode(
        f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}".encode()
    ).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        "https://cloud.langfuse.com/api/public/otel"  # EU data region
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
