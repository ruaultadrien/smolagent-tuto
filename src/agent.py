"""Sandbox for using the smolagents library."""

import base64
import logging
import os

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
)


def get_agent(model: Model, tools: list, code_agent: bool) -> CodeAgent:
    """Returns the tuto agent."""
    if code_agent:
        logging.info("Creating code agent")
        agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=["datetime"],
            add_base_tools=True,
        )
    else:
        logging.info("Creating tool calling agent")
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
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
