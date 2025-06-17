"""MCP tools utils."""

from smolagents import Tool, ToolCollection


def process_mcp_tools(tool_collection: ToolCollection) -> list[Tool]:
    """Process the MCP tools to add a mcp prefix to the name."""
    mcp_tools = tool_collection.tools
    for tool in mcp_tools:
        tool.name = f"mcp_{tool.name}"
    return mcp_tools
