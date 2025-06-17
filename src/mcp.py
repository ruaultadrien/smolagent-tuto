from smolagents import ToolCollection, Tool


def process_mcp_tools(tool_collection: ToolCollection) -> list[Tool]:
    mcp_tools = tool_collection.tools
    for tool in mcp_tools:
        tool.name = f"mcp_{tool.name}"
    return mcp_tools
