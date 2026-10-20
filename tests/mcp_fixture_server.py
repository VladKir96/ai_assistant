from mcp.server.fastmcp import FastMCP

server = FastMCP("fixture")


@server.tool()
def read_label(label: str) -> str:
    """Read a synthetic label."""
    return "Synthetic label: " + label


if __name__ == "__main__":
    server.run(transport="stdio")
