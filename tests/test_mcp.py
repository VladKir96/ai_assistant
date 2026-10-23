import json
import sys
from pathlib import Path

import pytest

from personal_assistant.integrations.mcp_tools import MCPTools


async def test_stdio_mcp_discovery_call_and_allowlist(harness, tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "fixture": {
                    "command": sys.executable,
                    "args": [str(Path(__file__).with_name("mcp_fixture_server.py"))],
                    "read_tools": ["read_label"],
                }
            }
        )
    )
    settings = harness.settings.model_copy(update={"mcp_servers_file": config})
    tools = MCPTools(settings)
    catalog = await tools.catalog()
    assert [(item["server"], item["tool"]) for item in catalog] == [("fixture", "read_label")]
    response = await tools.read("fixture", "read_label", {"label": "English / Русский"})
    assert response["untrusted"] and "Synthetic label" in response["content"]
    with pytest.raises(ValueError, match="allowed"):
        await tools.read("fixture", "delete_everything", {})
