import asyncio
import copy
import json
import os
import sys
from contextlib import asynccontextmanager
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field

from personal_assistant.config import Settings
from personal_assistant.llm import clip


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str
    args: list[str] = Field(default_factory=list)
    read_tools: list[str] = Field(default_factory=list)
    environment_variables: list[str] = Field(default_factory=list)


class MCPTools:
    """Explicit, trusted stdio integrations. Agent cannot install servers or change this config."""

    def __init__(self, settings: Settings):
        self.servers = {}
        self.warnings: list[str] = []
        self._catalog_cache: dict[str, tuple[float, list[dict], bool]] = {}
        self._catalog_lock = asyncio.Lock()
        if settings.mcp_servers_file:
            raw = json.loads(settings.mcp_servers_file.read_text())
            self.servers = {name: ServerConfig.model_validate(value) for name, value in raw.items()}
        if settings.pdf_enabled:
            if "pdf-reader" in self.servers:
                raise ValueError("pdf-reader is reserved while ASSISTANT_PDF_ENABLED=true")
            args = [
                "-m",
                "personal_assistant.pdf_mcp",
                "--root",
                str(settings.pdf_directory.resolve()),
                "--uploads",
                str(settings.pdf_upload_directory.resolve()),
            ]
            if settings.pdf_tessdata_directory:
                args += ["--tessdata", str(settings.pdf_tessdata_directory.resolve())]
            self.servers["pdf-reader"] = ServerConfig(
                command=sys.executable,
                args=args,
                read_tools=["list_pdf_files", "read_pdf_pages"],
            )

    @asynccontextmanager
    async def connect(self, name: str):
        if name not in self.servers:
            raise ValueError("MCP server is not configured")
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as error:
            raise ValueError(
                "Install the mcp optional dependency to use MCP integrations"
            ) from error
        config = self.servers[name]
        env = {key: os.environ[key] for key in config.environment_variables if key in os.environ}
        parameters = StdioServerParameters(command=config.command, args=config.args, env=env)
        async with (
            asyncio.timeout(30),
            stdio_client(parameters) as streams,
            ClientSession(*streams) as session,
        ):
            await session.initialize()
            yield session

    async def catalog(self) -> list[dict]:
        async with self._catalog_lock:
            self.warnings = []

            async def discover(name, config):
                cached = self._catalog_cache.get(name)
                if cached and monotonic() < cached[0]:
                    if cached[2]:
                        self.warnings.append(
                            "An MCP integration is unavailable; other tools remain available."
                        )
                    return cached[1]
                try:
                    async with asyncio.timeout(5), self.connect(name) as session:
                        tools = (await session.list_tools()).tools
                        entries = [
                            {
                                "server": name,
                                "tool": tool.name,
                                "description": clip(tool.description or "", 300),
                                "arguments_schema": tool.inputSchema,
                            }
                            for tool in tools
                            if tool.name in config.read_tools
                        ]
                    self._catalog_cache[name] = (monotonic() + 300, entries, False)
                    return entries
                except Exception:
                    # MCP can wrap transport failures in ExceptionGroup. Never expose its text.
                    self._catalog_cache[name] = (monotonic() + 30, [], True)
                    self.warnings.append(
                        "An MCP integration is unavailable; other tools remain available."
                    )
                    return []

            groups = await asyncio.gather(
                *(discover(name, config) for name, config in self.servers.items())
            )
            result = []
            for entry in (entry for group in groups for entry in group):
                if len(json.dumps([*result, entry]).encode()) <= 2200:
                    result.append(entry)
                else:
                    self.warnings.append(
                        "Some MCP tools were omitted to fit the catalog; enable fewer tools."
                    )
            self.warnings = sorted(set(self.warnings))
            return copy.deepcopy(result)

    async def read(self, server: str, tool: str, arguments: dict) -> dict:
        config = self.servers.get(server)
        if not config or tool not in config.read_tools:
            raise ValueError("MCP tool has not been explicitly allowed for read access")
        try:
            async with self.connect(server) as session:
                result = await session.call_tool(tool, arguments)
                if result.isError:
                    raise ValueError("MCP tool reported an error")
                return {
                    "untrusted": True,
                    "server": server,
                    "tool": tool,
                    "content": clip(
                        json.dumps(result.model_dump(mode="json"), ensure_ascii=False), 3000
                    ),
                }
        except Exception:
            raise ValueError("MCP tool unavailable or failed; no result was obtained") from None
