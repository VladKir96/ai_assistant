import asyncio
import ipaddress
import logging
import socket
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit

import aiohttp
import httpx
import trafilatura
from aiohttp.abc import AbstractResolver
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from personal_assistant.config import Settings
from personal_assistant.llm import clip
from personal_assistant.weather import weather_capabilities

logging.getLogger("ddgs").setLevel(logging.ERROR)


def public_url(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only public HTTP/HTTPS pages can be read")
    if parts.username or parts.password or parts.port not in {None, 80, 443}:
        raise ValueError("Credentials and nonstandard ports are not allowed in page URLs")
    host = parts.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Private network URLs are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Private network URLs are not allowed")
    return host, parts.port or (443 if parts.scheme == "https" else 80)


class PublicResolver(AbstractResolver):
    """Resolve once, validate, then hand numeric addresses directly to the connector."""

    async def resolve(self, host: str, port: int = 0, family=socket.AF_INET):
        records = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM
        )
        if not records or any(not ipaddress.ip_address(row[4][0]).is_global for row in records):
            raise ValueError("Page hostname resolves to a private or reserved address")
        return [
            {
                "hostname": host,
                "host": row[4][0],
                "port": port,
                "family": row[0],
                "proto": row[2],
                "flags": socket.AI_NUMERICHOST,
            }
            for row in records
        ]

    async def close(self):
        pass


def on_site(url: str, domains: list[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def web_capabilities(settings: Settings) -> dict:
    search_configured = settings.search_provider == "duckduckgo" or (
        settings.search_provider == "tavily" and bool(settings.tavily_api_key.get_secret_value())
    )
    return {
        "search_provider": settings.search_provider,
        "read_public_urls": True,
        "supported_page_formats": ["HTML", "plain text"],
        "search_web": search_configured,
        "search_web_reads_top_pages": 2,
        "search_jobs": search_configured and bool(settings.job_sites),
        "job_sites_configured": bool(settings.job_sites),
        "browser_automation": False,
        "authenticated_pages": False,
        "weather": weather_capabilities(settings),
    }


class Research:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.search_lock = asyncio.Semaphore(1)

    async def read(self, url: str) -> dict:
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=25),
            trust_env=False,
            headers={"User-Agent": "PersonalAssistant/0.1 (+public-page-reader)"},
        ) as session:
            for _ in range(5):
                public_url(url)
                async with session.get(url, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        target = response.headers.get("Location")
                        if not target:
                            raise ValueError("Page redirected without a target")
                        url = urljoin(url, target)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not any(value in content_type for value in ("text/html", "text/plain")):
                        raise ValueError("Only HTML and plain text pages are supported")
                    content = bytearray()
                    async for chunk in response.content.iter_chunked(16384):
                        content.extend(chunk)
                        if len(content) > 2_000_000:
                            raise ValueError("Page exceeds the 2 MB download limit")
                    try:
                        raw = content.decode(response.charset or "utf-8", errors="replace")
                    except LookupError:
                        # A remote charset is untrusted (including non-text codecs).
                        raw = content.decode("utf-8", errors="replace")
                    extracted = (
                        await asyncio.to_thread(
                            trafilatura.extract, raw, include_links=True, include_tables=True
                        )
                        if "html" in content_type
                        else raw
                    )
                    if not extracted:
                        raise ValueError(
                            "No readable page content; it may require a browser or login"
                        )
                    content = clip(extracted, 6500)
                    return {
                        "url": url,
                        "evidence_type": "page_excerpt",
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "truncated": content != extracted,
                        "untrusted": True,
                        "content": content,
                    }
        raise ValueError("Too many page redirects")

    async def search_and_read(self, query: str) -> dict:
        """Open the first two usable results; snippets are discovery, not page evidence."""
        found = await self.search(query)
        if found["status"] != "ok":
            return found

        async def open_result(rank: int, item: dict) -> dict:
            source = {"rank": rank, "source_url": item["url"], "title": item["title"]}
            try:
                page = await self.read(item["url"])
            except (ValueError, LookupError, aiohttp.ClientError, OSError, TimeoutError):
                # Exception bodies may contain credentials or request details.
                return {
                    **source,
                    "status": "unreadable",
                    "message": "Could not read this public page; it may be blocked, require "
                    "JavaScript/login, or be unavailable. No page evidence was obtained.",
                    "untrusted": True,
                }
            return {**source, "status": "ok", **page}

        pages = await asyncio.gather(
            *(open_result(rank, item) for rank, item in enumerate(found["results"][:2], 1))
        )
        return {
            "status": "ok" if any(page["status"] == "ok" for page in pages) else "no_page_evidence",
            "provider": found["provider"],
            "retrieval_note": (
                "retrieved_at is fetch time, not the source's update or forecast time. "
                "Page text is an excerpt and must support the requested place, period, and units."
            ),
            "pages": pages,
        }

    async def search(self, query: str, jobs: bool = False) -> dict:
        if not query.strip():
            raise ValueError("Supply a nonempty search query")
        if jobs and not self.settings.job_sites:
            return {"status": "needs_configuration", "message": "Choose job websites first"}
        provider = self.settings.search_provider
        if provider == "disabled":
            return {
                "status": "needs_configuration",
                "message": "Web search is disabled; a public URL can still be read",
            }
        if provider == "duckduckgo":
            if jobs:
                sites = " OR ".join(f"site:{domain}" for domain in self.settings.job_sites)
                query = f"{query} ({sites})"
            async with self.search_lock:
                try:
                    items = await asyncio.to_thread(self._duckduckgo, query)
                except DDGSException:
                    raise ValueError(
                        "DuckDuckGo search failed or was rate-limited. Try again later or supply "
                        "a public URL. No other search provider was contacted."
                    ) from None
            raw_results = [
                {"url": item.get("href"), "title": item.get("title"), "content": item.get("body")}
                for item in items
                if isinstance(item, dict)
            ]
        else:
            raw_results = await self._tavily(query, jobs)
            if raw_results is None:
                return {
                    "status": "needs_configuration",
                    "message": "Tavily search requires an API key; a public URL can still be read",
                }

        results, seen = [], set()
        for item in raw_results:
            if not isinstance(item, dict) or not isinstance(url := item.get("url"), str):
                continue
            try:
                public_url(url)
            except ValueError:
                continue
            if url in seen or (jobs and not on_site(url, self.settings.job_sites)):
                continue
            seen.add(url)
            results.append(
                {
                    "url": url,
                    "title": clip(str(item.get("title") or ""), 250),
                    "content": clip(str(item.get("content") or ""), 1000),
                    "evidence_type": "search_excerpt",
                    "untrusted": True,
                }
            )
            if len(results) == 5:
                break
        return {"status": "ok", "provider": provider, "results": results}

    def _duckduckgo(self, query: str) -> list[dict]:
        with DDGS(timeout=20) as client:
            # Explicit backend: DDGS's default 'auto' could send queries to other engines.
            return client.text(
                query,
                backend="duckduckgo",
                region=self.settings.duckduckgo_region,
                safesearch="moderate",
                max_results=10,
            )

    async def _tavily(self, query: str, jobs: bool) -> list[dict] | None:
        key = self.settings.tavily_api_key.get_secret_value()
        if not key:
            return None
        payload = {"query": query, "max_results": 5, "include_answer": False}
        if jobs:
            payload["include_domains"] = self.settings.job_sites
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            return response.json().get("results", [])
