"""Private, content-addressed PDF uploads, read through the existing MCP adapter."""

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

from personal_assistant.core.config import Settings
from personal_assistant.inference.llm import ModelError
from personal_assistant.integrations.mcp_tools import MCPTools

MAX_PDF_UPLOAD_BYTES = 10_000_000


class PDFInputError(ValueError):
    pass


class PDFUnavailable(PDFInputError):
    pass


def display_filename(filename: str) -> str:
    # Display only: client filenames never determine a storage path.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()[:180]
    return name or "document.pdf"


class PDFUploads:
    def __init__(self, settings: Settings, mcp: MCPTools):
        self.enabled = settings.pdf_enabled
        self.directory = settings.pdf_upload_directory.resolve()
        self.mcp = mcp

    def store(self, raw: bytes, filename: str) -> dict:
        if not raw or len(raw) > MAX_PDF_UPLOAD_BYTES:
            raise PDFInputError("PDF upload limit: one file up to 10 MB")
        try:
            import pymupdf
        except ImportError:
            raise PDFUnavailable("PDF dependencies are missing; run `uv sync --frozen`.") from None
        try:
            with pymupdf.open(stream=raw, filetype="pdf") as document:
                if not document.is_pdf or document.needs_pass or not len(document):
                    raise PDFInputError("Send a valid, nonempty, unencrypted PDF")
                pages = len(document)
        except PDFInputError:
            raise
        except Exception:
            raise PDFInputError("PDF is invalid, damaged or password-protected") from None
        digest = hashlib.sha256(raw).hexdigest()
        target = self.directory / f"{digest}.pdf"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A completed file is linked into place atomically. Existing files/symlinks are
        # never overwritten, and a truncated/crashed upload never becomes readable.
        descriptor, temporary = tempfile.mkstemp(prefix=".upload-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.stat().st_size != len(raw)
                    or target.read_bytes() != raw
                ):
                    raise PDFInputError(
                        "Stored PDF conflicts with this upload; check the upload directory"
                    ) from None
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "untrusted": True,
            "filename": display_filename(filename),
            "sha256": digest,
            "file_path": f"uploads/{digest}.pdf",
            "page_count": pages,
            "initial_pages": [1, min(pages, 3)],
        }

    async def extract(self, attachment: dict) -> dict:
        try:
            result = await self.mcp.read(
                "pdf-reader",
                "read_pdf_pages",
                {
                    "file_path": attachment["file_path"],
                    "start_page": 1,
                    "end_page": attachment["initial_pages"][1],
                },
            )
        except ValueError:
            raise ModelError(
                "PDF was saved, but extraction failed. Check the PDF MCP setup and retry."
            ) from None
        return {**attachment, "excerpt": result["content"]}

    async def prepare(self, raw: bytes, filename: str) -> dict:
        if not self.enabled:
            raise PDFUnavailable(
                "PDF uploads are disabled; enable ASSISTANT_PDF_ENABLED and restart the API"
            )
        try:
            return await asyncio.to_thread(self.store, raw, filename)
        except OSError:
            raise PDFInputError(
                "PDF could not be saved; check the upload directory permissions and free space"
            ) from None
