"""Read-only, bounded stdio adapter for the pinned mcp-pdf-reader package.

No .env is loaded here, so this subprocess does not receive assistant credentials.
The upstream server's write tool is never registered on this server.
"""

import argparse
import json
import logging
import os
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Literal

from personal_assistant.inference.llm import clip

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PAGES = 3


class PDFReader:
    def __init__(self, root: Path, tessdata: Path | None = None, uploads: Path | None = None):
        self.root = root.resolve()
        self.tessdata = tessdata.resolve() if tessdata else None
        self.uploads = uploads.resolve() if uploads else None

    def reference(self, path: Path) -> str:
        if self.uploads and path.is_relative_to(self.uploads):
            return "uploads/" + str(path.relative_to(self.uploads))
        return str(path.relative_to(self.root))

    def file(self, file_path: str) -> Path:
        supplied = Path(file_path)
        uploaded = supplied.parts[:1] == ("uploads",) and self.uploads is not None
        root = self.uploads if uploaded else self.root
        relative = Path(*supplied.parts[1:]) if uploaded else supplied
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".pdf":
            raise ValueError("Only PDFs inside the configured PDF directory can be read")
        if not path.is_file():
            raise ValueError("PDF not found; use list_pdf_files for available names")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("PDF exceeds the 50 MiB file limit")
        return path

    def list_files(self, offset: int = 0) -> str:
        if offset < 0 or offset > 500:
            raise ValueError("offset must be between 0 and 500")
        if not self.root.is_dir() and not (self.uploads and self.uploads.is_dir()):
            return "PDF directory is empty or missing; create it and add PDFs first."
        # Bound both the directory scan and response. Symlinks are checked before listing.
        files = []
        scanned = 0
        from itertools import chain

        candidates = chain(
            self.root.iterdir() if self.root.is_dir() else (),
            self.uploads.iterdir() if self.uploads and self.uploads.is_dir() else (),
        )
        for path in candidates:
            scanned += 1
            if scanned > 500:
                break
            try:
                reference = self.reference(path)
                self.file(reference)
            except (OSError, ValueError):
                continue
            files.append(reference)
        files.sort()
        selected = []
        for name in files[offset : offset + 20]:
            if len(json.dumps([*selected, name], ensure_ascii=False).encode()) > 1400:
                break
            selected.append(name)
        next_offset = offset + len(selected)
        return json.dumps(
            {
                "files": selected,
                "next_offset": next_offset if next_offset < len(files) else None,
                "scan_limited": scanned > 500,
                "scope": "top-level PDFs and uploads/; subfolders need an exact relative path",
            },
            ensure_ascii=False,
        )

    def read(
        self,
        file_path: str,
        start_page: int = 1,
        end_page: int | None = None,
        ocr: bool = False,
        language: Literal["eng", "rus", "eng+rus"] = "eng+rus",
    ) -> str:
        import pdf_server
        import pymupdf

        path = self.file(file_path)
        if start_page < 1 or (end_page is not None and end_page < start_page):
            raise ValueError("Use a positive, ascending 1-based page range")
        with pymupdf.open(path) as document:
            if not document.is_pdf or document.needs_pass:
                raise ValueError("A readable, unencrypted PDF is required")
            total = len(document)
            end = min(total, start_page + MAX_PAGES - 1) if end_page is None else end_page
            if start_page > total or end > total or end - start_page >= MAX_PAGES:
                raise ValueError("Request at most three existing pages at a time")
            if ocr:
                if not self.tessdata or any(
                    not (self.tessdata / f"{code}.traineddata").is_file()
                    for code in language.split("+")
                ):
                    raise ValueError("OCR requires configured Tesseract eng/rus traineddata files")
                for page_number in range(start_page - 1, end):
                    area = document[page_number].rect.get_area() * (150 / 72) ** 2
                    if area > 12_000_000:
                        raise ValueError("Page is too large for bounded OCR at 150 DPI")

        try:
            if ocr:
                # Upstream logs OCR failures and silently returns embedded text. Surface failure
                # instead, so a scanned document is never claimed to have been recognized.
                class OCRFailure(logging.Handler):
                    failed = False

                    def emit(self, record):
                        if record.pathname == pdf_server.__file__ and record.thread == thread_id:
                            self.failed = True

                thread_id = threading.get_ident()
                failure = OCRFailure(level=logging.WARNING)
                logger = logging.getLogger()
                logger.addHandler(failure)
                try:
                    result = pdf_server.read_by_ocr.fn(
                        str(path), start_page, end, language=language, dpi=150
                    )
                finally:
                    logger.removeHandler(failure)
                if failure.failed:
                    raise ValueError("OCR failed; check Tesseract language data and PDF quality")
                text = result["text"]
            else:
                text = pdf_server.read_pdf_text.fn(str(path), start_page, end)
        except ValueError:
            raise
        except Exception:
            raise ValueError("PDF extraction failed; no text was obtained") from None
        header = (
            f"File: {self.reference(path)}; pages {start_page}-{end} of {total}; "
            f"method: {'OCR' if ocr else 'embedded text'}. "
            "Bounded excerpt; omitted text/pages are not analyzed.\n"
        )
        if not ocr and not any(
            line.strip() for line in text.splitlines() if not line.startswith("----------")
        ):
            header += "No embedded text found; scanned pages may require OCR.\n"
        return header + clip(text, 2000)


def create_server(root: Path, tessdata: Path | None = None, uploads: Path | None = None):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    # Upstream imports the deprecated fitz alias, which can print a warning to stdout.
    # Import before starting stdio and keep that diagnostic off the JSON-RPC transport.
    with redirect_stdout(sys.stderr):
        __import__("pdf_server")

    reader = PDFReader(root, tessdata, uploads)
    server = FastMCP("PDF reader (mcp-pdf-reader)")
    annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )

    @server.tool(annotations=annotations, structured_output=False)
    def list_pdf_files(offset: int = 0) -> str:
        """List local PDF filenames, 20 at a time. Use next_offset for another batch."""
        return reader.list_files(offset)

    @server.tool(
        annotations=annotations,
        structured_output=False,
        description=(
            "Read <=3 local PDF pages (1-based); bounded excerpt. Cite filename/pages. "
            "Use relative file_path, including uploads/ for attached PDFs; no URL downloads. "
            + (
                "OCR needs installed Tesseract language files."
                if tessdata
                else "OCR unconfigured: use ocr=false."
            )
        ),
    )
    def read_pdf_pages(
        file_path: str,
        start_page: int = 1,
        end_page: int | None = None,
        ocr: bool = False,
        language: Literal["eng", "rus", "eng+rus"] = "eng+rus",
    ) -> str:
        """Read up to 3 PDF pages (1-based); bounded excerpt. Relative path in PDF folder.
        ocr=true needs configured Tesseract language files; otherwise extract embedded text.
        """
        return reader.read(file_path, start_page, end_page, ocr, language)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--uploads", type=Path)
    args = parser.parse_args()
    if args.tessdata:
        os.environ["TESSDATA_PREFIX"] = str(args.tessdata.resolve())
    else:
        os.environ.pop("TESSDATA_PREFIX", None)
    create_server(args.root, args.tessdata, args.uploads).run(transport="stdio")


if __name__ == "__main__":
    main()
