# Local PDF reading through MCP

The assistant uses [mcp-pdf-reader](https://pypi.org/project/mcp-pdf-reader/) 0.1.7,
with compatible FastMCP 2 and PyMuPDF 1 dependencies pinned in `uv.lock`. A managed
stdio adapter exposes only `list_pdf_files` and `read_pdf_pages`. It delegates extraction
to the package's text/OCR functions and never registers its write-to-file tool.

## Native Mac setup

From the project root:

```sh
uv sync --frozen --extra dev --extra mcp --extra speech --extra pdf
mkdir -p data/pdf data/tessdata
```

Keep extras for other enabled companions, such as `--extra embeddings`, when syncing.
In your existing private `.env`, add or update:

```dotenv
ASSISTANT_PDF_ENABLED=true
ASSISTANT_PDF_DIRECTORY=data/pdf
ASSISTANT_PDF_UPLOAD_DIRECTORY=data/pdf-uploads
ASSISTANT_PDF_TESSDATA_DIRECTORY=
```

Send a PDF as a Telegram **file/document**, or copy a local PDF into `data/pdf`.
Restart the launcher after installing this update:

```sh
uv run --extra speech --extra pdf assistant start
```

For a text-only setup, omit `--extra speech`. The assistant launches and closes its
stdio server automatically during MCP discovery/reads. No extra terminal, network
port, API key, or generic MCP JSON configuration is needed. `pdf-reader` is a reserved
server name while the managed integration is enabled. Other configured servers remain
available, subject to the shared catalog budget.

Examples to send as ordinary Telegram text messages:

- `Which PDF files are available?`
- `Read the first page of resume.pdf.`
- `Summarize pages 2–3 of contract.pdf and cite the page numbers.`
- `Run OCR on the first page of scan.pdf.`

Name the exact file when reading it. Directory discovery and a read requiring a
discovered filename may need separate messages. Listing covers top-level local PDFs
and uploads in batches of 20; an exact relative path can address a subfolder. At most 500 directory
entries are scanned per listing. The result reports when that scan is limited.

## Telegram PDF uploads

Attach one PDF as a file in the owner's private bot chat. Add a question as its caption,
for example `Briefly summarize this resume` or `Summarize pages 4–5`. Without a caption,
the bot asks for a brief summary of the first three pages using an English default request.
Add a caption to choose the response language.
Uploads work with text-only models; image/vision support is not required. Files with
PDF MIME type or a `.pdf` filename are routed to PDF handling, but decoded content is
validated on the API. Fake, damaged, empty and password-protected PDFs are rejected.

The limit is **one PDF per message, 10 MB**. Albums are rejected. Owner/private-chat,
announced size and actual streamed-download size are checked before an upload reaches
the API. Send scanned PDFs with a caption asking for OCR; this still requires the
configured language data described below.

The API retains the original bytes in `ASSISTANT_PDF_UPLOAD_DIRECTORY`, default
`data/pdf-uploads`, with private file permissions and a SHA-256 filename. Client names
are display metadata and cannot choose a filesystem path. Identical bytes reuse the
same file; different PDFs with the same original name do not overwrite one another.
The MCP reader exposes these files as `uploads/<sha256>.pdf` and lists them alongside
local PDFs. Filename/reference and excerpt are retained in conversation history for
follow-up questions. The files are not automatically added to semantic/vector memory.

Each upload first obtains a bounded MCP excerpt of pages 1–3 (or fewer). The agents
can read another range or request OCR using the exact uploaded file reference. All
PDF replies require response review; incomplete excerpts never establish whole-file
analysis. Original file bytes stay on disk; only digest/metadata/excerpts enter the
database and checkpoints. Uploading authorizes storing that file, not note/fact writes.

Migration `0004` adds upload identity and cached evidence for retries. `assistant start`
applies it; for manual startup run `uv run assistant migrate` before restarting the
API and bot. Matching request ID/text/bytes reuse completed analysis/checkpoints;
different or missing attachments under that ID are rejected. Failed question processing
may still leave the uploaded PDF stored for retry.

The authenticated multipart API endpoint is `POST /v1/chat/pdf`, with fields `pdf`,
optional `text`, `thread_id`, and UUID `request_id`; it returns the usual `ChatResponse`.
Repeated/extra fields are rejected. Authentication/body limits run before multipart
spooling. Disabled PDF support returns 415, invalid PDF/fields 422, oversized upload
413, conflicting request identity 409, and unavailable MCP extraction 503.

Uploaded files have no automatic expiration. Include the upload directory in backups;
remove files there manually when no longer needed. Removing a file does not erase
its archived text excerpts or run/checkpoint history. Restarting the bot retains files.

## Optional OCR for scanned PDFs

Normal text extraction needs no OCR models. OCR uses PyMuPDF's integrated Tesseract
engine and requires language support data; see the
[PyMuPDF OCR setup](https://pymupdf.readthedocs.io/en/latest/installation.html#enabling-integrated-ocr-support).
Place `eng.traineddata` and `rus.traineddata` from the official
[tessdata_fast repository](https://github.com/tesseract-ocr/tessdata_fast) in
`data/tessdata`. These commands use the revision verified with this project:

```sh
mkdir -p data/tessdata
curl --fail --location --output data/tessdata/eng.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/eng.traineddata
curl --fail --location --output data/tessdata/rus.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/rus.traineddata
shasum -a 256 data/tessdata/*.traineddata
```

Expected SHA-256 checksums:

| File | SHA-256 |
| --- | --- |
| `eng.traineddata` | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |
| `rus.traineddata` | `e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225` |

Then set:

```dotenv
ASSISTANT_PDF_TESSDATA_DIRECTORY=data/tessdata
```

Restart the assistant after changing settings. Supported language choices are `eng`,
`rus`, and `eng+rus` (default). OCR is explicit (`ocr=true`), runs at 150 DPI, and rejects
pages above 12 million rendered pixels. Missing language data or upstream OCR errors
fail the read rather than silently presenting fallback text as recognized text.
No language models are downloaded during chat or startup.

## Docker

The application image includes the PDF extra. Compose mounts `data/pdf` and
`data/tessdata` read-only at `/app/data/pdf` and `/app/data/tessdata`; its environment
overrides the native paths. Create these directories before starting the API. Ensure
the container user (UID 10001) can read the mounted directories and files.
Uploads use the separate writable named volume `pdf-uploads`, mounted at
`/app/data/pdf-uploads`; the image creates its initial directory for UID 10001. The
reference PDF and language-data bind mounts remain read-only. Native uploads and Docker
uploads use separate storage by default: migrate/back up their files explicitly when
switching deployment modes. `docker compose down -v` deletes the upload volume too.

```sh
mkdir -p data/pdf data/tessdata
docker compose up -d --build --wait api
docker compose --profile telegram up -d --force-recreate bot
```

Enable `ASSISTANT_PDF_ENABLED=true` in `.env` for Docker too. OCR with the default `eng+rus` setting needs
the language files in the mounted tessdata folder. An empty tessdata folder permits
embedded-text reads; OCR will report failure if requested without the files.

## Bounds and privacy

Only `.pdf` files resolving inside the configured reference/upload folders are accepted,
with checks against traversal and symlinks escaping either root. The local-read limit is 50 MiB per
file and three existing pages per call. The default range starts at page 1 and ends
at page 3, or the final page if earlier. Password-protected PDFs are rejected.

Each result contains the source filename, page range, total page count, extraction
method, and a bounded text excerpt (2,000 UTF-8 bytes). The existing MCP/tool budgets
can shorten it further. A request for several pages does not guarantee their complete
text fits; ask focused questions or request narrower ranges. The graph treats the
result as untrusted evidence and sends PDF answers through its response reviewer.

Reading a PDF does not add it to semantic memory. Long-document chunking/indexing,
whole-document analysis, automatic URL downloads,
image extraction, PDF edits and TXT exports are outside this integration. Explicit
user-supplied text can still be saved through the existing notes tools.

The PDF subprocess receives no assistant credentials and does not load `.env`. Folder
validation is an application guard, not OS process isolation: use trusted local files
and server packages. Docker adds a non-root process, read-only reference mounts and a
writable upload volume. Files in
`data/` remain Git-ignored. Text excerpts enter the configured model's context, so a
hosted model receives those excerpts if you select it.

If a read fails, check the PDF extra, folder/file permissions, exact filename, page
range, encryption, and OCR language files. A generic MCP failure is deliberately
sanitized. The launcher checks missing PDF dependencies/folder before starting services;
unavailable MCP discovery otherwise warns without disabling ordinary conversation.
