FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.15
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra mcp --extra pdf --extra python-runner --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra mcp --extra pdf --extra python-runner
COPY alembic.ini ./
COPY migrations ./migrations
RUN useradd --create-home --uid 10001 assistant && mkdir -p /app/data/pdf-uploads /app/data/artifacts && chown -R assistant:assistant /app
USER assistant
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["assistant", "api", "--host", "0.0.0.0"]

