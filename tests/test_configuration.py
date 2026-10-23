import stat
from io import StringIO
from pathlib import Path

import pytest
from dotenv import dotenv_values
from dotenv.parser import parse_stream

from personal_assistant.app.cli import initialize
from personal_assistant.core.config import Settings

TEMPLATE = Path(__file__).resolve().parents[1] / ".env.example"


def test_environment_template_documents_every_setting_once():
    bindings = list(parse_stream(StringIO(TEMPLATE.read_text())))
    assert not any(binding.error for binding in bindings)
    keys = [binding.key for binding in bindings if binding.key]
    assert len(keys) == len(set(keys))
    assert set(keys) == {"ASSISTANT_" + field.upper() for field in Settings.model_fields}
    settings = Settings(_env_file=TEMPLATE)
    assert not settings.voice_enabled and not settings.voice_output_enabled
    assert not settings.pdf_enabled and not settings.python_enabled


def test_initializer_uses_documented_template_and_preserves_existing_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text(TEMPLATE.read_text())
    initialize()
    target = tmp_path / ".env"
    values = dotenv_values(target)
    assert set(values) == set(dotenv_values(TEMPLATE))
    assert len(values["ASSISTANT_API_TOKEN"]) >= 32
    assert len(values["ASSISTANT_LOCAL_API_KEY"]) >= 32
    assert len(values["ASSISTANT_POSTGRES_PASSWORD"]) >= 24
    assert values["ASSISTANT_POSTGRES_PASSWORD"] in values["ASSISTANT_DATABASE_URL"]
    assert values["ASSISTANT_EMBEDDING_API_KEY"] == values["ASSISTANT_API_TOKEN"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    original = target.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):
        initialize()
    assert target.read_bytes() == original
