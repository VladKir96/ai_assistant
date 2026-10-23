import stat

import pytest

from personal_assistant.core.config import Settings
from personal_assistant.inference.model_profiles import ProfileError, load_profiles, select_profile


@pytest.fixture
def profile_files(tmp_path, monkeypatch):
    import os

    for key in os.environ:
        if key.startswith("ASSISTANT_"):
            monkeypatch.delenv(key)
    model = tmp_path / "model's folder" / "alternative.gguf"
    model.parent.mkdir()
    model.touch()
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        f"""[profiles.alternative]
model_provider = "local"
local_model = "alternative-model"
local_model_file = "{model.relative_to(tmp_path)}"
context_window = 32768
max_output_tokens = 4096
enable_thinking = true

[profiles.cloud]
model_provider = "openai"
openai_model = "test-cloud-model"
context_window = 16384
max_output_tokens = 2048
"""
    )
    env = tmp_path / ".env"
    env.write_text(
        "# Private configuration\n"
        "ASSISTANT_API_TOKEN='unchanged-test-api-token'\n"
        "ASSISTANT_DATABASE_URL=sqlite+aiosqlite:///personal.db\n"
        "ASSISTANT_OPENAI_API_KEY='unchanged-test-cloud-key'\n"
        "ASSISTANT_LLAMA_EXECUTABLE=/existing/llama-server\n"
        "ASSISTANT_PIPER_RU_MODEL=voices/russian.onnx\n"
        "ASSISTANT_EMBEDDING_MODEL=unchanged-embedding-model\n"
        "ASSISTANT_MODEL_PROVIDER=local\n"
        "ASSISTANT_LOCAL_MODEL=previous-model\n"
        "ASSISTANT_CONTEXT_WINDOW=16384\n"
        "ASSISTANT_MAX_OUTPUT_TOKENS=2048"
    )
    return profiles, env, model


def test_switch_local_cloud_and_back_preserves_private_configuration(profile_files):
    profiles, env, model = profile_files
    private_lines = env.read_text().splitlines()[:7]

    select_profile("alternative", profiles, env)
    settings = Settings(_env_file=env)
    assert settings.local_model == "alternative-model"
    assert env.parent / settings.local_model_file == model
    assert settings.context_window == 32768
    assert settings.max_output_tokens == 4096
    assert settings.enable_thinking
    assert stat.S_IMODE(env.stat().st_mode) == 0o600

    select_profile("cloud", profiles, env)
    settings = Settings(_env_file=env)
    assert settings.model_provider == "openai"
    assert settings.openai_model == "test-cloud-model"
    assert settings.local_model_file is None
    assert settings.context_window == 16384
    assert not settings.enable_thinking

    select_profile("alternative", profiles, env)
    settings = Settings(_env_file=env)
    assert settings.model_provider == "local"
    assert settings.local_model == "alternative-model"
    assert settings.openai_api_key.get_secret_value() == "unchanged-test-cloud-key"
    assert settings.database_url == "sqlite+aiosqlite:///personal.db"
    assert settings.llama_executable == "/existing/llama-server"
    assert settings.piper_ru_model == "voices/russian.onnx"
    assert settings.embedding_model == "unchanged-embedding-model"
    assert env.read_text().splitlines()[:7] == private_lines


@pytest.mark.parametrize("failure", ["missing_model", "unknown_profile", "missing_cloud_key"])
def test_failed_selection_does_not_modify_env(profile_files, failure):
    profiles, env, model = profile_files
    name = "alternative"
    if failure == "missing_model":
        model.unlink()
    elif failure == "unknown_profile":
        name = "unknown"
    else:
        name = "cloud"
        env.write_text(env.read_text().replace("unchanged-test-cloud-key", ""))
    before = env.read_bytes()
    with pytest.raises(ProfileError):
        select_profile(name, profiles, env)
    assert env.read_bytes() == before


@pytest.mark.parametrize(
    "extra", ["api_token = 'do-not-print-this'", "database_url = 'do-not-print-this'"]
)
def test_profiles_cannot_override_credentials_or_storage(profile_files, extra):
    profiles, _, _ = profile_files
    profiles.write_text(profiles.read_text() + "\n" + extra + "\n")
    with pytest.raises(ProfileError) as caught:
        load_profiles(profiles)
    assert "do-not-print-this" not in str(caught.value)


def test_exported_environment_still_overrides_selected_profile(profile_files, monkeypatch):
    profiles, env, _ = profile_files
    select_profile("alternative", profiles, env)
    monkeypatch.setenv("ASSISTANT_LOCAL_BASE_URL", "http://inference:8080/v1")
    assert Settings(_env_file=env).local_base_url == "http://inference:8080/v1"


def test_vision_profile_requires_projector_and_text_profile_clears_it(profile_files):
    profiles, env, model = profile_files
    projector = model.parent / "mmproj.gguf"
    profiles.write_text(
        profiles.read_text()
        + f'''
[profiles.vision]
model_provider = "local"
local_model = "synthetic-vision-model"
local_model_file = "{model.relative_to(env.parent)}"
local_mmproj_file = "{projector.relative_to(env.parent)}"
model_supports_images = true
context_window = 16384
max_output_tokens = 2048
'''
    )
    original = env.read_bytes()
    with pytest.raises(ProfileError, match="mmproj"):
        select_profile("vision", profiles, env)
    assert env.read_bytes() == original
    projector.touch()
    select_profile("vision", profiles, env)
    settings = Settings(_env_file=env)
    assert settings.model_supports_images and settings.local_mmproj_file
    select_profile("alternative", profiles, env)
    settings = Settings(_env_file=env)
    assert not settings.model_supports_images and settings.local_mmproj_file is None
