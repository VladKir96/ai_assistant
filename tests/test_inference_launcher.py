import subprocess
import sys

import pytest

from personal_assistant.app import cli


@pytest.mark.parametrize("explicit_override", [False, True])
def test_inference_uses_configured_file_and_allows_override(
    harness, tmp_path, monkeypatch, explicit_override
):
    configured = tmp_path / "configured.gguf"
    override = tmp_path / "override.gguf"
    configured.touch()
    override.touch()
    settings = harness.settings.model_copy(
        update={
            "local_model_file": configured,
            "llama_executable": "/test/runtime/llama-server",
        }
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    arguments = ["assistant", "inference"]
    if explicit_override:
        arguments += ["--model-file", str(override)]
    monkeypatch.setattr(sys, "argv", arguments)
    calls = []

    def launch(command, *, env, check):
        calls.append(command)
        assert check
        assert "LLAMA_API_KEY" in env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", launch)
    cli.main()
    expected = override if explicit_override else configured
    assert calls[0][0] == settings.llama_executable
    assert calls[0][calls[0].index("-m") + 1] == str(expected.resolve())
    assert calls[0][calls[0].index("--port") + 1] == "8080"


def test_vision_inference_passes_matching_projector(harness, tmp_path, monkeypatch):
    model, projector = tmp_path / "vision.gguf", tmp_path / "mmproj.gguf"
    model.touch()
    settings = harness.settings.model_copy(
        update={
            "model_supports_images": True,
            "local_model_file": model,
            "local_mmproj_file": projector,
        }
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["assistant", "inference"])
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    with pytest.raises(SystemExit, match="MMPROJ"):
        cli.main()
    assert commands == []
    projector.touch()
    cli.main()
    assert commands[0][commands[0].index("--mmproj") + 1] == str(projector.resolve())
