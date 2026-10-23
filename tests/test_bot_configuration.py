import sys

import pytest

from personal_assistant.app import bot, cli


@pytest.mark.parametrize(
    "token", ["", "secret-suffix-only", "12345:", "12345:secret with spaces", "https://example.org"]
)
async def test_invalid_token_fails_before_creating_bot(harness, monkeypatch, token):
    from pydantic import SecretStr

    settings = harness.settings.model_copy(
        update={"telegram_token": SecretStr(token), "telegram_owner_id": 12345}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid configuration must fail before creating network clients")

    monkeypatch.setattr(bot, "Bot", forbidden)
    with pytest.raises(bot.BotConfigurationError) as caught:
        await bot.run_bot(settings)
    assert "ASSISTANT_TELEGRAM_TOKEN" in str(caught.value)
    if token:
        assert token not in str(caught.value)


async def test_negative_owner_fails_before_creating_bot(harness, monkeypatch):
    from pydantic import SecretStr

    settings = harness.settings.model_copy(
        update={"telegram_token": SecretStr("12345:testing-only"), "telegram_owner_id": -1}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Owner configuration must be checked before creating network clients")

    monkeypatch.setattr(bot, "Bot", forbidden)
    with pytest.raises(bot.BotConfigurationError, match="ASSISTANT_TELEGRAM_OWNER_ID"):
        await bot.run_bot(settings)


def test_cli_exits_with_actionable_message_without_echoing_token(harness, monkeypatch):
    from pydantic import SecretStr

    settings = harness.settings.model_copy(
        update={"telegram_token": SecretStr("secret-suffix-only"), "telegram_owner_id": 12345}
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["assistant", "bot"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert "BotFather" in str(caught.value)
    assert "secret-suffix-only" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("owner_id", [0, 12345])
async def test_private_id_bootstrap_preserves_assistant_access_restrictions(
    harness, monkeypatch, caplog, owner_id
):
    from aiogram.types import Update
    from pydantic import SecretStr

    settings = harness.settings.model_copy(
        update={"telegram_token": SecretStr("12345:testing-only"), "telegram_owner_id": owner_id}
    )
    replies = []

    async def fake_call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def fake_polling(dispatcher, instance, **kwargs):
        for index, (text, chat_type) in enumerate(
            [("/id", "private"), ("/notes", "private"), ("/start", "private"), ("/id", "group")]
        ):
            message = {
                "message_id": index + 1,
                "date": 0,
                "from": {"id": 98765, "is_bot": False, "first_name": "Test"},
                "chat": {"id": 98765 if chat_type == "private" else -98765, "type": chat_type},
                "text": text,
            }
            update = Update.model_validate({"update_id": index, "message": message})
            await dispatcher.feed_update(instance, update)

    monkeypatch.setattr(bot.Bot, "__call__", fake_call)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", fake_polling)
    await bot.run_bot(settings)
    assert len(replies) == 1
    assert replies[0].chat_id == 98765
    assert replies[0].text.endswith("98765")
    assert "not the bot ID" in caplog.text
    assert "testing-only" not in caplog.text
