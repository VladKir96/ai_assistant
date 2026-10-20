import asyncio
import logging
from io import BytesIO
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message
from aiogram.utils.token import TokenValidationError, validate_token

from personal_assistant.artifacts import MAX_ARTIFACT_BYTES
from personal_assistant.config import Settings
from personal_assistant.images import MAX_IMAGE_BYTES
from personal_assistant.pdf_uploads import MAX_PDF_UPLOAD_BYTES
from personal_assistant.schemas import Artifact

logger = logging.getLogger(__name__)


class BotConfigurationError(ValueError):
    pass


def permitted(message: Message, owner_id: int) -> bool:
    return bool(
        owner_id
        and message.from_user
        and message.from_user.id == owner_id
        and message.chat.type == "private"
        and message.chat.id == owner_id
    )


def split_text(text: str, limit: int = 3500) -> list[str]:
    # Telegram measures UTF-16 units. This is safe even for non-BMP emoji.
    chunks, current, units = [], [], 0
    for char in text:
        size = len(char.encode("utf-16-le")) // 2
        if units + size > limit:
            chunks.append("".join(current))
            current, units = [], 0
        current.append(char)
        units += size
    if current:
        chunks.append("".join(current))
    return chunks


async def run_bot(settings: Settings):
    token = settings.telegram_token.get_secret_value()
    if not token:
        raise BotConfigurationError("Set ASSISTANT_TELEGRAM_TOKEN in .env before starting the bot.")
    try:
        validate_token(token)
    except TokenValidationError:
        raise BotConfigurationError(
            "ASSISTANT_TELEGRAM_TOKEN has an invalid format. Copy the complete bot token "
            "from BotFather into .env, including its numeric prefix and ':' separator, "
            "with no spaces. Use the token itself, not the bot username or a URL."
        ) from None
    if settings.telegram_owner_id < 0:
        raise BotConfigurationError(
            "Set ASSISTANT_TELEGRAM_OWNER_ID to your positive numeric Telegram user ID, "
            "or 0 to discover it with /id."
        )
    if not settings.telegram_owner_id or settings.telegram_owner_id == int(token.partition(":")[0]):
        logger.warning(
            "ASSISTANT_TELEGRAM_OWNER_ID must be your personal user ID, not the bot ID. "
            "Send /id in a private chat with this bot, set the returned ID in .env, "
            "then restart the bot. Assistant commands remain restricted."
        )
    bot = Bot(token)
    dispatcher = Dispatcher()
    thread_id = f"tg:{settings.telegram_owner_id}"
    headers = {"Authorization": f"Bearer {settings.api_token.get_secret_value()}"}
    # The API may make several serialized model calls per request.
    async with httpx.AsyncClient(timeout=1200, headers=headers) as client:

        @dispatcher.message(Command("id"))
        async def identity_message(message: Message):
            # Bootstrap command exposes only the requesting user's own ID, never stored data.
            if (
                message.chat.type == "private"
                and message.from_user
                and not message.from_user.is_bot
                and message.chat.id == message.from_user.id
            ):
                await message.answer(
                    f"Your Telegram user ID / Ваш Telegram ID: {message.from_user.id}"
                )

        @dispatcher.message(Command("start", "help"))
        async def help_message(message: Message):
            if permitted(message, settings.telegram_owner_id):
                await message.answer(
                    "Your personal assistant is ready. /new starts a conversation; "
                    "/notes lists notes; /tasks lists tasks; /export downloads notes; "
                    "/agenda lists the next seven days; /calendar downloads the calendar; "
                    "/time shows the current date, time, and timezone.\n"
                    "Destructive commands require an exact ID in a separate message: "
                    "/delete_note <ID>, /forget <ID>, /cancel_event <ID>.\n"
                    "Send a photo or JPEG/PNG/WebP file with a caption "
                    "when a vision model is configured.\n"
                    "Send a PDF as a file with a question when PDF reading is enabled.\n"
                    "Пишите по-русски или по-английски. Для сохранения факта скажите «запомни»."
                )

        @dispatcher.message(Command("new"))
        async def new_thread(message: Message):
            nonlocal thread_id
            if permitted(message, settings.telegram_owner_id):
                thread_id = f"tg:{settings.telegram_owner_id}:{uuid4().hex[:12]}"
                await message.answer(
                    "New conversation / Новый разговор. Notes and facts are retained."
                )

        @dispatcher.message(Command("time"))
        async def current_time(message: Message):
            if not permitted(message, settings.telegram_owner_id):
                return
            try:
                response = await client.get(f"{settings.api_url}/v1/time")
                response.raise_for_status()
                clock = response.json()
                await message.answer(
                    f"{clock['current_time']}\n{clock['timezone']} · {clock['current_weekday']}"
                )
            except httpx.HTTPError:
                await message.answer("Service unavailable / Сервис недоступен.")

        @dispatcher.message(Command("notes", "tasks", "export", "agenda", "calendar"))
        async def stored_data(message: Message):
            if not permitted(message, settings.telegram_owner_id):
                return
            command = message.text.split()[0].split("@")[0].lstrip("/")
            endpoint = {"agenda": "events", "calendar": "calendar.ics"}.get(command, command)
            try:
                response = await client.get(f"{settings.api_url}/v1/{endpoint}")
                response.raise_for_status()
                if command in {"export", "calendar"}:
                    await message.answer_document(
                        BufferedInputFile(
                            response.content,
                            filename="calendar.ics"
                            if command == "calendar"
                            else "assistant-notes.md",
                        )
                    )
                else:
                    items = response.json()
                    text = (
                        "\n\n".join(
                            f"{item['title']}\n"
                            + (
                                f"{item['local_start']} → {item['local_end']} "
                                f"({item['timezone']})"
                                + (" · all-day / весь день" if item["all_day"] else "")
                                + "\n"
                                if command == "agenda"
                                else ""
                            )
                            + f"ID: {item['id']}"
                            for item in items
                        )
                        or "No items / Пока пусто."
                    )
                    for part in split_text(text):
                        await message.answer(part)
            except httpx.HTTPError:
                await message.answer("Service unavailable / Сервис недоступен.")

        async def respond(
            message: Message, text: str, image: bytes | None = None, *, pdf: bytes | None = None
        ):
            request_id = str(
                uuid5(NAMESPACE_URL, f"telegram:{message.chat.id}:{message.message_id}")
            )
            await bot.send_chat_action(message.chat.id, "typing")
            try:
                data = {"thread_id": thread_id, "request_id": request_id, "text": text}
                if pdf is not None:
                    response = await client.post(
                        f"{settings.api_url}/v1/chat/pdf",
                        data=data,
                        files={
                            "pdf": (
                                message.document.file_name or "document.pdf",
                                pdf,
                                "application/pdf",
                            )
                        },
                    )
                elif image is None:
                    response = await client.post(f"{settings.api_url}/v1/chat", json=data)
                else:
                    response = await client.post(
                        f"{settings.api_url}/v1/chat/image",
                        data=data,
                        files={"image": ("image", image, "application/octet-stream")},
                    )
                response.raise_for_status()
                result = response.json()
                answer = result["text"]
                for part in split_text(answer):
                    await message.answer(part)
                for entry in result.get("artifacts", [])[:3]:
                    try:
                        artifact = Artifact.model_validate(entry)
                        raw = bytearray()
                        async with client.stream(
                            "GET", f"{settings.api_url}/v1/artifacts/{artifact.id}"
                        ) as download:
                            download.raise_for_status()
                            async for chunk in download.aiter_bytes():
                                if len(raw) + len(chunk) > MAX_ARTIFACT_BYTES:
                                    raise ValueError("Chart exceeds the download limit")
                                raw.extend(chunk)
                        upload = BufferedInputFile(bytes(raw), filename=artifact.filename)
                        try:
                            await message.answer_photo(upload)
                        except TelegramAPIError:
                            await message.answer_document(upload)
                    except (httpx.HTTPError, ValueError, TelegramAPIError):
                        await message.answer("Chart unavailable / График недоступен.")
                if settings.voice_enabled and settings.voice_output_enabled and message.voice:
                    from personal_assistant.voice import SpeechClient

                    try:
                        audio = await SpeechClient(settings).speak(answer[:1500], text)
                        if len(answer) > 1500:
                            await message.answer(
                                "Audio contains the first part / Озвучено начало ответа."
                            )
                        await message.answer_voice(BufferedInputFile(audio, filename="answer.ogg"))
                    except (httpx.HTTPError, ValueError):
                        await message.answer("Voice output unavailable / Озвучка недоступна.")
            except httpx.HTTPStatusError as error:
                try:
                    detail = error.response.json().get("detail", "Request failed")
                except (ValueError, AttributeError):
                    detail = "Request failed / Запрос не выполнен."
                await message.answer(str(detail)[:1500])
            except httpx.HTTPError:
                await message.answer("Service unavailable / Сервис недоступен.")

        @dispatcher.message(
            (F.document.mime_type == "application/pdf")
            | F.document.file_name.lower().endswith(".pdf")
        )
        async def pdf_message(message: Message):
            if not permitted(message, settings.telegram_owner_id):
                return
            if not settings.pdf_enabled:
                await message.answer("PDF reading is disabled / Чтение PDF отключено.")
                return
            if message.media_group_id:
                await message.answer("Send one PDF at a time / Отправьте один PDF за раз.")
                return
            if (message.document.file_size or 0) > MAX_PDF_UPLOAD_BYTES:
                await message.answer("PDF upload limit: 10 MB / Лимит PDF: 10 МБ.")
                return
            upload = BoundedUploadBuffer(MAX_PDF_UPLOAD_BYTES)
            try:
                await bot.download(message.document, destination=upload)
                default = (
                    "Кратко опиши PDF по первым трём страницам."
                    if (message.from_user.language_code or "").startswith("ru")
                    else "Briefly summarize the first three pages of this PDF."
                )
                await respond(message, message.caption or default, pdf=upload.getvalue())
            except (ValueError, OSError, TelegramAPIError):
                await message.answer("PDF download failed or exceeds 10 MB / Ошибка загрузки PDF.")

        @dispatcher.message(F.photo | F.document.mime_type.startswith("image/"))
        async def image_message(message: Message):
            if not permitted(message, settings.telegram_owner_id):
                return
            if message.media_group_id:
                await message.answer(
                    "Send one image per message / Отправьте одно изображение за раз."
                )
                return
            if not settings.model_supports_images:
                await message.answer(
                    "Image input is disabled for this model. Configure a vision model first. / "
                    "Для работы с изображениями сначала настройте модель с поддержкой зрения."
                )
                return
            if message.document and message.document.mime_type not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                await message.answer("Use JPEG, PNG or WebP / Используйте JPEG, PNG или WebP.")
                return
            media = message.photo[-1] if message.photo else message.document
            if (media.file_size or 0) > MAX_IMAGE_BYTES:
                await message.answer("Image limit: 10 MB / Лимит изображения: 10 МБ.")
                return
            image = BoundedUploadBuffer(MAX_IMAGE_BYTES)
            try:
                await bot.download(media, destination=image)
                default = (
                    "Опиши изображение."
                    if (message.from_user.language_code or "").startswith("ru")
                    else "Describe this image."
                )
                await respond(message, message.caption or default, image.getvalue())
            except (ValueError, OSError, TelegramAPIError):
                await message.answer(
                    "Image download failed or exceeds 10 MB / Ошибка загрузки изображения."
                )

        @dispatcher.message(F.voice)
        async def voice_message(message: Message):
            if not permitted(message, settings.telegram_owner_id):
                return
            if not settings.voice_enabled:
                await message.answer("Voice is disabled / Голосовой ввод отключён.")
                return
            if message.voice.duration > 180 or (message.voice.file_size or 0) > 10_000_000:
                await message.answer("Voice limit: 3 minutes, 10 MB / Лимит: 3 минуты, 10 МБ.")
                return
            from personal_assistant.voice import SpeechClient

            audio = BytesIO()
            try:
                await bot.download(message.voice, destination=audio)
                text = await SpeechClient(settings).transcribe(audio.getvalue())
                await respond(message, text)
            except (httpx.HTTPError, ValueError):
                await message.answer("Transcription unavailable / Распознавание недоступно.")

        @dispatcher.message(F.text)
        async def text_message(message: Message):
            if permitted(message, settings.telegram_owner_id):
                await respond(message, message.text)

        try:
            await dispatcher.start_polling(bot, handle_as_tasks=False)
        finally:
            await bot.session.close()


def main():
    try:
        asyncio.run(run_bot(Settings()))
    except BotConfigurationError as error:
        raise SystemExit(str(error)) from None


class BoundedUploadBuffer(BytesIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError("Upload exceeds the download limit")
        return super().write(data)
