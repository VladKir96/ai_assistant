"""Bounded Telegram speech delivery, separate from assistant reasoning and text output."""

import asyncio
import logging
from contextlib import suppress

import httpx
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, Message, ReplyParameters

from personal_assistant.core.config import Settings
from personal_assistant.media.tts import SpokenText, prepare_speech
from personal_assistant.media.voice import SpeechClient

logger = logging.getLogger(__name__)


class AudioDelivery:
    def __init__(self, settings: Settings):
        self.speech = SpeechClient(settings)
        self.queue: asyncio.Queue[tuple[Message, SpokenText]] = asyncio.Queue(maxsize=3)
        self.worker: asyncio.Task | None = None

    async def submit(self, message: Message, answer: str):
        spoken = prepare_speech(answer)
        if not spoken.text:
            await message.answer("No readable prose for audio; the full text is above.")
            return
        try:
            self.queue.put_nowait((message, spoken))
        except asyncio.QueueFull:
            await message.answer("Audio queue is full; the text response is available above.")
            return
        if self.worker is None:
            self.worker = asyncio.create_task(self.deliver(), name="telegram-audio")

    async def deliver(self):
        while True:
            message, spoken = await self.queue.get()
            try:
                audio = await self.speech.speak(spoken.text)
                await message.answer_voice(
                    BufferedInputFile(audio, filename="answer.ogg"),
                    caption="Audio contains the first part of the answer."
                    if spoken.truncated
                    else None,
                    reply_parameters=ReplyParameters(message_id=message.message_id),
                )
            except (httpx.HTTPError, ValueError, OSError, TelegramAPIError) as error:
                # Avoid logging speech content, URLs, credentials, or private paths.
                logger.warning("audio_delivery_failed type=%s", type(error).__name__)
                with suppress(TelegramAPIError):
                    await message.answer("Voice output unavailable; the text response is above.")
            finally:
                self.queue.task_done()

    async def cancel(self):
        """Cancel the current HTTP request and discard queued audio on /audio off or exit."""
        if self.worker is not None:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker
            self.worker = None
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
