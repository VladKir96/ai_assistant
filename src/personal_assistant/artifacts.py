"""Private, immutable PNG artifacts; no model-provided paths are used for downloads."""

import hashlib
import io
import os
import re
import stat
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from personal_assistant.schemas import Artifact

MAX_ARTIFACT_BYTES = 3_000_000


class ArtifactStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, filename: str, raw: bytes) -> Artifact:
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("Chart exceeds the 3 MB limit")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.width * image.height > 8_000_000:
                    raise ValueError("Chart must be a PNG with at most eight million pixels")
                image.load()
                image = image.convert("RGB")
                image.thumbnail((2000, 2000))
                image.info.clear()
                output = io.BytesIO()
                image.save(output, format="PNG")
                clean = output.getvalue()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("Invalid chart PNG") from None
        if len(clean) > MAX_ARTIFACT_BYTES:
            raise ValueError("Chart exceeds the 3 MB limit")
        identifier = hashlib.sha256(clean).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / f"{identifier}.png"
        descriptor, temporary = tempfile.mkstemp(prefix=".chart-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(clean)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if self.read(identifier) != clean:
                    raise ValueError("Conflicting chart storage") from None
        finally:
            Path(temporary).unlink(missing_ok=True)
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        name = "".join(char for char in name if char.isprintable())[:90]
        return Artifact(id=identifier, filename=name if name.endswith(".png") else "chart.png")

    def read(self, identifier: str) -> bytes:
        if not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise FileNotFoundError("Unknown artifact")
        descriptor = os.open(self.directory / f"{identifier}.png", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARTIFACT_BYTES:
                raise ValueError("Invalid stored artifact")
            data = stream.read(MAX_ARTIFACT_BYTES + 1)
            if len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("Invalid stored artifact size")
            return data
