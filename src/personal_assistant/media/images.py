"""Validate and normalize a single uploaded image before model access."""

import base64
import hashlib
import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_EDGE = 1280


class ImageInputError(ValueError):
    pass


class VisionUnavailable(ImageInputError):
    pass


@dataclass(frozen=True)
class PreparedImage:
    jpeg: bytes
    digest: str
    width: int
    height: int
    resized: bool

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii")


def prepare_image(raw: bytes) -> PreparedImage:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ImageInputError("Image limit: one JPEG, PNG or WebP file up to 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if (
                    image.format not in {"JPEG", "PNG", "WEBP"}
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ImageInputError("Only static JPEG, PNG and WebP images are supported")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageInputError("Image exceeds the 20 megapixel limit")
                image.load()
                oriented = ImageOps.exif_transpose(image)
                resized = max(oriented.size) > MAX_IMAGE_EDGE
                oriented.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                # Re-encoding strips metadata; flatten transparency on white for legibility.
                rgba = oriented.convert("RGBA")
                normalized = Image.new("RGB", rgba.size, "white")
                normalized.paste(rgba, mask=rgba.getchannel("A"))
                output = BytesIO()
                normalized.save(output, "JPEG", quality=90)
                return PreparedImage(
                    output.getvalue(),
                    hashlib.sha256(raw).hexdigest(),
                    normalized.width,
                    normalized.height,
                    resized,
                )
    except ImageInputError:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ImageInputError("Image is invalid, damaged, or too large to decode") from None
