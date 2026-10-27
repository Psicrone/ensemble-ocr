"""Ingest: content hash, MIME sniffing and page rendering."""

from __future__ import annotations

import hashlib
import io
import logging

from .types import PageContext

log = logging.getLogger(__name__)

_MAGIC = (
    (b"%PDF", "application/pdf"),
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
)


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sniff_mime(content: bytes) -> str:
    for magic, mime in _MAGIC:
        if content.startswith(magic):
            return mime
    raise ValueError("unsupported document type (expected PDF or an image)")


def render_pages(content: bytes, mime_type: str, dpi: int = 300) -> list[bytes]:
    """PNG per page. Needs pypdfium2 for PDFs and Pillow for images; returns [] if missing."""
    try:
        from PIL import Image
    except ImportError:
        log.info("Pillow not installed; skipping page rendering")
        return []
    pngs: list[bytes] = []
    if mime_type == "application/pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError:
            log.info("pypdfium2 not installed; the model will get the PDF itself")
            return []
        pdf = pdfium.PdfDocument(content)
        for page in pdf:
            image = page.render(scale=dpi / 72).to_pil()
            pngs.append(_png(image))
        return pngs
    image = Image.open(io.BytesIO(content))
    for frame in range(getattr(image, "n_frames", 1)):
        image.seek(frame)
        pngs.append(_png(image.convert("RGB")))
    return pngs


def _png(image: object) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")  # type: ignore[attr-defined]
    return buf.getvalue()


def ingest(
    content: bytes, mime_type: str | None = None, *, render: bool = True, dpi: int = 300
) -> tuple[str, PageContext]:
    mime_type = mime_type or sniff_mime(content)
    pages = PageContext(content=content, mime_type=mime_type)
    if render:
        for i, png in enumerate(render_pages(content, mime_type, dpi)):
            page = pages.page(i)
            page.image = png
    return content_hash(content), pages
