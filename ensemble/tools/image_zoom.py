"""``image.zoom``: crop and upscale a page region so the agent can read small print."""

from __future__ import annotations

import io

from pydantic import BaseModel, Field

from ..tool import RunContext, register


class ZoomIn(BaseModel):
    page: int = Field(description="0-based page index")
    x0: float = Field(description="left edge, 0..1")
    y0: float = Field(description="top edge, 0..1")
    x1: float = Field(description="right edge, 0..1")
    y1: float = Field(description="bottom edge, 0..1")
    scale: float = Field(2.0, description="upscale factor, 1-4")


class ZoomOut(BaseModel):
    found: bool
    # Raw PNG bytes. The ADK bridge turns this into an image part the model sees.
    image_png: bytes | None = None
    note: str | None = None


@register
class ImageZoom:
    """Crop and upscale a region of a page image for small or faint print. Returns the image."""

    name = "image.zoom"
    version = "1"
    kind = "lookup"
    Input = ZoomIn
    Output = ZoomOut
    llm_callable = True
    fields: list[str] = []

    def __init__(self, max_side: int = 2048) -> None:
        self.max_side = max_side

    async def run(self, ctx: RunContext, inp: ZoomIn) -> ZoomOut:
        if inp.page >= len(ctx.pages.pages) or ctx.pages.pages[inp.page].image is None:
            return ZoomOut(found=False, note="No rendered image for that page.")
        from PIL import Image

        image = Image.open(io.BytesIO(ctx.pages.pages[inp.page].image))
        w, h = image.size
        box = (int(inp.x0 * w), int(inp.y0 * h), int(inp.x1 * w), int(inp.y1 * h))
        if box[2] <= box[0] or box[3] <= box[1]:
            return ZoomOut(found=False, note="Empty region.")
        crop = image.crop(box)
        scale = max(1.0, min(4.0, inp.scale, self.max_side / max(crop.size)))
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return ZoomOut(found=True, image_png=buf.getvalue())
