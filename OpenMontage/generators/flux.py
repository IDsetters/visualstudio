"""FLUX image generation through OpenMontage's existing fal.ai adapter."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tools.graphics.bfl_image import BFLImage
from tools.graphics.flux_image import FluxImage


def _generate_flux(prompt: str, destination: Path) -> Path:
    if os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY"):
        model = os.environ.get("FLUX_IMAGE_MODEL", "flux-pro/v1.1").strip()
        result = FluxImage().execute(
            {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "model": model,
                "output_path": str(destination),
            }
        )
    elif os.environ.get("BFL_API_KEY"):
        result = BFLImage().execute(
            {
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "resolution": "1k",
                "grounding": False,
                "output_path": str(destination),
            }
        )
    else:
        raise RuntimeError("Configure FAL_KEY or BFL_API_KEY to generate FLUX images")
    if not result.success:
        raise RuntimeError(f"FLUX generation failed: {result.error}")

    output = Path(result.data.get("output", destination))
    if not output.is_file():
        raise RuntimeError(f"FLUX reported success but did not create {output}")
    return output


async def gerar_flux(prompt: str, destination: Path) -> Path:
    """Generate one image without blocking the async curation pipeline."""
    return await asyncio.to_thread(_generate_flux, prompt, destination)
