"""Local image curation, mock generation, WebP conversion, and Notion publishing."""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import io
import json
import math
import os
import re
import textwrap
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Awaitable, Callable, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageOps
from pydantic import AliasChoices, BaseModel, Field

from generators.flux import gerar_flux
from generators.kittl_automation import gerar_kittl
from generators.midjourney_web import gerar_midjourney as gerar_midjourney_web
from lib.env_loader import load_env

TARGET_COUNT = 16
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LLM_BATCH_SIZE = 16
LLM_FINAL_BATCH_SIZE = 32
NOTION_SINGLE_UPLOAD_LIMIT = 20 * 1024 * 1024
CHECKPOINT_FILENAME = ".curadoria_status.json"
DEFAULT_CRITERION = (
    "Selecione imagens visualmente fortes, nítidas, coerentes entre si e "
    "adequadas para uma coleção editorial. Evite duplicatas, imagens borradas, "
    "com composição confusa ou marcas d'água."
)
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = REPO_ROOT / "assets_teste"
MIDJOURNEY_ASPECT_RATIOS = {
    "1:1",
    "4:3",
    "3:2",
    "16:9",
    "21:9",
    "9:16",
    "2:3",
    "3:4",
    "4:5",
    "5:4",
}


@dataclass(frozen=True)
class ImageAsset:
    asset_id: str
    path: Path


class EagleUnavailableError(RuntimeError):
    """Raised when Eagle is unavailable or cannot serve the requested folder."""


def _validate_eagle_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.port != 41595
    ):
        raise ValueError("EAGLE_API_BASE_URL must use local HTTP on port 41595")
    return base_url.rstrip("/")


async def _eagle_data(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, str | int] | None = None,
) -> object:
    response = await client.get(endpoint, params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise EagleUnavailableError(f"Eagle returned an unsuccessful response for {endpoint}")
    return payload.get("data")


def _find_folder_id(folders: list[dict], folder_name: str) -> str | None:
    for folder in folders:
        if folder.get("name") == folder_name:
            return folder.get("id")
        children = folder.get("children", [])
        if isinstance(children, list):
            found = _find_folder_id(children, folder_name)
            if found:
                return found
    return None


async def _read_eagle_folder(folder_name: str) -> list[Path]:
    base_url = _validate_eagle_url(
        os.environ.get("EAGLE_API_BASE_URL", "http://127.0.0.1:41595")
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=4.0) as client:
        folders = await _eagle_data(client, "/api/folder/list")
        if not isinstance(folders, list):
            raise EagleUnavailableError("Eagle returned an invalid folder list")
        folder_id = _find_folder_id(folders, folder_name)
        if not folder_id:
            raise EagleUnavailableError(f"Eagle folder {folder_name!r} was not found")

        paths: list[Path] = []
        offset = 0
        page_size = 200
        while True:
            items = await _eagle_data(
                client,
                "/api/item/list",
                {"folders": folder_id, "limit": page_size, "offset": offset},
            )
            if not isinstance(items, list):
                raise EagleUnavailableError("Eagle returned an invalid item list")
            for item in items:
                if not isinstance(item, dict):
                    continue
                extension = f".{str(item.get('ext', '')).lower().lstrip('.')}"
                if extension not in SUPPORTED_EXTENSIONS:
                    continue
                item_id = item.get("id")
                if not item_id:
                    continue
                thumbnail = await _eagle_data(
                    client, "/api/item/thumbnail", {"id": str(item_id)}
                )
                if isinstance(thumbnail, str) and Path(thumbnail).is_file():
                    paths.append(Path(thumbnail))

            if len(items) < page_size:
                break
            offset += len(items)
        return paths


def _read_local_folder(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Local image folder does not exist: {folder}")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


async def ler_imagens_eagle(
    folder_name: str,
    local_dir: Path,
    recursive_fallback: bool = True,
) -> list[Path]:
    """Read Eagle's local folder, falling back to assets_teste/ if unavailable."""
    try:
        eagle_paths = await _read_eagle_folder(folder_name)
        if eagle_paths:
            print(f"Eagle: {len(eagle_paths)} imagens encontradas em {folder_name!r}.")
            return eagle_paths
        print(f"Eagle: nenhuma imagem encontrada; usando {local_dir}.")
    except (httpx.HTTPError, EagleUnavailableError) as exc:
        print(f"Eagle indisponível ({exc}); usando a pasta local {local_dir}.")

    local_paths = _read_local_folder(local_dir, recursive=recursive_fallback)
    if not local_paths:
        raise FileNotFoundError(f"Nenhuma imagem JPG, PNG ou WebP encontrada em {local_dir}")
    print(f"Fallback local: {len(local_paths)} imagens encontradas.")
    return local_paths


def _thumbnail_as_base64(path: Path) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((512, 512))
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _llm_select_batch(
    client: AsyncOpenAI,
    assets: list[ImageAsset],
    criterion: str,
    max_selected: int,
    model: str,
) -> list[ImageAsset]:
    content: list[dict] = []
    for asset in assets:
        image_data = await asyncio.to_thread(_thumbnail_as_base64, asset.path)
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "low",
                    },
                },
                {"type": "text", "text": f"Imagem {asset.asset_id}: {asset.path.name}"},
            ]
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Avalie as imagens acima segundo este critério:\n"
                f"{criterion}\n\n"
                f"Escolha no máximo {max_selected} imagens. Responda somente com "
                'JSON válido neste formato: {"selected": ["img_0001"]}. Use '
                "apenas os IDs apresentados e mantenha a ordem de preferência. "
                "Se nenhuma imagem atender ao critério, retorne uma lista vazia."
            ),
        }
    )
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1200,
        messages=[{"role": "user", "content": content}],
    )

    response_text = response.choices[0].message.content
    if not isinstance(response_text, str):
        raise ValueError("The LLM proxy did not return a text selection")
    json_start = response_text.find("{")
    if json_start < 0:
        raise ValueError("The LLM proxy did not return the required JSON selection")
    try:
        result, _ = json.JSONDecoder().raw_decode(response_text[json_start:])
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM proxy returned invalid JSON for its image selection") from exc

    selected_ids = result.get("selected") if isinstance(result, dict) else None
    if not isinstance(selected_ids, list) or not all(
        isinstance(asset_id, str) for asset_id in selected_ids
    ):
        raise ValueError("LLM response must contain a string array named 'selected'")

    available = {asset.asset_id: asset for asset in assets}
    if any(asset_id not in available for asset_id in selected_ids):
        raise ValueError("LLM selected an image ID that was not sent in this request")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("LLM returned duplicate image IDs")
    if len(selected_ids) > max_selected:
        raise ValueError("LLM selected more images than requested")
    return [available[asset_id] for asset_id in selected_ids]


async def selecionar_imagens(
    paths: list[Path],
    criterio: str,
    target_count: int = TARGET_COUNT,
) -> list[ImageAsset]:
    """Use a local OpenAI-compatible vision model to curate images."""
    if not paths:
        return []
    if not criterio.strip():
        raise ValueError("The curation criterion must not be empty")
    api_key, base_url, model = _local_llm_settings()

    assets = [
        ImageAsset(asset_id=f"img_{index:04d}", path=path)
        for index, path in enumerate(paths, start=1)
    ]
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        candidates = assets
        while len(candidates) > LLM_FINAL_BATCH_SIZE:
            next_candidates: list[ImageAsset] = []
            total = len(candidates)
            for start in range(0, total, LLM_BATCH_SIZE):
                batch = candidates[start : start + LLM_BATCH_SIZE]
                quota = min(
                    len(batch),
                    max(1, math.ceil(target_count * len(batch) / total)),
                )
                next_candidates.extend(
                    await _llm_select_batch(client, batch, criterio, quota, model)
                )
            if len(next_candidates) >= len(candidates):
                raise RuntimeError("LLM curation did not reduce the large image set")
            candidates = next_candidates

        return await _llm_select_batch(
            client,
            candidates,
            criterio,
            min(target_count, len(candidates)),
            model,
        )


def _local_llm_settings() -> tuple[str, str, str]:
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")
    api_key = os.environ.get("LLM_API_KEY", "").strip() or "sk-dummy-key"
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError("Set LLM_MODEL to a model served by the local proxy")
    return api_key, base_url, model


async def expandir_direcao_visual_midjourney(
    direcao_visual: str,
    quantidade: int = 1,
) -> list[str]:
    """Ask the local LLM for validated Midjourney prompt variants."""
    if not direcao_visual.strip():
        raise ValueError("The visual direction must not be empty")
    if not 1 <= quantidade <= TARGET_COUNT:
        raise ValueError(f"Midjourney prompt count must be between 1 and {TARGET_COUNT}")

    api_key, base_url, model = _local_llm_settings()
    request_text = (
        f"Transforme esta direção de arte em exatamente {quantidade} prompts "
        "distintos para geração no Midjourney. Priorize técnicas visuais, "
        "elementos e princípios de design que comuniquem a intenção; não reduza "
        "a proposta a um rótulo de estilo. Quando houver narrativa, explore "
        "ponto de partida, virada e ideia que permanece. Descreva hierarquia, "
        "ponto de entrada do olhar, ritmo e metáfora visual como escolhas para "
        "o diretor experimentar; não declare efeitos universais sobre emoções "
        "ou respostas neuropsicológicas:\n"
        f"{direcao_visual}\n\n"
        "Responda somente com JSON válido no formato "
        '{"variations":[{"prompt":"descrição visual","sref":null,'
        '"stylize":100,"ar":"3:2"}]}. '
        "Cada prompt deve descrever conteúdo "
        "visual específico, sem incluir parâmetros --. Use sref null se a "
        "direção não fornecer uma referência de estilo conhecida; não invente "
        "códigos. Caso use sref, informe um código numérico ou URL HTTPS. "
        "stylize deve ser um inteiro de 0 a 1000. ar deve ser um destes valores: "
        + ", ".join(sorted(MIDJOURNEY_ASPECT_RATIOS))
        + "."
    )
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max(1200, quantidade * 180),
            messages=[{"role": "user", "content": request_text}],
        )

    response_text = response.choices[0].message.content
    if not isinstance(response_text, str):
        raise ValueError("The local LLM did not return Midjourney prompt text")
    json_start = response_text.find("{")
    if json_start < 0:
        raise ValueError("The local LLM did not return JSON prompt variations")
    try:
        result, _ = json.JSONDecoder().raw_decode(response_text[json_start:])
    except json.JSONDecodeError as exc:
        raise ValueError("The local LLM returned invalid Midjourney prompt JSON") from exc

    variations = result.get("variations") if isinstance(result, dict) else None
    if not isinstance(variations, list) or len(variations) != quantidade:
        raise ValueError(f"The local LLM must return exactly {quantidade} variations")

    prompts = []
    for variation in variations:
        if not isinstance(variation, dict):
            raise ValueError("Each Midjourney variation must be a JSON object")
        prompt = variation.get("prompt")
        sref = variation.get("sref")
        stylize = variation.get("stylize")
        aspect_ratio = variation.get("ar")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each Midjourney variation needs a non-empty prompt")
        if re.search(r"--[A-Za-z][A-Za-z0-9-]*", prompt):
            raise ValueError("LLM prompt text must not contain raw style parameters")
        if sref is not None:
            if isinstance(sref, int) and not isinstance(sref, bool):
                sref = str(sref)
            if not isinstance(sref, str) or not (
                sref.isdecimal()
                or (
                    urlsplit(sref).scheme == "https"
                    and bool(urlsplit(sref).hostname)
                    and not urlsplit(sref).username
                    and not urlsplit(sref).password
                )
            ):
                raise ValueError("Midjourney sref must be null, numeric, or an HTTPS URL")
        if type(stylize) is not int or not 0 <= stylize <= 1000:
            raise ValueError("Midjourney stylize must be an integer from 0 to 1000")
        if (
            not isinstance(aspect_ratio, str)
            or aspect_ratio not in MIDJOURNEY_ASPECT_RATIOS
        ):
            raise ValueError("LLM returned an unsupported Midjourney aspect ratio")

        parameters = []
        if sref is not None:
            parameters.append(f"--sref {sref}")
        parameters.extend((f"--stylize {stylize}", f"--ar {aspect_ratio}"))
        prompts.append(f"{prompt.strip()} {' '.join(parameters)}")
    return prompts


async def expandir_estetica_midjourney(
    estetica: str,
    quantidade: int = 1,
) -> list[str]:
    """Compatibility wrapper for callers using the original aesthetic name."""
    return await expandir_direcao_visual_midjourney(estetica, quantidade)


def _write_mock_image(provider: str, prompt: str, destination: Path) -> Path:
    digest = hashlib.sha256(f"{provider}:{prompt}".encode("utf-8")).digest()
    background = tuple(40 + value % 150 for value in digest[:3])
    image = Image.new("RGB", (1024, 1024), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 976, 976), outline="white", width=4)
    label = f"MOCK - {provider}\n\n" + textwrap.fill(prompt, width=38)
    draw.multiline_text((88, 88), label, fill="white", spacing=18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination


async def _gerar_mock(provider: str, prompt: str, destination: Path) -> Path:
    print(f"[MOCK {provider}] Simulando geração: {prompt[:100]}")
    return await asyncio.to_thread(_write_mock_image, provider, prompt, destination)


async def gerar_higgsfield(prompt: str, destination: Path) -> Path:
    return await _gerar_mock("Higgsfield", prompt, destination)


async def gerar_imagens(
    deficit: int,
    criterio: str,
    output_dir: Path,
    reference_image: Path | None = None,
    reference_url: str | None = None,
) -> list[Path]:
    """Fill missing slots using the configured generator or local mocks."""
    if deficit < 0:
        raise ValueError("Image deficit cannot be negative")
    if deficit == 0:
        return []

    provider = os.environ.get("IMAGE_GENERATION_PROVIDER", "auto").strip().lower()
    if provider == "auto":
        provider = (
            "flux"
            if (
                os.environ.get("FAL_KEY")
                or os.environ.get("FAL_AI_API_KEY")
                or os.environ.get("BFL_API_KEY")
            )
            else "mock"
        )

    if provider == "midjourney":
        configured_reference = os.environ.get(
            "MIDJOURNEY_REFERENCE_IMAGE_PATH", ""
        ).strip()
        if reference_image is None and configured_reference:
            reference_image = Path(configured_reference).expanduser()
        if reference_image is not None and not reference_image.is_file():
            raise FileNotFoundError(
                f"Midjourney reference image not found: {reference_image}"
            )
        if reference_url is None:
            reference_url = (
                os.environ.get("MIDJOURNEY_REFERENCE_IMAGE_URL", "").strip() or None
            )
        if reference_url:
            _validate_midjourney_reference_url(reference_url)
        prompts = await expandir_direcao_visual_midjourney(criterio, deficit)
    else:
        prompts = [
            (
                f"Imagem {index + 1} de {deficit}. "
                f"Use this curation criterion as the creative direction: {criterio}"
            )
            for index in range(deficit)
        ]

    jobs = []
    for index in range(deficit):
        prompt = prompts[index]
        destination = output_dir / f"{provider}_{index + 1:02d}.png"
        if provider == "flux":
            jobs.append(gerar_flux(prompt, destination))
        elif provider == "kittl":
            jobs.append(gerar_kittl(prompt, destination))
        elif provider == "midjourney":
            jobs.append(
                gerar_midjourney_web(
                    prompt,
                    destination,
                    reference_image=reference_image,
                    reference_url=reference_url,
                )
            )
        elif provider == "mock":
            mock_providers = ("Higgsfield", "Kittl", "Midjourney")
            mock_provider = mock_providers[index % len(mock_providers)]
            jobs.append(_gerar_mock(mock_provider, prompt, destination))
        else:
            raise ValueError(
                "IMAGE_GENERATION_PROVIDER must be auto, flux, kittl, mock, "
                "or midjourney"
            )
    return list(await asyncio.gather(*jobs))


def _convert_one_to_webp(source_path: Path, destination: Path) -> Path:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((1600, 1600))
        if "A" in image.getbands() or "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="WEBP", quality=84, method=6)
    return destination


async def converter_webp(paths: list[Path], output_dir: Path) -> list[Path]:
    """Convert each selected/generated image to a size-limited WebP file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, path in enumerate(paths, start=1):
        clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
        destination = output_dir / f"{index:02d}_{clean_stem or 'image'}.webp"
        jobs.append(asyncio.to_thread(_convert_one_to_webp, path, destination))
    return list(await asyncio.gather(*jobs))


async def publicar_notion(
    webp_paths: list[Path],
    title: str,
    database_id: str = "",
    parent_page_id: str = "",
) -> str:
    """Upload WebP files to Notion and publish them as image blocks on a page."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    database_id = database_id.strip()
    parent_page_id = parent_page_id.strip()
    if not token:
        raise RuntimeError("NOTION_TOKEN is required to publish to Notion")
    if not database_id and not parent_page_id:
        raise RuntimeError("Set NOTION_DATABASE_ID or NOTION_PARENT_PAGE_ID")
    if not webp_paths:
        raise ValueError("Cannot publish an empty image collection")

    title_property = os.environ.get("NOTION_TITLE_PROPERTY", "Name").strip()
    if database_id and not title_property:
        raise ValueError("NOTION_TITLE_PROPERTY cannot be empty for a database")
    api_version = os.environ.get("NOTION_API_VERSION", "2026-03-11").strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": api_version,
        "Accept": "application/json",
    }
    interval = float(os.environ.get("NOTION_MIN_REQUEST_INTERVAL_SECONDS", "0.4"))
    if interval < 0:
        raise ValueError("NOTION_MIN_REQUEST_INTERVAL_SECONDS must be non-negative")
    try:
        max_image_bytes = int(
            os.environ.get("NOTION_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))
        )
    except ValueError as exc:
        raise ValueError("NOTION_MAX_IMAGE_BYTES must be an integer") from exc
    if not 0 < max_image_bytes <= NOTION_SINGLE_UPLOAD_LIMIT:
        raise ValueError("NOTION_MAX_IMAGE_BYTES must be between 1 and 20 MiB")

    loop = asyncio.get_running_loop()
    last_request_at = 0.0
    async with httpx.AsyncClient(base_url="https://api.notion.com/v1", timeout=60.0) as client:

        async def notion_request(method: str, endpoint: str, **kwargs) -> httpx.Response:
            nonlocal last_request_at
            delay = interval - (loop.time() - last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            last_request_at = loop.time()
            response = await client.request(method, endpoint, **kwargs)
            if response.is_error:
                raise RuntimeError(
                    f"Notion request {method} {endpoint} failed with HTTP "
                    f"{response.status_code}: {response.text[:500]}"
                )
            return response

        file_upload_ids: list[str] = []
        for path in webp_paths:
            size = path.stat().st_size
            if size > max_image_bytes:
                raise ValueError(
                    f"{path} is larger than the configured "
                    f"{max_image_bytes / (1024 * 1024):g} MiB upload limit; "
                    "reduce its dimensions or quality before publishing"
                )
            create_response = await notion_request(
                "POST",
                "/file_uploads",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "mode": "single_part",
                    "filename": path.name,
                    "content_type": "image/webp",
                },
            )
            upload_id = create_response.json().get("id")
            if not upload_id:
                raise RuntimeError(f"Notion did not return an upload ID for {path.name}")

            image_bytes = await asyncio.to_thread(path.read_bytes)
            await notion_request(
                "POST",
                f"/file_uploads/{upload_id}/send",
                headers=headers,
                files={"file": (path.name, image_bytes, "image/webp")},
            )
            file_upload_ids.append(upload_id)

        if database_id:
            parent = {"database_id": database_id}
            properties = {
                title_property: {
                    "title": [{"type": "text", "text": {"content": title[:2000]}}]
                }
            }
        else:
            parent = {"page_id": parent_page_id}
            properties = {
                "title": {"title": [{"type": "text", "text": {"content": title[:2000]}}]}
            }

        children = [
            {
                "object": "block",
                "type": "image",
                "image": {
                    "type": "file_upload",
                    "file_upload": {"id": upload_id},
                },
            }
            for upload_id in file_upload_ids
        ]
        page_response = await notion_request(
            "POST",
            "/pages",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "parent": parent,
                "properties": properties,
                "children": children,
            },
        )
    page = page_response.json()
    page_url = page.get("url")
    if not page_url:
        raise RuntimeError("Notion created the page but did not return its URL")
    return page_url


def _safe_path_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "folder"


def _validate_midjourney_reference_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Midjourney reference URLs must be public HTTPS URLs")
    return value


def _discover_local_image_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image input directory does not exist: {root}")
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    image_folders = [
        folder
        for folder in directories
        if any(
            item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
            for item in folder.iterdir()
        )
    ]
    return sorted(image_folders, key=lambda path: path.relative_to(root).as_posix())


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


async def executar_pipeline_pasta(
    folder_name: str,
    local_dir: Path,
    criterio: str,
    output_dir: Path,
    titulo: str,
    database_id: str = "",
    parent_page_id: str = "",
    dry_run: bool = False,
    read_eagle: bool = True,
    recursive_fallback: bool = True,
    output_key: str | None = None,
    reference_image: Path | None = None,
    reference_url: str | None = None,
    target_count: int | None = None,
) -> dict[str, object]:
    if target_count is None:
        target_count = TARGET_COUNT
    checkpoint_path = local_dir / CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        print(f"Pasta já processada; ignorando: {folder_name}")
        return {"status": "skipped", "folder": folder_name}
    _validar_configuracao_pipeline(
        dry_run=dry_run,
        database_id=database_id,
        parent_page_id=parent_page_id,
    )

    if read_eagle:
        paths = await ler_imagens_eagle(
            folder_name,
            local_dir,
            recursive_fallback=recursive_fallback,
        )
    else:
        paths = await asyncio.to_thread(
            _read_local_folder,
            local_dir,
            recursive_fallback,
        )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {local_dir}")

    criterion = criterio.strip()
    selected = await selecionar_imagens(paths, criterion, target_count)
    deficit = target_count - len(selected)
    print(f"LLM local selecionou {len(selected)} imagens; déficit: {deficit}.")

    key = _safe_path_segment(output_key or folder_name)
    run_dir = output_dir / key / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if (
        os.environ.get("IMAGE_GENERATION_PROVIDER", "auto").strip().lower()
        == "midjourney"
        and reference_image is None
        and selected
    ):
        reference_image = selected[0].path
    generated = await gerar_imagens(
        deficit,
        criterion,
        run_dir / "generated",
        reference_image=reference_image,
        reference_url=reference_url,
    )
    final_sources = [asset.path for asset in selected] + generated
    if len(final_sources) != target_count:
        raise RuntimeError(
            f"Expected {target_count} final images, received {len(final_sources)}"
        )
    webp_paths = await converter_webp(final_sources, run_dir / "webp")
    print(f"{len(webp_paths)} arquivos WebP salvos em {run_dir / 'webp'}.")

    if dry_run:
        print("Dry run: publicação no Notion ignorada.")
        return {
            "status": "dry_run",
            "folder": folder_name,
            "webp_count": len(webp_paths),
        }

    page_url = await publicar_notion(
        webp_paths,
        title=titulo,
        database_id=database_id,
        parent_page_id=parent_page_id,
    )
    checkpoint = {
        "status": "completed",
        "processed_at": datetime.now().astimezone().isoformat(),
        "folder": folder_name,
        "notion_url": page_url,
        "selected_count": len(selected),
        "generated_count": len(generated),
        "webp_count": len(webp_paths),
    }
    await asyncio.to_thread(_write_checkpoint, checkpoint_path, checkpoint)
    print(f"Publicado no Notion: {page_url}")
    return checkpoint


ProgressCallback = Callable[[dict[str, object]], Awaitable[None]]


def _validar_configuracao_pipeline(
    dry_run: bool,
    database_id: str = "",
    parent_page_id: str = "",
) -> None:
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError("Set LLM_MODEL to a vision-capable model served by the local proxy")
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1").strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")
    if dry_run:
        return
    if not os.environ.get("NOTION_TOKEN", "").strip():
        raise RuntimeError("NOTION_TOKEN is required to publish to Notion")
    if not (
        database_id.strip()
        or parent_page_id.strip()
        or os.environ.get("NOTION_DATABASE_ID", "").strip()
        or os.environ.get("NOTION_PARENT_PAGE_ID", "").strip()
    ):
        raise RuntimeError("Set NOTION_DATABASE_ID or NOTION_PARENT_PAGE_ID")
    if (
        database_id.strip() or os.environ.get("NOTION_DATABASE_ID", "").strip()
    ) and not os.environ.get("NOTION_TITLE_PROPERTY", "Name").strip():
        raise ValueError("NOTION_TITLE_PROPERTY cannot be empty for a database")


async def processar_pastas_locais(
    root: Path,
    criterio: str,
    output_dir: Path,
    titulo: str,
    database_id: str = "",
    parent_page_id: str = "",
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
    reference_image: Path | None = None,
    reference_url: str | None = None,
    target_count: int | None = None,
) -> dict[str, object]:
    if target_count is None:
        target_count = TARGET_COUNT
    load_env()
    folders = await asyncio.to_thread(_discover_local_image_folders, root)
    if not folders:
        raise FileNotFoundError(f"No image folders found under {root}")
    pending_folders = [
        folder
        for folder in folders
        if not (folder / CHECKPOINT_FILENAME).is_file()
    ]
    if pending_folders:
        _validar_configuracao_pipeline(
            dry_run=dry_run,
            database_id=database_id,
            parent_page_id=parent_page_id,
        )

    summary: dict[str, object] = {
        "total": len(folders),
        "completed": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "current_folder": None,
    }
    if progress_callback:
        await progress_callback(dict(summary))

    for folder in folders:
        folder_label = folder.relative_to(root).as_posix() or folder.name
        checkpoint_path = folder / CHECKPOINT_FILENAME
        if checkpoint_path.is_file():
            summary["skipped"] = int(summary["skipped"]) + 1
            summary["completed"] = int(summary["completed"]) + 1
            summary["current_folder"] = folder_label
            print(f"Pasta já processada; ignorando: {folder_label}")
        else:
            summary["current_folder"] = folder_label
            if progress_callback:
                await progress_callback(dict(summary))
            try:
                result = await executar_pipeline_pasta(
                    folder_name=folder_label,
                    local_dir=folder,
                    criterio=criterio,
                    output_dir=output_dir,
                    titulo=f"{titulo} - {folder_label}",
                    database_id=database_id,
                    parent_page_id=parent_page_id,
                    dry_run=dry_run,
                    read_eagle=False,
                    recursive_fallback=False,
                    output_key=folder_label,
                    reference_image=reference_image,
                    reference_url=reference_url,
                    target_count=target_count,
                )
                if result["status"] == "skipped":
                    summary["skipped"] = int(summary["skipped"]) + 1
                else:
                    summary["processed"] = int(summary["processed"]) + 1
                summary["completed"] = int(summary["completed"]) + 1
            except Exception as exc:
                errors = summary["errors"]
                assert isinstance(errors, list)
                errors.append(
                    {
                        "folder": folder_label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                summary["failed"] = int(summary["failed"]) + 1
                summary["completed"] = int(summary["completed"]) + 1
                print(f"Falha ao processar {folder_label}: {type(exc).__name__}: {exc}")

        if progress_callback:
            await progress_callback(dict(summary))
    summary["current_folder"] = None
    return summary


class CuradoriaRequest(BaseModel):
    criterio: str = Field(default=DEFAULT_CRITERION, min_length=1, max_length=4000)
    reference_url: str | None = Field(default=None, max_length=2048)
    pasta_raiz: str | None = Field(default=None, max_length=4096)
    meta_imagens: int = Field(default=TARGET_COUNT, ge=1, le=TARGET_COUNT)


class ExploracaoDirecaoRequest(BaseModel):
    brief: str = Field(
        min_length=1,
        max_length=4000,
        validation_alias=AliasChoices("brief", "estetica"),
    )
    quantidade: int = Field(default=5, ge=1, le=TARGET_COUNT)


class DirecaoArteRequest(BaseModel):
    prompt_template: str = Field(min_length=1, max_length=4000)
    variacoes: list[str] = Field(
        min_length=1,
        max_length=1001,
        validation_alias=AliasChoices("variacoes", "esteticas"),
    )
    prompts: list[str] | None = Field(default=None, max_length=1001)
    motor: Literal["midjourney", "kittl", "flux"]
    image_path: str | None = Field(default=None, max_length=4096)
    image_data: str | None = Field(default=None, max_length=28_000_000)


ClonagemRequest = DirecaoArteRequest


load_env()


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    if os.environ.get("IDSTUDIO_AUTO_OPEN_BROWSER", "true").lower() in {
        "1",
        "true",
        "yes",
    }:
        webbrowser.open(
            os.environ.get("IDSTUDIO_URL", "http://127.0.0.1:8000")
        )
    yield


app = FastAPI(title="OpenMontage Image Curation", lifespan=_app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CURATION_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
curation_jobs: dict[str, dict[str, object]] = {}
OUTPUT_DIR = REPO_ROOT / "output" / "pipeline_imagens"


@app.get("/", include_in_schema=False)
async def painel_idstudio() -> FileResponse:
    return FileResponse(REPO_ROOT / "static" / "index.html")


app.mount(
    "/generated",
    StaticFiles(directory=OUTPUT_DIR, check_dir=False),
    name="generated",
)


def _job_is_active() -> bool:
    return any(
        job.get("status") in {"queued", "running"}
        for job in curation_jobs.values()
    )


def _job_thumbnails() -> list[str]:
    if not OUTPUT_DIR.is_dir():
        return []
    images = sorted(
        (
            path
            for path in OUTPUT_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:8]
    return [
        "/generated/" + path.relative_to(OUTPUT_DIR).as_posix()
        for path in images
    ]


async def _run_api_job(
    job_id: str,
    criterio: str,
    reference_url: str | None = None,
    root: Path = DEFAULT_INPUT_DIR,
    target_count: int = TARGET_COUNT,
) -> None:
    job = curation_jobs[job_id]
    job["status"] = "running"

    async def report_progress(update: dict[str, object]) -> None:
        job.update(update)
        job["thumbnails"] = _job_thumbnails()

    try:
        summary = await processar_pastas_locais(
            root=root,
            criterio=criterio,
            output_dir=OUTPUT_DIR,
            titulo=f"Curadoria {date.today().isoformat()}",
            database_id=os.environ.get("NOTION_DATABASE_ID", ""),
            parent_page_id=os.environ.get("NOTION_PARENT_PAGE_ID", ""),
            progress_callback=report_progress,
            reference_url=reference_url,
            target_count=target_count,
        )
        job.update(summary)
        job["status"] = "completed_with_errors" if int(summary["failed"]) else "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        raise


async def _run_clone_job(
    job_id: str,
    variacoes: list[str],
    motor: Literal["midjourney", "kittl", "flux"],
    reference_image: Path | None,
) -> None:
    job = curation_jobs[job_id]
    job.update({"status": "running", "total": len(variacoes)})
    errors: list[dict[str, str]] = []
    generated: list[str] = []
    items = job["items"]
    assert isinstance(items, list)
    destination_dir = OUTPUT_DIR / "direcao_arte" / job_id

    for index, variacao in enumerate(variacoes, start=1):
        destination = destination_dir / f"{motor}_{index:04d}.png"
        job["current_folder"] = variacao
        item = items[index - 1]
        assert isinstance(item, dict)
        prompt = item["prompt"]
        assert isinstance(prompt, str)
        item["status"] = "running"
        try:
            if motor == "flux":
                image_path = await gerar_flux(prompt, destination)
            elif motor == "kittl":
                image_path = await gerar_kittl(prompt, destination)
            else:
                image_path = await gerar_midjourney_web(
                    prompt,
                    destination,
                    reference_image=reference_image,
                )
            thumbnail = "/generated/" + image_path.relative_to(OUTPUT_DIR).as_posix()
            generated.append(thumbnail)
            item.update({"status": "completed", "thumbnail": thumbnail})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"variacao": variacao, "error": error})
            item.update({"status": "failed", "error": error})

        job.update(
            {
                "completed": index,
                "processed": len(generated),
                "failed": len(errors),
                "errors": errors,
                "thumbnails": generated[-8:],
            }
        )

    job["current_folder"] = None
    job["status"] = "completed_with_errors" if errors else "completed"


@app.post("/iniciar-curadoria", status_code=202)
async def iniciar_curadoria(
    request: CuradoriaRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    load_env()
    if not request.criterio.strip():
        raise HTTPException(status_code=422, detail="criterio must not be empty")
    if request.reference_url:
        try:
            _validate_midjourney_reference_url(request.reference_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    root = (
        Path(request.pasta_raiz).expanduser()
        if request.pasta_raiz
        else DEFAULT_INPUT_DIR
    )
    if request.pasta_raiz is None:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise HTTPException(status_code=422, detail=f"Folder not found: {root}")
    if _job_is_active():
        raise HTTPException(status_code=409, detail="A curation job is already active")
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    curation_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": 0,
        "completed": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "current_folder": None,
    }
    background_tasks.add_task(
        _run_api_job,
        job_id,
        request.criterio.strip(),
        request.reference_url,
        root,
        request.meta_imagens,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/iniciar-direcao", status_code=202)
@app.post("/iniciar-clonagem", status_code=202, include_in_schema=False)
async def iniciar_direcao_arte(
    request: DirecaoArteRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    if "{variavel}" not in request.prompt_template:
        raise HTTPException(
            status_code=422,
            detail="prompt_template must include the {variavel} placeholder",
        )
    variacoes = [item.strip() for item in request.variacoes]
    if any(not item for item in variacoes):
        raise HTTPException(
            status_code=422,
            detail="Variation entries must not be empty",
        )
    prompts = request.prompts
    if prompts is not None:
        if len(prompts) != len(variacoes):
            raise HTTPException(
                status_code=422,
                detail="Provide exactly one prompt for each variation",
            )
        prompts = [prompt.strip() for prompt in prompts]
        if any(not prompt for prompt in prompts):
            raise HTTPException(status_code=422, detail="Prompts must not be empty")
    if bool(request.image_path) and request.image_data is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide either image_path or image_data, not both",
        )
    if (
        (request.image_path or request.image_data is not None)
        and request.motor != "midjourney"
    ):
        raise HTTPException(
            status_code=422,
            detail="Only Midjourney Web currently accepts an image reference",
        )
    if _job_is_active():
        raise HTTPException(status_code=409, detail="Another image job is already active")

    job_id = uuid.uuid4().hex
    reference_image = None
    if request.image_path:
        reference_image = Path(request.image_path).expanduser()
        if not reference_image.is_file():
            raise HTTPException(
                status_code=422,
                detail=f"Reference image not found: {reference_image}",
            )
        if reference_image.stat().st_size > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Reference images must be 20 MiB or smaller",
            )
        try:
            with Image.open(reference_image) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Invalid reference image",
            ) from exc
    elif request.image_data is not None:
        try:
            encoded = request.image_data.partition(",")[2] or request.image_data
            image_bytes = base64.b64decode(encoded, validate=True)
            if len(image_bytes) > 20 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Reference images must be 20 MiB or smaller",
                )
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_format = image.format
                image.verify()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid reference image") from exc
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
            image_format or ""
        )
        if extension is None:
            raise HTTPException(
                status_code=422,
                detail="Reference image must be JPEG, PNG, or WebP",
            )
        reference_image = OUTPUT_DIR / "direcao_arte" / job_id / f"reference{extension}"
        reference_image.parent.mkdir(parents=True, exist_ok=True)
        reference_image.write_bytes(image_bytes)

    curation_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "total": len(variacoes),
        "completed": 0,
        "processed": 0,
        "failed": 0,
        "errors": [],
        "current_folder": None,
        "thumbnails": [],
        "items": [
            {
                "index": index,
                "variacao": variacao,
                "estetica": variacao,
                "prompt": (
                    prompts[index - 1]
                    if prompts is not None
                    else request.prompt_template.replace("{variavel}", variacao)
                ),
                "motor": request.motor,
                "status": "queued",
                "thumbnail": None,
                "error": None,
            }
            for index, variacao in enumerate(variacoes, start=1)
        ],
    }
    background_tasks.add_task(
        _run_clone_job,
        job_id,
        variacoes,
        request.motor,
        reference_image,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/progresso/{job_id}")
async def progresso_curadoria(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
) -> dict[str, object]:
    job = curation_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Curation job not found")
    snapshot = dict(job)
    items = snapshot.get("items")
    if isinstance(items, list):
        start = (page - 1) * page_size
        snapshot["items_total"] = len(items)
        snapshot["items_page"] = page
        snapshot["items_page_size"] = page_size
        snapshot["items"] = items[start : start + page_size]
    return snapshot


@app.post("/explorar-direcao")
@app.post("/explorar-estetica", include_in_schema=False)
async def explorar_direcao(
    request: ExploracaoDirecaoRequest,
) -> dict[str, list[str]]:
    try:
        prompts = await expandir_direcao_visual_midjourney(
            request.brief.strip(),
            quantidade=request.quantidade,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"prompts": prompts}


async def executar_pipeline(args: argparse.Namespace) -> None:
    load_env()
    if args.todas_pastas_locais:
        summary = await processar_pastas_locais(
            root=args.local_dir,
            criterio=args.criterio,
            output_dir=args.output_dir,
            titulo=args.titulo,
            database_id=args.notion_database_id,
            parent_page_id=args.notion_parent_page_id,
            dry_run=args.dry_run,
            reference_image=args.reference_image,
            reference_url=args.reference_url,
        )
        print(
            "Lote finalizado: "
            f"{summary['processed']} processadas, {summary['skipped']} ignoradas, "
            f"{summary['failed']} com falha."
        )
        if int(summary["failed"]):
            raise RuntimeError(f"{summary['failed']} folder(s) failed; see errors above")
        return

    result = await executar_pipeline_pasta(
        folder_name=args.eagle_folder,
        local_dir=args.local_dir,
        criterio=args.criterio,
        output_dir=args.output_dir,
        titulo=args.titulo,
        database_id=args.notion_database_id,
        parent_page_id=args.notion_parent_page_id,
        dry_run=args.dry_run,
        reference_image=args.reference_image,
        reference_url=args.reference_url,
    )
    if result["status"] == "skipped":
        print("Checkpoint encontrado; nada a fazer.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eagle-folder",
        default=os.environ.get("EAGLE_FOLDER_NAME", "assets_teste"),
        help="Nome da pasta no Eagle; fallback local é assets_teste/.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Pasta local usada se Eagle estiver indisponível.",
    )
    parser.add_argument(
        "--todas-pastas-locais",
        action="store_true",
        help="Processa cada pasta local com imagens e respeita os checkpoints.",
    )
    parser.add_argument(
        "--criterio",
        default=os.environ.get("IMAGE_SELECTION_CRITERION", DEFAULT_CRITERION),
        help="Critério editorial enviado ao modelo de visão do proxy local.",
    )
    reference_image = os.environ.get("MIDJOURNEY_REFERENCE_IMAGE_PATH", "").strip()
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=Path(reference_image).expanduser() if reference_image else None,
        help="Imagem local de referência enviada ao Midjourney (opcional).",
    )
    parser.add_argument(
        "--reference-url",
        default=os.environ.get("MIDJOURNEY_REFERENCE_IMAGE_URL", "").strip() or None,
        help="URL HTTPS de imagem usada como referência no prompt do Midjourney.",
    )
    parser.add_argument(
        "--titulo",
        default=f"Curadoria de imagens {date.today().isoformat()}",
        help="Título da página/item criado no Notion.",
    )
    parser.add_argument(
        "--notion-database-id",
        default=os.environ.get("NOTION_DATABASE_ID", ""),
        help="ID de database do Notion; substitui NOTION_DATABASE_ID.",
    )
    parser.add_argument(
        "--notion-parent-page-id",
        default=os.environ.get("NOTION_PARENT_PAGE_ID", ""),
        help="Alternativa: criar a página sob uma página existente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "pipeline_imagens",
        help="Diretório local das imagens geradas e WebP resultantes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Curar e converter imagens, sem enviar arquivos ao Notion.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = _parse_args()
    if args.local_dir == DEFAULT_INPUT_DIR:
        args.local_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(executar_pipeline(args))


if __name__ == "__main__":
    main()
