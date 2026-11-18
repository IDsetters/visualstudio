"""Midjourney web UI automation using a manually authenticated local profile."""

from __future__ import annotations

import argparse
import asyncio
import os
from asyncio import AbstractEventLoop, Lock
from pathlib import Path
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from PIL import Image, UnidentifiedImageError

from lib.env_loader import load_env


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = REPO_ROOT / ".midjourney-profile"
_LOOP_LOCKS: WeakKeyDictionary[AbstractEventLoop, Lock] = WeakKeyDictionary()
MIDJOURNEY_HOSTS = {"midjourney.com", "www.midjourney.com"}


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before using Midjourney web automation")
    return value


def _profile_dir() -> Path:
    value = os.environ.get("MIDJOURNEY_USER_DATA_DIR", "").strip()
    return Path(value).expanduser() if value else DEFAULT_PROFILE_DIR


def _official_url(name: str, value: str) -> str:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.username or parsed.password:
        raise ValueError(f"{name} must not include embedded credentials")
    if (
        parsed.scheme != "https"
        or not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in MIDJOURNEY_HOSTS
        )
    ):
        raise ValueError(f"{name} must use an HTTPS Midjourney website URL")
    return value


def _reference_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Midjourney reference URLs must be public HTTPS URLs")
    return value


def _loop_lock() -> Lock:
    loop = asyncio.get_running_loop()
    lock = _LOOP_LOCKS.get(loop)
    if lock is None:
        lock = Lock()
        _LOOP_LOCKS[loop] = lock
    return lock


def _timeout_ms() -> int:
    try:
        timeout_seconds = int(
            os.environ.get("MIDJOURNEY_GENERATION_TIMEOUT_SECONDS", "600")
        )
    except ValueError as exc:
        raise ValueError(
            "MIDJOURNEY_GENERATION_TIMEOUT_SECONDS must be an integer"
        ) from exc
    if timeout_seconds <= 0:
        raise ValueError("MIDJOURNEY_GENERATION_TIMEOUT_SECONDS must be positive")
    return timeout_seconds * 1000


def _headless() -> bool:
    value = os.environ.get("MIDJOURNEY_HEADLESS", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("MIDJOURNEY_HEADLESS must be true or false")
    return value == "true"


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Midjourney download is not a valid image: {path}") from exc


def _playwright_context():
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Install Playwright with `python -m pip install playwright` and "
            "`python -m playwright install chromium`"
        ) from exc
    return async_playwright()


async def gerar_midjourney(
    prompt: str,
    destination: Path,
    reference_image: Path | None = None,
    reference_url: str | None = None,
) -> Path:
    """Submit one prompt in the official UI and save its visible download."""
    if not prompt.strip():
        raise ValueError("Midjourney prompt must not be empty")

    url = _official_url(
        "MIDJOURNEY_GENERATOR_URL",
        os.environ.get(
            "MIDJOURNEY_GENERATOR_URL", "https://www.midjourney.com/imagine"
        ).strip()
        or "https://www.midjourney.com/imagine",
    )
    prompt_selector = _required_setting("MIDJOURNEY_PROMPT_SELECTOR")
    generate_selector = _required_setting("MIDJOURNEY_GENERATE_SELECTOR")
    download_selector = _required_setting("MIDJOURNEY_DOWNLOAD_SELECTOR")
    reference_selector = os.environ.get(
        "MIDJOURNEY_REFERENCE_UPLOAD_SELECTOR", ""
    ).strip()
    upscale_selector = os.environ.get("MIDJOURNEY_UPSCALE_SELECTOR", "").strip()
    timeout_ms = _timeout_ms()

    if reference_url:
        reference_url = _reference_url(reference_url)
    if reference_image is not None:
        reference_image = Path(reference_image).expanduser()
        if not reference_image.is_file():
            raise FileNotFoundError(f"Midjourney reference image not found: {reference_image}")
        if not reference_selector:
            raise RuntimeError(
                "Set MIDJOURNEY_REFERENCE_UPLOAD_SELECTOR to upload a local reference image"
            )
        await asyncio.to_thread(_validate_image, reference_image)

    profile_dir = _profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submitted_prompt = f"{reference_url}\n{prompt}" if reference_url else prompt

    async with _loop_lock():
        async with _playwright_context() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=_headless(),
                accept_downloads=True,
            )
            try:
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                if reference_image is not None:
                    await page.locator(reference_selector).set_input_files(
                        str(reference_image),
                        timeout=timeout_ms,
                    )
                await page.locator(prompt_selector).fill(
                    submitted_prompt,
                    timeout=timeout_ms,
                )
                await page.locator(generate_selector).click(timeout=timeout_ms)
                if upscale_selector:
                    await page.locator(upscale_selector).click(timeout=timeout_ms)
                async with page.expect_download(timeout=timeout_ms) as download_info:
                    await page.locator(download_selector).click(timeout=timeout_ms)
                download = await download_info.value
                await download.save_as(str(destination))
                try:
                    await asyncio.to_thread(_validate_image, destination)
                except RuntimeError:
                    destination.unlink(missing_ok=True)
                    raise
            finally:
                await context.close()

    return destination


async def _login() -> None:
    login_url = _official_url(
        "MIDJOURNEY_LOGIN_URL",
        os.environ.get("MIDJOURNEY_LOGIN_URL", "https://www.midjourney.com").strip()
        or "https://www.midjourney.com",
    )
    profile_dir = _profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    async with _playwright_context() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            accept_downloads=True,
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(login_url, wait_until="domcontentloaded")
            print("Faça login manualmente na janela do Chromium do Midjourney.")
            await asyncio.to_thread(input, "Pressione Enter após concluir o login: ")
        finally:
            await context.close()


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Configure a local Midjourney session")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Open Chromium for a manual login saved in the local profile",
    )
    args = parser.parse_args()
    if not args.login:
        parser.error("Use --login to open the browser for manual account sign-in")
    asyncio.run(_login())


if __name__ == "__main__":
    main()
