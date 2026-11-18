"""Kittl UI automation using a user-managed, persistent Chromium profile."""

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
DEFAULT_PROFILE_DIR = REPO_ROOT / ".kittl-profile"
_LOOP_LOCKS: WeakKeyDictionary[AbstractEventLoop, Lock] = WeakKeyDictionary()


def _required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before using Kittl automation")
    return value


def _profile_dir() -> Path:
    value = os.environ.get("KITTL_USER_DATA_DIR", "").strip()
    return Path(value).expanduser() if value else DEFAULT_PROFILE_DIR


def _validate_web_url(name: str, value: str) -> str:
    parsed = urlsplit(value)
    if parsed.username or parsed.password:
        raise ValueError(f"{name} must not include embedded credentials")
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if parsed.scheme == "http" and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        return value
    raise ValueError(f"{name} must use HTTPS or local HTTP")


def _loop_lock() -> Lock:
    loop = asyncio.get_running_loop()
    lock = _LOOP_LOCKS.get(loop)
    if lock is None:
        lock = Lock()
        _LOOP_LOCKS[loop] = lock
    return lock


def _timeout_ms() -> int:
    try:
        timeout_seconds = int(os.environ.get("KITTL_GENERATION_TIMEOUT_SECONDS", "300"))
    except ValueError as exc:
        raise ValueError("KITTL_GENERATION_TIMEOUT_SECONDS must be an integer") from exc
    if timeout_seconds <= 0:
        raise ValueError("KITTL_GENERATION_TIMEOUT_SECONDS must be positive")
    return timeout_seconds * 1000


def _headless() -> bool:
    value = os.environ.get("KITTL_HEADLESS", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("KITTL_HEADLESS must be true or false")
    return value == "true"


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Kittl download is not a valid image: {path}") from exc


def _playwright_context():
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Install Playwright with `python -m pip install playwright` and "
            "`python -m playwright install chromium`"
        ) from exc
    return async_playwright()


async def gerar_kittl(prompt: str, destination: Path) -> Path:
    """Generate and download one image through the configured Kittl UI."""
    if not prompt.strip():
        raise ValueError("Kittl prompt must not be empty")

    url = _validate_web_url(
        "KITTL_GENERATOR_URL",
        _required_setting("KITTL_GENERATOR_URL"),
    )
    prompt_selector = _required_setting("KITTL_PROMPT_SELECTOR")
    generate_selector = _required_setting("KITTL_GENERATE_SELECTOR")
    download_selector = _required_setting("KITTL_DOWNLOAD_SELECTOR")
    timeout_ms = _timeout_ms()

    profile_dir = _profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)

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
                await page.locator(prompt_selector).fill(prompt, timeout=timeout_ms)
                await page.locator(generate_selector).click(timeout=timeout_ms)
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
    configured_url = os.environ.get("KITTL_LOGIN_URL", "").strip()
    login_url = _validate_web_url(
        "KITTL_LOGIN_URL",
        configured_url or "https://www.kittl.com",
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
            print("Faça login manualmente na janela do Chromium do Kittl.")
            await asyncio.to_thread(input, "Pressione Enter após concluir o login: ")
        finally:
            await context.close()


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Configure a local Kittl session")
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
