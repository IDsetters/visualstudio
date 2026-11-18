from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from generators import flux as flux_generator
from generators import kittl_automation
from generators import midjourney_web
import pipeline_imagens


def test_llm_thumbnail_is_limited_to_512_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (2048, 1024), "green").save(image_path)

    thumbnail = base64.b64decode(pipeline_imagens._thumbnail_as_base64(image_path))
    with Image.open(io.BytesIO(thumbnail)) as image:
        assert image.size == (512, 256)


def test_eagle_falls_back_to_local_images(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "brief.png"
    Image.new("RGB", (20, 20), "blue").save(image_path)

    async def unavailable(_folder_name: str) -> list[Path]:
        raise pipeline_imagens.EagleUnavailableError("Eagle is not running")

    monkeypatch.setattr(pipeline_imagens, "_read_eagle_folder", unavailable)

    result = asyncio.run(pipeline_imagens.ler_imagens_eagle("test", tmp_path))

    assert result == [image_path]


def test_local_llm_selection_uses_validated_asset_ids(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (20, 20), "red").save(image_path)

    class FakeClient:
        instance = None

        def __init__(self, **kwargs) -> None:
            self.configuration = kwargs
            FakeClient.instance = self
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_message)
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def create_message(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"selected":["img_0001"]}')
                    )
                ]
            )

    monkeypatch.setenv("LLM_API_KEY", "sk-dummy-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local-vision-model")
    monkeypatch.setattr(pipeline_imagens, "AsyncOpenAI", FakeClient)

    result = asyncio.run(
        pipeline_imagens.selecionar_imagens([image_path], "Escolha a mais nítida")
    )

    assert [asset.path for asset in result] == [image_path]
    assert FakeClient.instance.configuration == {
        "api_key": "sk-dummy-key",
        "base_url": "http://127.0.0.1:8080/v1",
    }


def test_mock_generation_and_webp_conversion_fill_output(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    generated = asyncio.run(
        pipeline_imagens.gerar_imagens(2, "composição editorial", tmp_path / "mock")
    )
    webp_paths = asyncio.run(
        pipeline_imagens.converter_webp(generated, tmp_path / "webp")
    )

    assert len(generated) == len(webp_paths) == 2
    with Image.open(webp_paths[0]) as converted:
        assert converted.format == "WEBP"
        assert converted.size == (1024, 1024)


def test_flux_generation_is_used_when_configured(monkeypatch, tmp_path: Path) -> None:
    generated_paths = []

    async def fake_flux(prompt: str, destination: Path) -> Path:
        assert "composição editorial" in prompt
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"generated")
        generated_paths.append(destination)
        return destination

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.delenv("IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.setattr(pipeline_imagens, "gerar_flux", fake_flux)

    result = asyncio.run(
        pipeline_imagens.gerar_imagens(2, "composição editorial", tmp_path / "flux")
    )

    assert result == generated_paths
    assert [path.name for path in result] == ["flux_01.png", "flux_02.png"]


def test_flux_adapter_calls_existing_tool_in_a_worker_thread(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeFluxImage:
        def execute(self, inputs: dict) -> SimpleNamespace:
            output_path = Path(inputs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"flux-image")
            return SimpleNamespace(
                success=True,
                data={"output": str(output_path)},
            )

    monkeypatch.setenv("FLUX_IMAGE_MODEL", "flux-pro/v1.1")
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    monkeypatch.setattr(flux_generator, "FluxImage", FakeFluxImage)

    result = asyncio.run(
        flux_generator.gerar_flux("composição editorial", tmp_path / "flux.png")
    )

    assert result == tmp_path / "flux.png"
    assert result.read_bytes() == b"flux-image"


def test_flux_adapter_uses_direct_bfl_when_fal_is_not_configured(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeBFLImage:
        def execute(self, inputs: dict) -> SimpleNamespace:
            assert inputs["prompt"] == "composição editorial"
            assert inputs["aspect_ratio"] == "1:1"
            assert inputs["grounding"] is False
            output_path = Path(inputs["output_path"])
            output_path.write_bytes(b"bfl-image")
            return SimpleNamespace(
                success=True,
                data={"output": str(output_path)},
            )

    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
    monkeypatch.setenv("BFL_API_KEY", "test-key")
    monkeypatch.setattr(flux_generator, "BFLImage", FakeBFLImage)

    result = asyncio.run(
        flux_generator.gerar_flux("composição editorial", tmp_path / "bfl.png")
    )

    assert result.read_bytes() == b"bfl-image"


def test_local_llm_expands_midjourney_style_parameters(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.configuration = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_message)
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def create_message(self, **kwargs):
            assert "exatamente 1 prompts" in kwargs["messages"][0]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"variations":[{"prompt":"Editorial portrait",'
                                '"sref":"12345","stylize":250,"ar":"3:2"}]}'
                            )
                        )
                    )
                ]
            )

    monkeypatch.setenv("LLM_API_KEY", "sk-dummy-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local-text-model")
    monkeypatch.setattr(pipeline_imagens, "AsyncOpenAI", FakeClient)

    prompts = asyncio.run(
        pipeline_imagens.expandir_estetica_midjourney("editorial", quantidade=1)
    )

    assert prompts == ["Editorial portrait --sref 12345 --stylize 250 --ar 3:2"]


def test_midjourney_web_uses_profile_selectors_and_local_reference(
    monkeypatch, tmp_path: Path
) -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (18, 18), "navy").save(image_buffer, format="PNG")
    reference = tmp_path / "reference.png"
    Image.new("RGB", (18, 18), "orange").save(reference)

    class FakeDownload:
        async def save_as(self, path: str) -> None:
            Path(path).write_bytes(image_buffer.getvalue())

    class FakeDownloadInfo:
        def __init__(self) -> None:
            self.value = asyncio.get_running_loop().create_future()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            self.value.set_result(FakeDownload())

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        async def fill(self, value: str, **_kwargs) -> None:
            assert self.selector == "#prompt"
            assert value.startswith("https://images.example/reference.jpg\n")
            assert "--stylize 250" in value

        async def click(self, **_kwargs) -> None:
            assert self.selector in {"#generate", "#download"}

        async def set_input_files(self, path: str, **_kwargs) -> None:
            assert self.selector == "input[type=file]"
            assert Path(path) == reference

    class FakePage:
        def __init__(self) -> None:
            self.navigated_to = None

        async def goto(self, url: str, **_kwargs) -> None:
            self.navigated_to = url

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def expect_download(self, **_kwargs) -> FakeDownloadInfo:
            return FakeDownloadInfo()

    page = FakePage()

    class FakeContext:
        pages = [page]

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch_persistent_context(self, **kwargs) -> FakeContext:
            assert kwargs["user_data_dir"] == str(tmp_path / "mj-profile")
            assert kwargs["headless"] is True
            assert kwargs["accept_downloads"] is True
            return FakeContext()

    class FakePlaywright:
        chromium = FakeChromium()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        midjourney_web,
        "_playwright_context",
        lambda: FakePlaywright(),
    )
    monkeypatch.setenv("MIDJOURNEY_GENERATOR_URL", "https://www.midjourney.com/imagine")
    monkeypatch.setenv("MIDJOURNEY_PROMPT_SELECTOR", "#prompt")
    monkeypatch.setenv("MIDJOURNEY_GENERATE_SELECTOR", "#generate")
    monkeypatch.setenv("MIDJOURNEY_DOWNLOAD_SELECTOR", "#download")
    monkeypatch.setenv("MIDJOURNEY_REFERENCE_UPLOAD_SELECTOR", "input[type=file]")
    monkeypatch.setenv("MIDJOURNEY_USER_DATA_DIR", str(tmp_path / "mj-profile"))
    monkeypatch.setenv("MIDJOURNEY_HEADLESS", "true")

    result = asyncio.run(
        midjourney_web.gerar_midjourney(
            "Editorial portrait --stylize 250 --ar 3:2",
            tmp_path / "generated.png",
            reference_image=reference,
            reference_url="https://images.example/reference.jpg",
        )
    )

    assert result == tmp_path / "generated.png"
    assert page.navigated_to == "https://www.midjourney.com/imagine"
    with Image.open(result) as image:
        assert image.size == (18, 18)


def test_pipeline_routes_midjourney_through_style_explorer_and_web(
    monkeypatch, tmp_path: Path
) -> None:
    async def fake_expand(_aesthetic: str, quantity: int) -> list[str]:
        assert quantity == 2
        return [
            "style one --stylize 100 --ar 1:1",
            "style two --stylize 200 --ar 3:2",
        ]

    generated = []

    async def fake_midjourney(prompt, destination, reference_image, reference_url):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"generated")
        generated.append((prompt, reference_image, reference_url))
        return destination

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "midjourney")
    monkeypatch.setattr(pipeline_imagens, "expandir_estetica_midjourney", fake_expand)
    monkeypatch.setattr(pipeline_imagens, "gerar_midjourney_web", fake_midjourney)

    result = asyncio.run(
        pipeline_imagens.gerar_imagens(
            2,
            "editorial style",
            tmp_path / "generated",
            reference_image=reference,
            reference_url="https://images.example/style.jpg",
        )
    )

    assert len(result) == 2
    assert generated == [
        ("style one --stylize 100 --ar 1:1", reference, "https://images.example/style.jpg"),
        ("style two --stylize 200 --ar 3:2", reference, "https://images.example/style.jpg"),
    ]


def test_kittl_requires_explicit_ui_configuration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "kittl")
    monkeypatch.setenv("KITTL_GENERATOR_URL", "")

    with pytest.raises(RuntimeError, match="Set KITTL_GENERATOR_URL"):
        asyncio.run(
            pipeline_imagens.gerar_imagens(1, "composição editorial", tmp_path)
        )


def test_kittl_ui_flow_uses_configured_selectors_and_download(
    monkeypatch, tmp_path: Path
) -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "teal").save(image_buffer, format="PNG")

    class FakeDownload:
        async def save_as(self, path: str) -> None:
            Path(path).write_bytes(image_buffer.getvalue())

    class FakeDownloadInfo:
        def __init__(self) -> None:
            self.value = asyncio.get_running_loop().create_future()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            self.value.set_result(FakeDownload())

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        async def fill(self, value: str, **_kwargs) -> None:
            assert self.selector == "#prompt"
            assert value == "Imagem editorial"

        async def click(self, **_kwargs) -> None:
            assert self.selector in {"#generate", "#download"}

    class FakePage:
        def __init__(self) -> None:
            self.navigated_to = None

        async def goto(self, url: str, **_kwargs) -> None:
            self.navigated_to = url

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def expect_download(self, **_kwargs) -> FakeDownloadInfo:
            return FakeDownloadInfo()

    page = FakePage()

    class FakeContext:
        pages = [page]

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch_persistent_context(self, **kwargs) -> FakeContext:
            assert kwargs["accept_downloads"] is True
            assert kwargs["headless"] is True
            return FakeContext()

    class FakePlaywright:
        chromium = FakeChromium()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        kittl_automation,
        "_playwright_context",
        lambda: FakePlaywright(),
    )
    monkeypatch.setenv("KITTL_GENERATOR_URL", "https://www.kittl.com/generator")
    monkeypatch.setenv("KITTL_PROMPT_SELECTOR", "#prompt")
    monkeypatch.setenv("KITTL_GENERATE_SELECTOR", "#generate")
    monkeypatch.setenv("KITTL_DOWNLOAD_SELECTOR", "#download")
    monkeypatch.setenv("KITTL_HEADLESS", "true")
    monkeypatch.setenv("KITTL_USER_DATA_DIR", str(tmp_path / "profile"))

    result = asyncio.run(
        kittl_automation.gerar_kittl(
            "Imagem editorial",
            tmp_path / "generated.png",
        )
    )

    assert result == tmp_path / "generated.png"
    assert page.navigated_to == "https://www.kittl.com/generator"
    assert result.is_file()
    with Image.open(result) as image:
        assert image.size == (16, 16)


def test_pipeline_routes_kittl_generation_to_browser_adapter(
    monkeypatch, tmp_path: Path
) -> None:
    async def fake_kittl(prompt: str, destination: Path) -> Path:
        assert "composição editorial" in prompt
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"kittl-image")
        return destination

    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "kittl")
    monkeypatch.setattr(pipeline_imagens, "gerar_kittl", fake_kittl)

    result = asyncio.run(
        pipeline_imagens.gerar_imagens(1, "composição editorial", tmp_path)
    )

    assert result == [tmp_path / "kittl_01.png"]
    assert result[0].read_bytes() == b"kittl-image"


def test_notion_publishing_uploads_and_embeds_every_image(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeResponse:
        status_code = 200
        is_error = False
        text = ""

        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []
            self.upload_count = 0
            self.page_payload = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def request(self, method: str, endpoint: str, **kwargs):
            self.calls.append((method, endpoint, kwargs))
            if endpoint == "/file_uploads":
                self.upload_count += 1
                return FakeResponse({"id": f"upload-{self.upload_count}"})
            if endpoint == "/pages":
                self.page_payload = kwargs["json"]
                return FakeResponse({"url": "https://notion.so/test-page"})
            return FakeResponse({"status": "uploaded"})

    fake_client = FakeClient()
    monkeypatch.setenv("NOTION_TOKEN", "test-token")
    monkeypatch.setenv("NOTION_MIN_REQUEST_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(
        pipeline_imagens.httpx,
        "AsyncClient",
        lambda **_kwargs: fake_client,
    )
    image_paths = []
    for index in range(pipeline_imagens.TARGET_COUNT):
        image_path = tmp_path / f"{index:02d}.webp"
        image_path.write_bytes(b"webp")
        image_paths.append(image_path)

    page_url = asyncio.run(
        pipeline_imagens.publicar_notion(image_paths, "Teste", database_id="database-id")
    )

    assert page_url == "https://notion.so/test-page"
    assert fake_client.upload_count == pipeline_imagens.TARGET_COUNT
    assert len(fake_client.page_payload["children"]) == pipeline_imagens.TARGET_COUNT
    assert all(
        child["image"]["type"] == "file_upload"
        for child in fake_client.page_payload["children"]
    )


def test_batch_skips_folders_with_completed_checkpoints(tmp_path: Path) -> None:
    folder = tmp_path / "folder-a"
    folder.mkdir()
    Image.new("RGB", (20, 20), "purple").save(folder / "photo.jpg")
    (folder / pipeline_imagens.CHECKPOINT_FILENAME).write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    summary = asyncio.run(
        pipeline_imagens.processar_pastas_locais(
            root=tmp_path,
            criterio="editorial",
            output_dir=tmp_path / "output",
            titulo="Curadoria",
        )
    )

    assert summary["total"] == 1
    assert summary["processed"] == 0
    assert summary["skipped"] == 1
    assert summary["failed"] == 0


def test_checkpoint_is_written_after_successful_notion_publish(
    monkeypatch, tmp_path: Path
) -> None:
    folder = tmp_path / "folder-a"
    folder.mkdir()
    image_path = folder / "photo.png"
    Image.new("RGB", (20, 20), "orange").save(image_path)
    monkeypatch.setattr(pipeline_imagens, "TARGET_COUNT", 1)
    monkeypatch.setenv("LLM_MODEL", "local-vision-model")
    monkeypatch.setenv("NOTION_TOKEN", "test-token")
    monkeypatch.setenv("NOTION_DATABASE_ID", "database-id")

    async def select_one(paths, _criterion):
        return [pipeline_imagens.ImageAsset("img_0001", paths[0])]

    async def no_generated_images(
        deficit, _criterion, _output, reference_image=None, reference_url=None
    ):
        assert deficit == 0
        assert reference_image == image_path
        assert reference_url is None
        return []

    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "midjourney")

    async def publish(_paths, **_kwargs):
        return "https://notion.so/published"

    monkeypatch.setattr(pipeline_imagens, "selecionar_imagens", select_one)
    monkeypatch.setattr(pipeline_imagens, "gerar_imagens", no_generated_images)
    monkeypatch.setattr(pipeline_imagens, "publicar_notion", publish)

    result = asyncio.run(
        pipeline_imagens.executar_pipeline_pasta(
            folder_name="folder-a",
            local_dir=folder,
            criterio="editorial",
            output_dir=tmp_path / "output",
            titulo="Curadoria",
            read_eagle=False,
            recursive_fallback=False,
        )
    )

    checkpoint_path = folder / pipeline_imagens.CHECKPOINT_FILENAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert checkpoint["notion_url"] == "https://notion.so/published"
    assert checkpoint["webp_count"] == 1


def test_api_job_exposes_batch_progress(monkeypatch, tmp_path: Path) -> None:
    async def fake_batch(**kwargs):
        assert kwargs["reference_url"] == "https://images.example/reference.png"
        await kwargs["progress_callback"](
            {
                "total": 1,
                "completed": 1,
                "processed": 1,
                "skipped": 0,
                "failed": 0,
                "current_folder": "folder-a",
            }
        )
        return {
            "total": 1,
            "completed": 1,
            "processed": 1,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "current_folder": None,
        }

    local_root = tmp_path / "assets_teste"
    monkeypatch.setattr(pipeline_imagens, "DEFAULT_INPUT_DIR", local_root)
    monkeypatch.setattr(pipeline_imagens, "processar_pastas_locais", fake_batch)

    with TestClient(pipeline_imagens.app) as client:
        start = client.post(
            "/iniciar-curadoria",
            json={
                "criterio": "editorial",
                "reference_url": "https://images.example/reference.png",
            },
        )
        assert start.status_code == 202
        assert local_root.is_dir()
        job_id = start.json()["job_id"]
        progress = client.get(f"/progresso/{job_id}").json()

    assert pipeline_imagens.curation_jobs[job_id]["status"] == "completed"
    assert progress["completed"] == 1
    assert progress["processed"] == 1
    assert {route.path for route in pipeline_imagens.app.routes} >= {
        "/iniciar-curadoria",
        "/progresso/{job_id}",
    }
    del pipeline_imagens.curation_jobs[job_id]
