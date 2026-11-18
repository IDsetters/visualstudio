# Integration orchestration (Phase 1)

## Runtime and layout

OpenMontage's main runtime is Python 3.11 (`.python-version` and
`requirements.txt`). Node 22.23.4 is used by `remotion-composer`; there is no
root Node application to host a second worker stack. This phase therefore uses
Celery with Redis rather than BullMQ.

Keep integrations as small provider adapters and put workflow scheduling in
Celery tasks:

```text
lib/
  integrations/
    notion.py              # Notion pages and production-brief database
    llm.py                 # adapter for the local OpenAI-compatible LLM proxy
    tess.py                # Tess/Pareto adapter (after API access is confirmed)
    canva.py               # Canva OAuth/API adapter
    image_generation/      # BFL/Fal and authorized Midjourney provider adapters
    design/                # Canva/Kittl adapters
    eagle.py               # local Eagle library API
workers/
  celery_app.py            # broker, serialization, worker-wide policy
  tasks.py                 # small orchestration entry points
tests/
  integrations/
  workers/
```

Keep API calls, credentials, and provider response normalization inside
`lib/integrations/`; Celery tasks should pass IDs and small JSON payloads rather
than large media. Use separate queues/workers for slow image/video generation
as those adapters are added. The Notion task is limited to 2 task starts per
second per worker, and the adapter spaces individual requests by at least 0.4
seconds. Run one Notion worker with concurrency 1; this in-process pacing is
not a distributed limit. The task retries HTTP 429/5xx responses with
exponential backoff. Redis should be private to the application, and result
retention is limited to one hour because task results may contain script
content.

## Setup and run

From the repository root:

```bash
python -m pip install 'celery[redis]' notion-client openai httpx Pillow
cp .env.example .env
```

Set `NOTION_TOKEN` to an **internal integration token** and explicitly share
only the Notion pages/databases it needs with that integration. Start Redis
(for example, in a local development environment):

```bash
docker run --name openmontage-redis --rm -p 6379:6379 redis:7-alpine
```

Then start a worker:

```bash
celery -A workers.celery_app:celery_app worker --loglevel=INFO --concurrency=1
```

Queue a page read from another Python process:

```python
from workers.tasks import read_notion_page

job = read_notion_page.delay("NOTION_PAGE_ID")
print(job.id)
```

`job.get(timeout=60)` can be used by a trusted caller when it needs the result.
Avoid returning script text to public clients or logging it.

## Provider access notes

- **Notion:** use `NOTION_TOKEN` for a narrowly shared internal integration.
- **LLM:** `pipeline_imagens.py` uses `LLM_BASE_URL`, `LLM_API_KEY`, and
  `LLM_MODEL` with the OpenAI-compatible client, defaulting to the local proxy
  at `http://127.0.0.1:8080/v1`. The selected model must accept image inputs.
- **Tess and Canva:** credentials and API availability are account/product
  dependent. Confirm official API access and scopes before implementing an
  adapter; these environment variables do not imply that API access is
  included in a plan.
- **Midjourney:** the curation pipeline uses its official web UI with a
  manually authenticated local Playwright profile. Confirm that this
  automation is permitted for your account and use the documented visible UI
  controls; no Discord user-account automation or private API calls are used.
- **Kittl:** optional Playwright UI automation uses a local browser profile.
  Enable it only when permitted by the account terms.
- **Flux:** Flux is from Black Forest Labs, not Google. The repository already
  supports direct BFL credentials (`BFL_API_KEY`) and the fal.ai gateway
  (`FAL_KEY`). `GOOGLE_API_KEY` is for Google AI services, not Flux.
- **Higgsfield:** keep using the existing `HIGGSFIELD_API_KEY` /
  `HIGGSFIELD_API_SECRET` settings.
- **Eagle:** its API is local to the running Eagle application; enable it in
  Eagle and configure `EAGLE_API_BASE_URL` (and the key if required).

All integration credentials are optional for existing OpenMontage workflows.
Never commit `.env`; it is ignored by Git.

## Phase 2 image curation

`pipeline_imagens.py` creates temporary in-memory thumbnails no larger than
512x512 before sending them to the local vision model. It reads an Eagle folder
over its local API and falls back
to the ignored `assets_teste/` directory when Eagle is unavailable. It fills
the 16-image target with generated or clearly labeled local mock images,
converts the result to WebP under `output/`, and uploads those files to a
Notion database or parent page. A successful publish writes
`.curadoria_status.json` inside its source folder; batch mode skips folders
with that checkpoint. Run it from the repository root:

```bash
python pipeline_imagens.py --criterio "Imagens nítidas, com composição editorial"
python pipeline_imagens.py --todas-pastas-locais
uvicorn pipeline_imagens:app --host 127.0.0.1 --port 8000
```

The local API serves the IDstudio dark-mode panel at `http://127.0.0.1:8000/`
and opens it in the default browser at startup. Set `IDSTUDIO_URL` when using a
different host or port, or set `IDSTUDIO_AUTO_OPEN_BROWSER=false` to disable
automatic browser opening. The panel submits local-folder curation jobs,
polls `/progresso/{job_id}` for live status and generated thumbnails, and uses
`/explorar-direcao` to request Midjourney prompt variations from the local
OpenAI-compatible LLM configured with `LLM_BASE_URL` and `LLM_MODEL`. The art
direction workspace composes a brief from design elements, principles, and
visual techniques, includes aesthetics as one selectable visual axis, and
supports story arcs, audience/context, perceptual direction (attention entry,
rhythm, contrast and information density), accessibility intent, and custom
parameters such as palette, era, or material. These are creative hypotheses,
not universal neuropsychological predictions. Its preview computes Cartesian
combinations (up to 1001) before generation, allows prompt editing, shows
suggested technique-to-element/principle relationships, and provides a
paginated, per-prompt generation pipeline. Creative drafts are stored in the
browser's local storage; uploaded files and image paths are not saved.
`/iniciar-direcao` accepts generic `variacoes` and optional edited `prompts`;
`/iniciar-clonagem` and the `esteticas` request field remain as compatibility
aliases. Local image paths entered in the panel must be accessible to the
machine running the API.
Midjourney Web accepts the selected image as a visual reference, while Flux
and Kittl currently generate from prompt text only. Image uploads are limited
to 20 MiB. Progress responses paginate prompt items with `page` (1-based,
default `1`) and `page_size` (default `10`, maximum `50`); `items_total`,
`items_page`, and `items_page_size` describe the returned slice.

The API starts a local batch with `POST /iniciar-curadoria` and returns a job
ID; poll `GET /progresso/{job_id}` for counts, the current folder, and errors:

```bash
curl -X POST http://127.0.0.1:8000/iniciar-curadoria \
  -H 'Content-Type: application/json' \
  -d '{"criterio":"Imagens nítidas, com composição editorial"}'
curl http://127.0.0.1:8000/progresso/JOB_ID
```

Progress is in memory for the running API process; completed-folder
checkpoints persist across restarts. The CORS allowlist defaults to localhost
React/Next development origins and is configurable with
`CURATION_CORS_ORIGINS`.

The normal run needs `LLM_MODEL`, `NOTION_TOKEN`, and either
`NOTION_DATABASE_ID` (plus its title-property name) or
`NOTION_PARENT_PAGE_ID`. `LLM_BASE_URL` defaults to the local proxy and
`LLM_API_KEY` defaults to the dummy key shown in `.env.example`. Thumbnails
are sent only to that configured LLM endpoint; the final WebP files are
uploaded to Notion. For a deficit, `IMAGE_GENERATION_PROVIDER=auto` uses the
existing FLUX adapters when `FAL_KEY`, `FAL_AI_API_KEY`, or `BFL_API_KEY` is
configured; otherwise it keeps the labeled local mocks for development. Set
`FLUX_IMAGE_MODEL` to select an adapter-supported fal.ai FLUX model. FLUX API
calls may incur provider charges. Midjourney's Discord workflow remains
manual; the pipeline does not automate a Discord user account. Its official
web UI can be selected explicitly with
`IMAGE_GENERATION_PROVIDER=midjourney`; the local LLM expands the aesthetic
into validated `--sref`, `--stylize`, and `--ar` prompt variations. For
image-to-image, each folder uses its first selected image as the default local
reference; `--reference-image` can override it, and `--reference-url` adds a
public HTTPS image URL to the prompt. The API accepts `reference_url` in the
`POST /iniciar-curadoria` JSON body. A Midjourney batch submits one generation
per missing slot and consumes the plan's generation allowance; review the
requested deficit before starting a large batch.

Playwright uses only the official visible UI and normal browser downloads.
Configure its selectors in `.env` to match the controls visible in your
account. Log in manually once to the ignored persistent profile:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m generators.midjourney_web --login
python -m generators.kittl_automation --login
```

Set `MIDJOURNEY_PROMPT_SELECTOR`, `MIDJOURNEY_GENERATE_SELECTOR`, and
`MIDJOURNEY_DOWNLOAD_SELECTOR`; optionally set
`MIDJOURNEY_REFERENCE_UPLOAD_SELECTOR` and `MIDJOURNEY_UPSCALE_SELECTOR`.
`MIDJOURNEY_GENERATOR_URL` defaults to
`https://www.midjourney.com/imagine`; the profile defaults to the ignored
`.midjourney-profile/` directory. The configured download control should
download the single image intended for that pipeline slot.

Kittl can be selected explicitly with `IMAGE_GENERATION_PROVIDER=kittl`.
Configure `KITTL_GENERATOR_URL`, `KITTL_PROMPT_SELECTOR`,
`KITTL_GENERATE_SELECTOR`, and `KITTL_DOWNLOAD_SELECTOR`. Its browser profile
defaults to the ignored `.kittl-profile/` folder. Both browser integrations
require manual sign-in and do not bypass login, CAPTCHAs, or access controls.

The default Notion upload cap is 5 MiB per image for compatibility with free
workspaces; it can be raised up to the API's 20 MiB single-upload limit with
`NOTION_MAX_IMAGE_BYTES`.
