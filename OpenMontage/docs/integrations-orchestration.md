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
- **Tess, Canva, Midjourney, and Kittl:** credentials and API availability are
  account/product dependent. Confirm official API access, scopes, and terms
  before implementing an adapter; these environment variables do not imply
  that API access is included in a plan.
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
calls may incur provider charges. Midjourney remains manual or requires an
authorized provider integration; the pipeline does not automate a user Discord
account. Kittl can be selected explicitly with `IMAGE_GENERATION_PROVIDER=kittl`.
Kittl uses
Playwright only through the visible UI and normal browser downloads. Enable it
only when permitted by the account terms, then configure the Kittl page and
selectors in `.env`. Sign in manually once using a local persistent browser
profile:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m generators.kittl_automation --login
```

Set `KITTL_GENERATOR_URL`, `KITTL_PROMPT_SELECTOR`,
`KITTL_GENERATE_SELECTOR`, and `KITTL_DOWNLOAD_SELECTOR` to match the controls
in the authorized Kittl interface. The browser profile defaults to the ignored
`.kittl-profile/` folder; do not share or commit it. The automation does not
bypass login, CAPTCHAs, or access controls.
The default Notion upload cap is 5 MiB per image for compatibility with free
workspaces; it can be raised up to the API's 20 MiB single-upload limit with
`NOTION_MAX_IMAGE_BYTES`.
