# FinPulse AI

Explore Apple, Microsoft and NVIDIA financial filings, select a historical cutoff and compare two evidence-backed readings of the same accounts. Select live on-device research or the original deterministic baseline. Every quantitative thesis links to the observations used to calculate it.

## Document research

Ask a qualitative question against three SEC annual filings using BM25, local semantic search, reciprocal rank fusion or a reranked hybrid shortlist. Each answer quotes dated filing evidence. For dollar amounts, ratios and growth rates, use the original SQL-backed financial tools. The upgrade includes 643 passages, a pinned vector index and a reviewed 20-question evaluation. See [Document research](DOCUMENT_RESEARCH.md) for the sources, setup, API, evaluation design and telemetry contract.

## Run the demo

On the verified Mac, use the existing project directory. The launcher loads the bundled SEC snapshot into a dedicated demo database, starts or reuses the model service and opens Streamlit. The model and CPU runtime have already been prepared on this machine.

```sh
cd "/Users/shriyahr/Documents/Codex/2026-11-25/new-chat-2/finpulse"
bash run_demo.sh
```

For a freshly extracted copy, run `bash setup_local_model.sh` first. This downloads a checksum-verified 986 MB Qwen model and builds llama.cpp locally with CMake and a C++ compiler. The download and build stay in .runtime/. The offline path is `bash run_demo.sh --offline`. Existing databases can be selected with DATABASE_URL.

Select **AAPL**, set the filing cutoff to **2024-12-31**, then choose **Bull vs Bear → Run research**. Read the opposing theses and download the Markdown report. The included observations show annual sales growth of about 2.02% alongside a net income decline of about 3.36%. Choose another cutoff to see how evidence availability changes.

For a fresh environment, use Python 3.13 on macOS. The lock records the versions of the runtime and transitive dependencies used here. A clean online installation was unavailable for verification.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python pipeline.py demo
python evaluate.py
python benchmark.py
python -m streamlit run dashboard.py
```

Start the optional API in a separate terminal using the same environment and working directory. Visit the local FastAPI /docs page to explore all eight routes.

```sh
python -m uvicorn api:app --port 8000
python -m pytest -q
```

## Architecture

```mermaid
flowchart TD
    A[SEC companyfacts] --> C[Bronze source records]
    B[Bundled SEC excerpts] --> C
    C --> D[Silver dated facts]
    D --> E[Gold metrics and evidence]
    K[SEC annual filing HTML] --> L[Section chunks and local index]
    L --> M[Rank and verify cited excerpts]
    M --> H
    M --> I
    E --> F[Bull and Bear agents]
    F --> G[Numeric verification and cross-examination]
    G --> H[Report and API]
    G --> I[EvalIQ JSON Lines]
    E --> J[Streamlit dashboard]
    H --> J
```

```mermaid
erDiagram
    COMPANIES ||--o{ FINANCIAL_FACTS : reports
    SOURCE_RECORDS ||--o{ FINANCIAL_FACTS : supplies
```

Ingestion runs record successes, rejections and errors. Research sessions persist the complete report and claim evidence IDs in JSON. Evaluation runs record separate result documents. Gold metrics are computed per cutoff, so requesting a historical report does not reuse a later restatement. Agent runs are recorded as JSON Lines.

## Data and reproducibility

The snapshots directory contains selected records retrieved from the official SEC companyconcept API. The excerpts are incomplete and each file identifies its source URL and capture time. The online path stores the complete companyfacts response, issuer, URL, retrieval timestamp and content-derived ID. After validating unit, reporting period, form and filing date, a single transaction saves each source and its facts. Repeating ingestion adds a run record while preserving the fact count. One SEC accession can report multiple periods.

Set SEC_USER_AGENT to a descriptive project identifier and researcher contact. For routine use include your contact email. It pauses between requests, retries temporary failures, honors numeric Retry-After delays up to two minutes and reuses responses collected within one hour. Run one refresh process at a time. The default covers the three named issuers. The live verification fetched complete responses for all three issuers twice. See examples/sec-live-check.json for status, hashes and counts.

```sh
export SEC_USER_AGENT='Your organization your-real-contact@example.org'
DATABASE_URL="sqlite:///$PWD/.runtime/live-sec.db" python pipeline.py refresh AAPL MSFT NVDA
```

Official references: [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) · [Fair access](https://www.sec.gov/about/developer-resources).

## Storage

SQLite is the verified local backend. For PostgreSQL, install the optional driver in the virtual environment, set POSTGRES_PASSWORD to a local development password and use the matching SQLAlchemy URL. Use URL encoding when the password contains reserved characters. Compose binds the database to loopback. Run python pipeline.py init to initialize an empty database. The alternative SQL initialization script is sql/init_postgres.sql.

```sh
python -m pip install 'psycopg[binary]>=3.2,<4'
docker compose up -d db
export DATABASE_URL='postgresql+psycopg://finpulse:YOUR_PASSWORD@127.0.0.1:5432/finpulse'
python pipeline.py demo
```

## Analytics

| Metric | Computation |
|---|---|
| Revenue, earnings or operating income growth | 100 × (current / prior − 1). Matches the same fiscal duration approximately one year apart. Growth uses a positive prior denominator. |
| Operating and net margins | 100 × corresponding income / positive revenue for identical start and end dates |
| Free cash flow | Operating cash inflow less capital expenditure for the same period |
| Return on assets or equity | 100 × annual net income / average opening and closing balance. Both balances must be available and positive. |
| Liabilities to equity | Total liabilities / positive equity on the same date |

Missing inputs suppress the metric. Instant balances, quarters, annual flows and cumulative flows have separate classifications. The pipeline does not turn six- or nine-month totals into standalone quarters. Duration windows are documented in pipeline.period. The SQL directory contains eight business questions including year-end issuer rankings, revision selection, cash generation and filing lineage. Run 01 through 08 in order on the same connection.

## Research and evaluation

Live inference uses Qwen2.5-1.5B-Instruct Q4_K_M on a local llama.cpp CPU server. Bull and Bear independently receive the same dated annual filing packet. Each selects a metric, proposes its value and may pose a follow-up question. Python checks the citation, amount, sign, period and issuer. A matching numeric observation enters the report. The peer review stage makes one model call per side to question the opposing position. A fifth call orders the verified evidence and selects a limitation. Python renders the checked synthesis and labels a deterministic summary if the model omits or invents a reference.

The single-agent mode uses the same model and data in two calls, one for both positions and one for synthesis. Questions are model interpretations. Only structured numeric findings have been verified against the financial records. Some attempted questions are withheld or repetitive.

The sidebar shows available local models and the mode used. The API accepts `mode` values `deterministic`, `live` and `single_live` at POST /research. Live errors are surfaced explicitly, and offline mode remains selectable. A locally installed Ollama model can also be selected through the normal loopback service. Its chat adapter requires local model metadata. The verified runtime on this Mac is llama.cpp.

```sh
export DATABASE_URL="sqlite:///$PWD/.runtime/demo.db"
export TELEMETRY_PATH="$PWD/.runtime/benchmark-events.jsonl"
/opt/anaconda3/bin/python live_benchmark.py
```

Keep the model service running during the benchmark. All three workflows get the same issuer-specific question, fiscal cutoff, candidate metrics and filing evidence. There are three scored reports per workflow. The scoring script checks each attempted value against annual-report reference figures and separately checks citation resolution, directional coverage and uncertainty. Read PROJECT_STATUS.md for results and limitations. Downloadable evidence includes examples/live-report.md, examples/live-benchmark.json and examples/live-events.jsonl.

- **Financial QA** uses 12 fixed questions, including an absent forecast and a contradicted sales claim. Both deterministic extraction paths answered 12/12; 8/8 numeric answers matched their references.
- **Report comparison** uses the same evidence, cutoff and selector for each issuer. One analyst selects one observation. The paired workflow gets up to two observations plus verification and a challenge round. Both use the same four criteria per issuer. See examples/benchmark.json for all decisions, denominators and timings. The explicit downside criterion and extra observation budget favor broader reports. Three issuers cannot establish a general model-quality difference.
- **Telemetry** uses examples/telemetry.schema.json and examples/telemetry.jsonl. Each line is one independent event. Join traces using trace_id and parent_event_id. Wall-clock timestamps are UTC, latency is measured in milliseconds, and unavailable costs or token counts are null. No keys, prompts or responses are captured. Set TELEMETRY_PATH to an existing writable directory plus a filename. Serialize writers or give each process a separate file for multi-process deployments.

## Project map

| Files | Purpose |
|---|---|
| pipeline.py, snapshots/, sql/ | Ingestion, normalization, storage, financial tools and analyst SQL |
| research.py, live.py, api.py, dashboard.py | Bounded workflow, evidence verification, application entry points |
| evaluate.py, benchmark.py, live_benchmark.py, tests/ | Independent answers, paired workflow rubric and regression tests |
| examples/ | Actual report, measured evaluations, telemetry sample and JSON schema |
| PROJECT_STATUS.md, INTERVIEW_GUIDE.md | Verification record, limitations, résumé wording and interview prep |
| requirements.lock, compose.yaml, .env.example | Runtime versions and local configuration |

## Attribution

SEC data retains its source links and issuer attribution. Application code was created for this project. Key runtime dependencies include Pandas, Plotly.py, SQLAlchemy, FastAPI, HTTPX, Pydantic, Streamlit and Pytest. See THIRD_PARTY.md for license metadata from the installed distributions. This is an educational filing-research application.
