# FinPulse hybrid RAG release

Verified on November 26, 2026, on the existing Apple Silicon Mac. Three FY2024 SEC filings were fetched, archived and refetched with matching checksums. Nine sections yielded 643 chunks. The original financial source hashes matched upon restoring all 4,288 verified observations. The archived financial database is included separately. The demo still loads 1,366 observations.

The final evaluation executed 80 question–method runs, including 60 live Qwen calls. BM25 answered 19/20 correctly, semantic 14/20, and both hybrid variants 18/20. Reranked hybrid had the largest labeled recall@3, 90.5%. All methods abstained on six of six negative cases. A true citation could still answer about the wrong product or only part of a question. Two questions constituted the final holdout. The other 18 were inspected during development or evaluation iteration.

The complete test suite has 20 passing tests. The prior dashboard test now expects the explicitly requested fifth tab. Financial-agent benchmarks, SQL scripts, financial snapshots and earlier examples are preserved. Both the old 49-event stream and the new 220-event stream imported through the existing EvalIQ CLI, and the new stream deduplicated on replay.

See [Document research](DOCUMENT_RESEARCH.md) for exact startup instructions, corpus metadata, metric denominators, independent source judgments, failure cases and telemetry mappings. PostgreSQL remains optional and its earlier verification status is unchanged.

# Financial workflow verification

Verified on November 25, 2026 on the user's arm64 Mac running macOS 26.2.

## Completed and tested

- Qwen2.5-1.5B-Instruct Q4_K_M ran locally through a project-scoped CPU build of llama.cpp b5430. The downloaded weights matched SHA-256 1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370. No hosted credentials or paid credits were used.
- Two independent live agents receive actual annual financial evidence and propose dated numeric findings. Each sees the verified positions during one reciprocal challenge round. A final inference orders the supporting metrics and chooses a limitation. Numeric values, signs, citations and synthesis references are checked before rendering. Rejected text and an explicitly labeled verified summary are supported. A qualitative question can still be weak or contain a misleading premise.
- The independent offline baseline, existing SQL examples and API routes remain available. Twelve automated tests passed, including nine existing cases and three focused regressions covering live orchestration, evidence rejection, malformed responses, usage/error telemetry, fallback access and invalid synthesis references. Pyflakes passed.
- The original 15-file demo loaded 1,366 distinct filing observations. A pre-existing working finpulse.db contained 1,442 observations across 17 source records before this upgrade. The launcher uses a dedicated .runtime/demo.db. The original working database was left in place. Tests execute against separate temporary databases.
- Live SEC requests returned HTTP 200 for all three companies on two fetches. Each second response matched the original raw response. Counts remained 1,458 for Apple, 1,580 for Microsoft and 1,250 for NVIDIA, or 4,288 total. The isolated verification database retained source URLs, complete response text, UTC retrieval times and content-derived identifiers. See examples/sec-live-check.json.
- Streamlit served its health endpoint on loopback. The Streamlit application test rendered all four tabs, selected live mode, ran an Apple research session, and reached cross-examination without a displayed error. The offline dashboard test also passed.
- The real EvalIQ importer ingested all 49 final benchmark events into a fresh SQLite database with zero invalid events. Re-import returned 49 duplicates, zero new events and zero invalid events. The live records conform to the existing v1.0 contract. Only live model calls carry provider token counts.

## Measured benchmark

Each workflow answered the same report question for AAPL, MSFT and NVDA using the same frozen FY2024 candidate evidence as available by December 31, 2024. The deterministic workflow and both LLM workflows can issue one strength and one downside per issuer or abstain. The same Qwen weights, greedy decoding settings and evidence packet were used across both live modes. Single-agent uses two model calls per issuer. Bull/Bear uses five. Constrained selection limits the candidate metric names on each side.

| Workflow | Correct numeric proposals | Valid metric citations | Unsupported numeric proposals | Report coverage | Median elapsed per report | Input/output tokens, total | Provider fee |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic | 4/4 | 4/4 | 0 | 9/9 | 4.0 ms | No model calls | $0 |
| Single LLM | 4/4 | 4/4 | 0 | 9/9 | 3.32 s | 1,425 / 389 | $0 |
| Bull/Bear LLM | 4/4 | 4/4 | 0 | 9/9 | 8.40 s | 3,411 / 947 | $0 |

Numeric scoring compares proposed values with independent annual-report figures at the displayed precision. Citation scoring checks the proposed metric's source records, issuer and filing cutoff. Coverage checks a supported strength or appropriate abstention, a supported downside or appropriate abstention, and an expression of uncertainty. Metric names resolve to the original filing record IDs in the evidence register.

The coverage rubric excludes challenge quality. The verifier withheld four of six proposed challenges in the paired run for an invalid citation, incorrect target, quantitative prose or failure to return a question. All three final paired syntheses passed the evidence check. The small number of issuers and checked numbers, fixed model and shared development cases limit generalization. Latency is one measured run per workflow and issuer with a running local service. Token totals are provider-reported. Provider fees exclude hardware and electricity. No approach exhibited better measured numeric accuracy.

## Partial or unavailable

- Ollama's standard service could not create its default configuration in this restricted session. Its standalone runner loaded the weights but could not initialize Metal. Apple's built-in Foundation Models API reported availability but returned a model service error during inference. Both informed the decision to verify CPU-only llama.cpp. The alternate standard Ollama chat adapter has no end-to-end verification here.
- No local PostgreSQL server or client was discovered. Port 5432 refused connections and the installed Docker client reported no running daemon. PostgreSQL schema, migrations and live duplicate prevention remain unverified. SQLite persistence and idempotency passed.
- The model snapshot requires a separate download for a new computer. The release ZIP excludes binaries and weights. The local setup script verified the model and rebuilt the runtime in the existing Mac environment. A clean Python environment on a second machine has not been tested.

## Reproduction

```sh
cd "/Users/shriyahr/Documents/Codex/2026-11-25/new-chat-2/finpulse"
bash run_demo.sh
```

For offline mode run `bash run_demo.sh --offline`. For a freshly extracted copy, run `bash setup_local_model.sh` once before launching live mode. Use another terminal in the same directory to run the following while the model service is up.

```sh
export DATABASE_URL="sqlite:///$PWD/.runtime/demo.db"
export TELEMETRY_PATH="$PWD/.runtime/benchmark-events.jsonl"
/opt/anaconda3/bin/python live_benchmark.py
/opt/anaconda3/bin/python -m pytest -q
```

The measured cases, event stream, live report and SEC request results are under examples/. The original deterministic evaluations remain in examples/evaluation.json and examples/benchmark.json.
