# Filing research

FinPulse can answer a qualitative filing question alongside a financial report or in its Document Research tab. The company and filing cutoff apply to every returned passage. The answer is extractive. Qwen selects supporting sentences and proposes exact answer spans. The application displays those words with numbered citations. If a proposed span cannot be matched inside the selected source sentence, it displays that sentence and records the rendering choice. The system can abstain, return a partial answer, or show retrieved passages alone in offline mode.

## Corpus

| Ticker | 10-K accession | Reporting period ended | Filed | Business / Risk / MD&A chunks |
|---|---|---|---|---|
| AAPL | 0000320193-24-000123 | 2024-09-28 | 2024-11-01 | 20 / 87 / 21 |
| MSFT | 0000950170-24-087843 | 2024-06-30 | 2024-07-30 | 87 / 96 / 68 |
| NVDA | 0001045810-24-000029 | 2024-01-28 | 2024-02-21 | 67 / 142 / 55 |

The filing index pages supplied the accession, filing date and reporting period. Three compressed original HTML responses are included. The manifest records the SEC URL, retrieval time and SHA-256 checksum computed over the uncompressed response bytes. An independent repeat fetch matched all three checksums. Source links are in [the manifest](filings/manifest.json).

The parser follows the original table-of-contents anchors to delimit Items 1, 1A and 7. It omits hidden elements, scripts and style data, normalizes whitespace and removes recurring page labels. It preserves substantive table text in reading order. Flattening a table can lose its visual layout, so the numeric research path continues to use aligned XBRL facts. Each section is split into 145-word windows with a 32-word overlap. This permits context around a boundary while keeping passages short enough for local inference. To experiment, use python rag_retrieve.py prepare --chunk-size 180 --overlap 40 --output .runtime/alternative.npz, then set FINPULSE_DOCUMENT_INDEX to that file before launch. The shipped catalog and benchmark labels describe the default configuration. A change requires rebuilding the vector index and relabeling affected gold IDs. SEC table-of-contents variations outside these three filings are untested.

A chunk ID hashes the accession, raw document checksum, section, starting word and text. Duplicate inputs therefore yield one set of IDs. Overlap intentionally preserves distinct neighboring chunks. An index fingerprint checks the ordered IDs and encoder revision on load. Query filters run before scoring. BM25 document frequencies are computed only over eligible documents. Cutoffs use filing dates at calendar-day precision, with availability assumed from the end of that day.

## Retrieval and generation

- **BM25** ranks distinctive matching terms using term frequency, inverse document frequency and length normalization. Parameters are k1 = 1.2 and b = 0.75.
- **Semantic retrieval** uses the pinned local all-mpnet-base-v2 model to turn passages and questions into 768-dimensional vectors. Mean-pooled unit vectors are compared by dot product, equivalent to cosine similarity here. NumPy scans eligible vectors. The compressed matrix is bundled.
- **Hybrid retrieval** combines the keyword and semantic lists by reciprocal rank fusion. Each item gains 1 / (60 + rank) from each of the top-40 lists. This avoids adding scores with incompatible scales.
- **Reranking** takes the top 12 fused passages and scores each question–passage pair jointly with the pinned local ms-marco-MiniLM-L-6-v2 cross-encoder. It can change their order but cannot recover a passage absent from the shortlist. All methods return at most three passages by default.
- **Answer generation** separately scores candidate sentences inside those passages, using the same cross-encoder and an issuer prefix. Up to eight candidates at logit ≥ 0 enter a constrained local Qwen call. For two explicit interrogatives joined by “and”, each part can get its own sentence selection from the same retrieved passages. The four benchmarks share this answer stage. Reranking the passage shortlist is an additional operation.
- **Evidence boundaries** keep official figures and arithmetic in the SQL pipeline. The document service checks issuer, filing cutoff, source identity and exact excerpt. The cross-encoder estimates topical relevance. Neither a matching excerpt nor a high relevance score proves the passage answers the question. The review file scores usefulness separately.
- **Untrusted text** enters as delimited data. The model has no execution tools or credentials. Known instruction-override strings are rejected or excluded before generation. Only selected source text can be published as the factual answer. This reduces instruction-following risk while allowing an irrelevant passage or incomplete interpretation to slip through. Injection fixtures are separate from genuine filings.

## Run

From the project's directory on the verified Mac:

```sh
cd "/Users/shriyahr/Documents/Codex/2026-11-25/new-chat-2/finpulse"
bash run_demo.sh
```

Select Document Research and a filing cutoff. For example, select MSFT, December 31, 2024, and ask which companies Microsoft names as rivals for Azure Security. Select a method and choose Research the filing. The research perspective can be Analyst, Bull or Bear. Optional filing evidence on the Bull vs Bear tab adds separately cited qualitative findings to the saved report. The standalone API is POST /research/documents. POST /research accepts an optional document_question.

For a new environment use the Python runtime dependencies from requirements.lock and requirements-rag.txt. The two retrieval checkpoints and the Qwen weights are separate from the archive. The verified Mac already has both checkpoints cached and the CPU model runtime prepared.

```sh
python -m pip install -r requirements.lock -r requirements-rag.txt
python rag_retrieve.py prepare --download-models
bash setup_local_model.sh
bash run_demo.sh
```

For a new SEC acquisition set a descriptive SEC_USER_AGENT with real contact details, then run python rag_ingest.py /path/to/an/empty/staging-directory. Acquisition runs sequentially, pauses between requests and validates filing metadata and checksums. The preserved 4,288-observation financial SQLite archive is financial_archive/live-sec.sqlite.gz. Decompress into a new destination when a separate copy is needed. The 1,366-observation launcher demo and existing working database retain their separate identities.

## Reproduce the research evaluation

With the model service running, execute the following from the project directory. Run on a working copy if you wish to preserve the shipped measurements. The evaluation script writes new measured outputs. The score script requires a review entry matching each question, answer and cited excerpt.

```sh
TELEMETRY_PATH="$PWD/.runtime/replay-rag.jsonl" /opt/anaconda3/bin/python rag_eval.py
/opt/anaconda3/bin/python rag_score.py
/opt/anaconda3/bin/python -m pytest -q
```

The gold set was authored from the source sections before running each query. Labels resolve independently chosen substantive phrases to chunk IDs. A duplicated or paraphrased passage outside that pool may answer correctly and still score as a retrieval miss. For answerable questions, precision@3 is relevant hits divided by three, recall@3 is relevant hits divided by labeled relevant chunks, and MRR@3 is the reciprocal rank of the first labeled hit or zero. Each question receives equal weight. Unanswerable questions are scored separately for abstention. Answer correctness requires all requested facts or a justified abstention. Citation validity checks metadata, cutoff, and verbatim source membership. Reviewed citation support asks whether the excerpt directly supports a responsive fact, even when a multi-part answer is incomplete.

The 20 cases include six development cases, 12 evaluation cases inspected during iteration, and two additional questions kept aside for the final run. The latter are a small holdout. The four retrieval methods ran sequentially against the same final cases and local service. Timings are single-machine observations that include cache and warm-up effects. Repeated query embeddings may be cached. End-to-end latency includes retrieval, sentence ranking, possible Qwen generation and validation. Source-based answer and citation judgments were reviewed by the project assistant without a second independent human annotator.

## EvalIQ retrieval event contract

Existing schema_version 1.0 is preserved. A tool_call event stores the retrieval method in tool_name (rag.bm25, rag.semantic, rag.rrf or rag.rrf_rerank). evaluation_case_id is a stable query identifier derived from question, issuer and cutoff. Ordered evidence_ids are the returned chunk IDs. Their count is the returned depth. The bundled filings/chunk-catalog.json.gz maps every ID to its accession, section, reporting period, filing date, URL and checksum. latency_ms is measured wall time. status is ok, missing or error, with error_type on failure. Model calls and the final response use the same trace. No raw user query, prompts or completions are included in JSON Lines.

An EvalIQ consumer can derive p50/p95 retrieval latency by method, zero-hit fraction, returned depth, and source diversity after joining the catalog. Precision, recall, reciprocal rank and answer support require external labels or review. No changes to EvalIQ are required to import the stream.

## Measured results

All methods answered the same 20 questions, including 14 answerable and six negative cases. Each answerable question contributed three retrieval slots. Correct answers include justified abstentions. Citation support was reviewed per cited subquestion. Fresh-query retrieval latency was measured in a separate pass after warming the models and clearing the query-embedding cache before each search. End-to-end medians use the original sequential answer runs.

| Method | Precision@3 | Recall@3 | MRR@3 | Fully correct | Citations valid | Citations responsive | Appropriate abstentions | Fresh-query median | End-to-end median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.381 | 0.845 | 0.750 | 19/20 | 15/15 | 15/15 | 6/6 | 0.3 ms | 1.135 s |
| Semantic | 0.238 | 0.536 | 0.536 | 14/20 | 13/13 | 10/13 | 6/6 | 27.9 ms | 0.737 s |
| Hybrid | 0.381 | 0.833 | 0.881 | 18/20 | 16/16 | 14/16 | 6/6 | 27.5 ms | 0.805 s |
| Hybrid with reranking | 0.405 | 0.905 | 0.845 | 18/20 | 16/16 | 14/16 | 6/6 | 121.5 ms | 0.916 s |

Reranking improved labeled recall and added about 94 ms to median hybrid retrieval. It did not improve full answer correctness on this set. BM25 distinguished GeForce NOW from GeForce Experience. Every method answered both parts of the Mellanox and CUDA question. None fully answered both parts of the Apple advertising question. On the two strictly held-out questions BM25 answered 2/2, while each other method answered 1/2. Reference labels are sparse, and the two-item holdout is too small to support model-selection claims.

The case file, individual answers, content-keyed reviewer judgments, aggregate metrics and fresh-query timings are in eval/. The original run emitted 80 retrieval events, 60 actual local Qwen calls and 80 final responses. EvalIQ accepted all 220 events with zero invalid records and counted 220 duplicates on replay. The historical 49-event financial benchmark also imported successfully.
