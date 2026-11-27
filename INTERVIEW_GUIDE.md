# FinPulse AI interview guide

**1. What problem does FinPulse solve?** An analyst may ask both “What was the operating margin?” and “What risks did management describe?” SQL supplies the margin from aligned financial records. Retrieval-augmented generation, or RAG, finds relevant filing text and lets a local model answer with cited excerpts.

**2. How did you keep filing evidence reproducible?** I kept the original SEC HTML, accession, filing date, reporting period, URL, retrieval time and checksum. The parser follows each filing’s own links to find Business, Risk Factors and MD&A. I filter by filing date so an earlier report uses only information already filed.

**3. What is BM25?** It scores a passage by the query words it contains. A rare name such as Mellanox matters more than a common word. Repeating a word has diminishing benefit, and the formula accounts for passage length.

**4. What is an embedding?** It is a list of numbers a model produces to represent meaning. I used a local MPNet encoder to turn each passage into 768 numbers. A question about dependence on factories can resemble a passage on contract manufacturing even when their wording differs.

**5. How does vector similarity work?** The encoder places related text in similar directions. I normalize the vectors to unit length, then take their dot product to measure that alignment. Scanning 643 vectors with NumPy was practical on this Mac.

**6. Why use hybrid retrieval and reciprocal rank fusion?** Keyword matching helps with names and precise terms. Embeddings help with paraphrases. Reciprocal rank fusion combines list positions, adding 1 / (60 + rank) for each appearance. In the benchmark hybrid returned the first labeled hit earliest on average, while BM25 answered more questions correctly.

**7. What does reranking do?** A cross-encoder reads the question together with each shortlisted passage and scores the pair. I reranked 12 candidates. Recall@3 rose from 83.3% to 90.5% compared with hybrid retrieval, but full answer correctness stayed at 18/20.

**8. How do precision, recall and MRR differ?** If two of the top three passages are relevant, precision@3 is two thirds. If four labeled passages exist and the search finds two, recall@3 is one half. If the first relevant hit is second, the reciprocal rank is one half. MRR averages that last quantity across questions.

**9. How did you detect hallucinations and weak answers?** I resolve the citation to its dated source and display an exact source span or sentence. I separately review whether that passage answers the question. In the final test a genuine NVIDIA citation described GeForce Experience when the question asked about GeForce NOW. The excerpt was real and the answer was wrong. Retrieved text is untrusted, the model has no action tools, and questions can receive an insufficient-evidence result.

**10. What do the evaluation and telemetry establish?** Four methods answered the same 20 questions in 80 runs. Six queries deliberately lacked usable evidence. Only two questions were reserved for the final holdout, so the ranking is descriptive. EvalIQ imported all 220 RAG events and identified every duplicate on replay. Earlier financial-agent benchmarks remain available separately.

## Three résumé bullets

- Built a Python and SQL financial research pipeline that verified repeatable ingestion of 4,288 SEC financial observations and indexed 643 passages from Apple, Microsoft and NVIDIA annual filings with filing-date filters and source provenance.
- Integrated local Qwen inference into a Streamlit and FastAPI research application with BM25, semantic vector search, reciprocal rank fusion, cross-encoder reranking and source-cited qualitative answers alongside verified financial calculations.
- Evaluated four retrieval methods across 80 question–method runs, measured 90.5% labeled recall@3 for reranked hybrid retrieval, and verified ingestion and deduplication of 220 RAG events in EvalIQ.
