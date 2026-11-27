# Third-party software

Installed distribution metadata for the primary dependencies. Consult the license shipped with each distribution for its terms.

- pandas 2.2.3 — BSD License
- sqlalchemy 2.0.39 — MIT License
- pydantic 2.10.3 — MIT License
- httpx 0.28.1 — BSD License
- fastapi 0.136.0 — MIT
- streamlit 1.45.1 — Apache Software License
- plotly 5.24.1 — MIT License
- pytest 8.3.4 — MIT License
- uvicorn 0.45.0 — BSD-3-Clause

## Optional local inference

The project-local setup uses llama.cpp b5430 (MIT) and the Qwen2.5-1.5B-Instruct model (Apache-2.0) quantized by Bartowski. The archive contains the setup script. Model weights, binaries and third-party source are downloaded separately. The setup verifies SHA-256 before use.

[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/b5430) · [Original model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Quantized weights](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF)

## Document retrieval

Local retrieval uses Sentence Transformers, Transformers, PyTorch, lxml and NumPy. The project records model repository names and commit revisions in rag_retrieve.py. Retrieval model weights remain separate downloads. The release contains the locally computed vectors and the public annual filing snapshots with SEC provenance.

[MPNet sentence encoder](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) · [MS MARCO MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
