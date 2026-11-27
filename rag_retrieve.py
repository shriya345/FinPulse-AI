"""Local BM25, unit-normalized sentence embeddings, RRF and optional reranking."""

import functools
import hashlib
import math
import os
import re
import tempfile
from collections import Counter
from datetime import date
from time import perf_counter
from pathlib import Path

import numpy as np

from rag_docs import ROOT, corpus

ENCODER = "sentence-transformers/all-mpnet-base-v2"
ENC_REV = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
RANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RANK_REV = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
METHODS = ("BM25", "Semantic", "Hybrid", "Hybrid + reranker")
STOP = set(
    "a an and are as at be by did do does for from how in is it of on or that the their to was what which why with company's company apple microsoft nvidia".split()
)


def terms(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


@functools.lru_cache(maxsize=2)
def model(kind, download=False):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(4)
    if kind == "encoder":
        return SentenceTransformer(
            ENCODER, revision=ENC_REV, device="cpu", local_files_only=not download
        )
    return CrossEncoder(
        RANKER, revision=RANK_REV, device="cpu", local_files_only=not download
    )


class Index:
    def __init__(self, rows, vectors=None):
        self.rows = rows
        self.vectors = vectors
        self.counts = [Counter(terms(r["text"])) for r in rows]
        self.lengths = [sum(c.values()) for c in self.counts]

    def search(self, question, ticker, cutoff, method, k=3):
        cutoff = date.fromisoformat(cutoff).isoformat()
        if method not in METHODS or not 1 <= k <= 10:
            raise ValueError("Invalid retrieval options")
        start = perf_counter()
        eligible = [
            i
            for i, r in enumerate(self.rows)
            if r["ticker"] == ticker and r["filed"] <= cutoff
        ]

        def ranked(scores):
            return sorted(scores, key=lambda i: (-scores[i], self.rows[i]["id"]))

        if not eligible or not question.strip():
            return [], (perf_counter() - start) * 1000
        if method != "Semantic":
            q = set(terms(question))
            n = len(eligible)
            avg = sum(self.lengths[i] for i in eligible) / n
            idf = {
                t: math.log1p((n - df + 0.5) / (df + 0.5))
                for t in q
                if (df := sum(t in self.counts[i] for i in eligible))
            }
            bm = {}
            for i in eligible:
                norm = 1.2 * (0.25 + 0.75 * self.lengths[i] / avg)
                score = sum(
                    v * 2.2 * tf / (tf + norm)
                    for t, v in idf.items()
                    if (tf := self.counts[i][t])
                )
                if score:
                    bm[i] = score
            scores = bm
        if method != "BM25":
            if self.vectors is None:
                raise ValueError(
                    "Build the embedding index with python rag_retrieve.py prepare"
                )
            vector = query_vector(question)
            dense = dict(zip(eligible, (self.vectors[eligible] @ vector).tolist()))
            scores = dense
        if method.startswith("Hybrid"):
            scores = Counter()
            for ranking in (ranked(bm), ranked(dense)):
                for rank, i in enumerate(ranking[:40], 1):
                    scores[i] += 1 / (60 + rank)
        order = ranked(scores)
        if method == METHODS[3] and order:
            shortlist = order[:12]
            values = model("ranker").predict(
                [(question, self.rows[i]["text"]) for i in shortlist],
                batch_size=8,
                show_progress_bar=False,
            )
            scores = dict(zip(shortlist, map(float, values)))
            order = ranked(scores)
        return [dict(self.rows[i], score=float(scores[i])) for i in order[:k]], (
            perf_counter() - start
        ) * 1000


@functools.lru_cache(maxsize=128)
def query_vector(query):
    return model("encoder").encode(
        query, normalize_embeddings=True, show_progress_bar=False
    )


def fingerprint(rows):
    return hashlib.sha256(
        (ENC_REV + "|" + "|".join(r["id"] for r in rows)).encode()
    ).hexdigest()


def load_index(path=None):
    path = Path(
        path or os.getenv("FINPULSE_DOCUMENT_INDEX") or ROOT / "filings/vectors.npz"
    )
    if not path.exists():
        return Index(corpus())
    with np.load(path, allow_pickle=False) as data:
        rows = corpus(
            size=int(data.get("chunk_size", 145)), overlap=int(data.get("overlap", 32))
        )
        if str(data["key"]) != fingerprint(rows) or data["vectors"].shape != (
            len(rows),
            768,
        ):
            raise ValueError("Embedding index differs from filing corpus")
        return Index(rows, data["vectors"])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare"])
    p.add_argument("--download-models", action="store_true")
    p.add_argument("--chunk-size", type=int, default=145)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--output", type=Path, default=ROOT / "filings/vectors.npz")
    a = p.parse_args()
    rows = corpus(size=a.chunk_size, overlap=a.overlap)
    enc = model("encoder", a.download_models)
    model("ranker", a.download_models)
    path = a.output
    if path.exists():
        existing = load_index(path)
        if fingerprint(existing.rows) != fingerprint(rows):
            p.error("Choose a new --output for a different chunk configuration.")
        print("Verified existing index", len(rows))
        raise SystemExit
    vectors = enc.encode(
        [r["text"] for r in rows],
        batch_size=12,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        np.savez_compressed(
            f,
            key=fingerprint(rows),
            vectors=vectors,
            chunk_size=a.chunk_size,
            overlap=a.overlap,
        )
    os.replace(f.name, path)
    print("Indexed", len(rows), vectors.shape)
