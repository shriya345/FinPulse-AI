"""Measured retrieval and live Qwen responses against pre-labeled SEC passages."""

import json
from rag import document_research, index
from rag_docs import ROOT
from rag_retrieve import METHODS


def measure(case, hits, k):
    gold = set().union(*(set(g["chunk_ids"]) for g in case["gold"]))
    ranks = [i for i, r in enumerate(hits, 1) if r["id"] in gold]
    return dict(hits=len(ranks), gold=len(gold), k=k, rr=1 / ranks[0] if ranks else 0)


def run():
    cases = json.loads((ROOT / "eval/document-cases.json").read_text())
    idx = index()
    out = []
    for case in cases:
        for method in METHODS:
            r = document_research(
                case["question"],
                case["ticker"],
                case["cutoff"],
                method,
                idx=idx,
                fixture=case["injected_context"],
            )
            out.append(
                dict(
                    case=case["id"],
                    split=case["split"],
                    method=method,
                    **measure(case, r["passages"], 3),
                    status=r["status"],
                    answer=r["answer"],
                    citations=r["citations"],
                    retrieved=[h["id"] for h in r["passages"]],
                    retrieval_ms=r["retrieval_ms"],
                    end_to_end_ms=r["end_to_end_ms"],
                    trace_id=r["trace_id"]
                )
            )
        print(case["id"], flush=True)
    (ROOT / "eval/document-runs.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    run()
