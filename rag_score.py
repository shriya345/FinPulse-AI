"""Score the frozen RAG run. New answer text requires new independent review."""

import hashlib
import json
from statistics import mean, median
from rag_docs import ROOT, corpus
from rag_retrieve import METHODS


def signature(row):
    key = [row["case"], row["answer"], [c["excerpt"] for c in row["citations"]]]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


def summarize(rows, judgments):
    chunks = {r["id"]: r for r in corpus()}
    cases = {
        r["id"]: r for r in json.loads((ROOT / "eval/document-cases.json").read_text())
    }
    groups = {}
    for split in ("all", "dev", "evaluation", "held-out"):
        methods = []
        for method in METHODS:
            rs = [
                r
                for r in rows
                if r["method"] == method and (split == "all" or r["split"] == split)
            ]
            positive = [r for r in rs if r["gold"]]
            negative = [r for r in rs if not r["gold"]]
            valid = support = citation_count = correct = 0
            for r in rs:
                case = cases[r["case"]]
                judgment = judgments[signature(r)]
                assert len(judgment["responsive_citations"]) == len(r["citations"])
                correct += int(judgment["fully_correct"])
                for c, direct in zip(r["citations"], judgment["responsive_citations"]):
                    source = chunks[c["chunk_id"]]
                    valid += int(
                        c["excerpt"] in source["text"]
                        and source["filed"] <= case["cutoff"]
                        and c["ticker"] == case["ticker"]
                        and all(
                            c[k] == source[k]
                            for k in ("accession", "section", "filed", "url", "period")
                        )
                    )
                    citation_count += 1
                    support += int(direct)
            methods.append(
                dict(
                    method=method,
                    queries=len(rs),
                    answerable=len(positive),
                    precision_at_3=mean(r["hits"] / 3 for r in positive),
                    recall_at_3=mean(r["hits"] / r["gold"] for r in positive),
                    mrr_at_3=mean(r["rr"] for r in positive),
                    relevant_retrieved=sum(r["hits"] for r in positive),
                    gold_chunks=sum(r["gold"] for r in positive),
                    correct=correct,
                    citations=citation_count,
                    valid=valid,
                    reviewed_support=support,
                    appropriate_abstentions=sum(
                        r["status"] == "abstained" for r in negative
                    ),
                    unanswerable=len(negative),
                    median_retrieval_ms=median(r["retrieval_ms"] for r in rs),
                    median_end_to_end_ms=median(r["end_to_end_ms"] for r in rs),
                )
            )
        groups[split] = methods
    return groups


if __name__ == "__main__":
    rows = json.loads((ROOT / "eval/document-runs.json").read_text())
    review = json.loads((ROOT / "eval/document-review.json").read_text())
    (ROOT / "eval/document-summary.json").write_text(
        json.dumps(summarize(rows, review), indent=2) + "\n"
    )
