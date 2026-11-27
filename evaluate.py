"""Fixed reference answers transcribed from the FY2024 consolidated statements."""

import json
from statistics import mean
from time import perf_counter
from uuid import uuid4

from pipeline import database, evaluations, put
from research import Claim, emit, evidence, verify_numeric_claim

APPLE = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
MS = "https://www.sec.gov/Archives/edgar/data/789019/000119312524242888/d815777dars.pdf"
# Values in dollars, percentages, or issuer symbols as specified by each task.
CASES = [
    ("Apple revenue", "AAPL", "revenue", 391035e6, APPLE),
    ("Apple earnings", "AAPL", "net_income", 93736e6, APPLE),
    ("Apple operating cash flow", "AAPL", "operating_cash", 118254e6, APPLE),
    ("Apple capex", "AAPL", "capex", 9447e6, APPLE),
    ("Apple free cash flow", "AAPL", "free_cash_flow", 108807e6, APPLE),
    ("Apple net margin, percent", "AAPL", "net_margin", 100 * 93736 / 391035, APPLE),
    ("Microsoft revenue", "MSFT", "revenue", 245122e6, MS),
    ("Microsoft net income", "MSFT", "net_income", 88136e6, MS),
    (
        "Which issuer had higher FY2024 revenue?",
        "both",
        "compare",
        "AAPL",
        APPLE + " " + MS,
    ),
    (
        "Does evidence include a forecast share price?",
        "AAPL",
        "target_price",
        None,
        None,
    ),
    (
        "Does Apple's revenue accession identify Apple's CIK?",
        "AAPL",
        "attribution",
        True,
        APPLE,
    ),
    ("Does the filing support annual sales of $1?", "AAPL", "challenge", False, APPLE),
]


def evaluate(engine):
    result = {}
    for mode in ["single extraction", "extraction with numeric verification"]:
        rows = []
        sid = str(uuid4())
        with engine.connect() as conn:
            pools = {t: evidence(conn, t, "2024-12-31") for t in ("AAPL", "MSFT")}

            def lookup(t, m):
                return next(
                    (
                        r
                        for r in pools[t]
                        if r["end"] == {"AAPL": "2024-09-28", "MSFT": "2024-06-30"}[t]
                        and r["kind"] == "annual"
                        and r["metric"] == m
                    ),
                    None,
                )

            for title, t, m, expected, source in CASES:
                start = perf_counter()
                records = []
                if m == "compare":
                    records = [lookup(i, "revenue") for i in pools]
                else:
                    records = [
                        lookup(t, "revenue" if m in {"challenge", "attribution"} else m)
                    ]
                records = [r for r in records if r]
                answer = None
                if records:
                    r = records[0]
                    if m == "compare" and len(records) == 2:
                        answer = max(records, key=lambda v: v["value"])["ticker"]
                    elif m == "attribution":
                        answer = r["accession"].startswith("0000320193")
                    elif m == "challenge":
                        answer = r["value"] == 1
                    else:
                        answer = r["value"]
                if mode.startswith("extraction with") and any(
                    verify_numeric_claim(
                        conn, Claim.model_validate(r), "2024-12-31", pools[r["ticker"]]
                    )
                    != "Supported"
                    for r in records
                ):
                    answer = None
                elapsed = (perf_counter() - start) * 1000
                correct = (
                    abs(answer - expected) < 1e-6
                    if type(answer) in (int, float) and type(expected) in (int, float)
                    else answer == expected
                )
                ids = [i for r in records for i in r["evidence_ids"]]
                emit(
                    session_id=sid,
                    trace_id=sid,
                    event_type="evaluation",
                    evaluation_case_id=m + "-" + t,
                    provider="deterministic",
                    latency_ms=elapsed,
                    evidence_ids=ids,
                )
                rows.append(
                    dict(
                        question=title,
                        expected=expected,
                        actual=answer,
                        correct=correct,
                        reference=source,
                        latency_ms=elapsed,
                        evidence_ids=ids,
                    )
                )
        numeric = [r for r in rows if type(r["expected"]) in (int, float)]
        issued = [r for r in rows if r["actual"] is not None]
        result[mode] = dict(
            correct=sum(r["correct"] for r in rows),
            total=len(rows),
            numeric_correct=sum(r["correct"] for r in numeric),
            numeric_total=len(numeric),
            supported=sum(r["correct"] for r in issued),
            unsupported=sum(not r["correct"] for r in issued),
            answers_issued=len(issued),
            mean_latency_ms=mean(r["latency_ms"] for r in rows),
            cases=rows,
        )
    with engine.begin() as conn:
        put(conn, evaluations, dict(id=str(uuid4()), result=json.dumps(result)))
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(database()), indent=2))
