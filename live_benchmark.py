"""Three workflow comparison on identical FY2024 financial evidence."""

import json
from decimal import Decimal, InvalidOperation
from time import perf_counter
from uuid import uuid4

from benchmark import REFERENCES
from live import QUESTION, ModelFailure, live_research
from pipeline import database, evaluations, get_source_evidence, put
from research import Rules, annual_candidates, evidence, research

AS_OF = "2024-12-31"


def reference_values(ticker):
    _, financials = REFERENCES[ticker]
    result = {m: 100 * (cur / old - 1) for m, (cur, old) in financials.items()}
    if ticker in ("MSFT", "AAPL"):
        result["net_margin"] = (
            100 * financials["net_income_yoy"][0] / financials["revenue_yoy"][0]
        )
    if ticker == "AAPL":
        result["free_cash_flow"] = 108807e6
    return result


def score(engine, ticker, report, rows):
    targets = reference_values(ticker)
    attempts = report.get("audit")
    if attempts is None:
        attempts = [
            dict(metric=c["metric"], proposed_value=str(c["value"]))
            for a in report["arguments"].values()
            if (c := a["claim"])
        ]
    numeric = []
    cited = []
    for item in attempts:
        if "metric" not in item or (
            item["metric"].lower() == "none" and not item.get("proposed_value")
        ):
            continue
        metric = item["metric"]
        ref = targets.get(metric)
        try:
            correct = ref is not None and abs(
                Decimal(item["proposed_value"]) - Decimal(str(ref))
            ) <= Decimal("0.005")
        except (ValueError, InvalidOperation):
            correct = False
        numeric.append(correct)
        r = next((r for r in rows if r["metric"] == metric), None)
        with engine.connect() as conn:
            cited.append(
                bool(
                    r
                    and all(
                        (f := get_source_evidence(conn, fid))
                        and f["ticker"] == ticker
                        and f["filed"] <= AS_OF
                        for fid in r["evidence_ids"]
                    )
                )
            )
    args = report["arguments"]
    coverage = {}
    for side, positive in [("Bull", True), ("Bear", False)]:
        available = any((r["value"] > 0 if positive else r["value"] < 0) for r in rows)
        claim = args[side]["claim"]
        coverage[side] = (
            bool(
                claim
                and args[side]["status"] == "Supported"
                and (claim["value"] > 0 if positive else claim["value"] < 0)
            )
            if available
            else args[side]["status"] == "Insufficient evidence"
        )
    conclusion = report["conclusion"].lower()
    coverage["uncertainty"] = not conclusion.startswith(
        ("interpretation withheld", "model synthesis withheld")
    ) and any(
        word in conclusion
        for word in [
            "uncertain",
            "inconclusive",
            "limit",
            "insufficient",
            "unclear",
            "cannot",
            "could",
            "may",
            "historical",
            "caution",
            "unresolved",
        ]
    )
    return dict(
        numeric_correct=sum(numeric),
        numeric_attempts=len(numeric),
        unsupported_numeric_proposals=numeric.count(False),
        valid_citations=sum(cited),
        citation_attempts=len(cited),
        coverage=coverage,
        rejected_notes=sum("stage" in x for x in attempts),
        prose_withheld=sum(bool(x.get("prose_withheld")) for x in attempts),
    )


def benchmark(engine):
    results = {}
    for mode in ("deterministic", "single_llm", "bull_bear_llm"):
        cases = []
        for ticker in REFERENCES:
            with engine.connect() as conn:
                rows = annual_candidates(evidence(conn, ticker, AS_OF))
            start = perf_counter()
            try:
                report = (
                    research(engine, ticker, AS_OF, Rules())
                    if mode == "deterministic"
                    else live_research(
                        engine,
                        ticker,
                        AS_OF,
                        single=mode == "single_llm",
                        case_id=ticker + "-FY2024",
                    )
                )
                elapsed = (perf_counter() - start) * 1000
                scores = score(engine, ticker, report, rows)
                calls = report.get("calls", [])

                def usage(field):
                    return (
                        sum(c[field] for c in calls)
                        if calls and all(c[field] is not None for c in calls)
                        else None
                    )

                cases.append(
                    dict(
                        ticker=ticker,
                        status="ok",
                        report_id=report["id"],
                        latency_ms=elapsed,
                        model=report.get("model"),
                        input_tokens=usage("input_tokens"),
                        output_tokens=usage("output_tokens"),
                        provider_fee_usd=0,
                        evidence_ids=[r["evidence_ids"] for r in rows],
                        sources=[REFERENCES[ticker][0]],
                        **scores,
                    )
                )
            except ModelFailure as exc:
                cases.append(
                    dict(
                        ticker=ticker,
                        status="failed",
                        error=str(exc),
                        latency_ms=(perf_counter() - start) * 1000,
                    )
                )
        results[mode] = cases
    outcome = dict(question=QUESTION, as_of=AS_OF, runs=results)
    with engine.begin() as conn:
        put(conn, evaluations, dict(id=str(uuid4()), result=json.dumps(outcome)))
    return outcome


if __name__ == "__main__":
    print(json.dumps(benchmark(database()), indent=2))
