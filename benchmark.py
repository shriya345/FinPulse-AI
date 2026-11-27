"""Paired historical report evaluation with a fixed FY2024 annual-report rubric."""

import json
from time import perf_counter
from uuid import uuid4

from pipeline import database, evaluations, get_source_evidence, put
from research import Rules, research

# Independent reported FY2024 and FY2023 figures, USD millions.
REFERENCES = {
    "AAPL": (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        {"revenue_yoy": (391035, 383285), "net_income_yoy": (93736, 96995)},
    ),
    "MSFT": (
        "https://www.sec.gov/Archives/edgar/data/789019/000119312524242888/d815777dars.pdf",
        {"revenue_yoy": (245122, 211915), "net_income_yoy": (88136, 72361)},
    ),
    "NVDA": (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
        {"revenue_yoy": (60922, 26974), "operating_income_yoy": (32972, 4224)},
    ),
}


def benchmark(engine):
    selector = Rules()
    result = {}
    for single in (True, False):
        checks = []
        supported = unsupported = valid = citations = 0
        timings = []
        for ticker, (reference, figures) in REFERENCES.items():
            start = perf_counter()
            report = research(engine, ticker, "2024-12-31", selector, single=single)
            timings.append(1000 * (perf_counter() - start))
            claims = [x["claim"] for x in report["arguments"].values() if x["claim"]]
            correct = []
            with engine.connect() as conn:
                for claim in claims:
                    nums = figures.get(claim["metric"])
                    matches = (
                        nums is not None
                        and abs(claim["value"] - 100 * (nums[0] / nums[1] - 1))
                        < 0.00001
                        and claim["end"].startswith("2024")
                        and claim["unit"] == "%"
                    )
                    correct.append(matches)
                    supported += int(matches)
                    unsupported += int(not matches)
                    for fid in claim["evidence_ids"]:
                        r = get_source_evidence(conn, fid)
                        valid += int(
                            r is not None
                            and r["ticker"] == ticker
                            and r["filed"] <= "2024-12-31"
                        )
                        citations += 1
            for label, ok in [
                (
                    "Quantify a verified FY2024 strength",
                    any(flag and c["value"] > 0 for c, flag in zip(claims, correct)),
                ),
                (
                    "Report an earnings decline or explicitly state limited downside evidence",
                    (
                        any(flag and c["value"] < 0 for c, flag in zip(claims, correct))
                        if ticker == "AAPL"
                        else report["arguments"].get("Bear", {}).get("status")
                        == "Insufficient evidence"
                    ),
                ),
                (
                    "Include source references for at least one numeric observation",
                    bool(claims and report["sources"]),
                ),
                (
                    "Leave future performance unresolved",
                    "Inconclusive" in report["conclusion"],
                ),
            ]:
                checks.append(
                    dict(ticker=ticker, question=label, passed=ok, reference=reference)
                )
        result["single" if single else "bull_bear"] = dict(
            provider=selector.name,
            model=selector.model,
            passed=sum(r["passed"] for r in checks),
            total=len(checks),
            supported=supported,
            unsupported=unsupported,
            numeric_claims=supported + unsupported,
            valid_citations=valid,
            total_citations=citations,
            latency_ms=timings,
            cases=checks,
        )
    with engine.begin() as conn:
        put(conn, evaluations, dict(id=str(uuid4()), result=json.dumps(result)))
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(database()), indent=2))
