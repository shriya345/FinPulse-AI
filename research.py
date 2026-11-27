"""Bounded evidence selection and numeric verification."""

import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from pipeline import (
    get_financial_facts,
    get_financial_metrics,
    get_source_evidence,
    now,
    put,
    sessions,
)


class Claim(BaseModel):
    ticker: str
    metric: str
    start: str
    end: str
    unit: str
    value: float = Field(allow_inf_nan=False)
    evidence_ids: list[str]


class Event(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    application: str = "FinPulse AI"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=now)
    session_id: str
    trace_id: str
    parent_event_id: str | None = None
    event_type: str
    agent_name: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str = "1"
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_usage_source: str | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    status: str = "ok"
    error_type: str | None = None
    tool_name: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evaluation_case_id: str | None = None


def emit(**fields):
    e = Event(**fields)
    with Path(os.getenv("TELEMETRY_PATH", "telemetry.jsonl")).open("a") as file:
        file.write(e.model_dump_json() + "\n")
    return e.event_id


def evidence(conn, ticker, as_of):
    metrics = get_financial_metrics(conn, ticker, as_of)
    records = get_financial_facts(conn, ticker, as_of)
    return metrics + [
        dict(r, evidence_ids=[r["id"]], sources=[r["url"]]) for r in records
    ]


def verify_numeric_claim(conn, claim: Claim, as_of: str, pool=None):
    if not claim.evidence_ids:
        return "Insufficient evidence"
    cited = [get_source_evidence(conn, i) for i in claim.evidence_ids]
    if any(row is None for row in cited):
        return "Unverifiable"
    if any(r["ticker"] != claim.ticker or r["filed"] > as_of for r in cited):
        return "Contradicted"
    rows = evidence(conn, claim.ticker, as_of) if pool is None else pool
    for r in rows:
        if (
            set(r["evidence_ids"]) == set(claim.evidence_ids)
            and r["metric"] == claim.metric
        ):
            if all(
                r[k] == getattr(claim, k) for k in ("ticker", "start", "end", "unit")
            ) and math.isclose(r["value"], claim.value, rel_tol=1e-12, abs_tol=1e-8):
                return "Supported"
            return "Contradicted"
    return "Insufficient evidence"


class Selector(Protocol):
    name: str
    model: str | None

    def choose(self, role: str, candidates: list[dict]) -> tuple[int, dict]: ...


class Rules:
    name = "deterministic"
    model = None

    def choose(self, role, candidates):
        return 0, {}


def report_text(report):
    parts = [
        f"# {report['ticker']} financial research",
        f"As of {report['as_of']} · {report['provider']} evidence selection",
        report["summary"],
    ]
    for side, item in report["arguments"].items():
        parts.extend(
            [f"## {side}", item["statement"], f"Verification: {item['status']}"]
        )
        if item["claim"]:
            parts.append("Evidence: " + ", ".join(item["claim"]["evidence_ids"]))
    parts += ["## Cross-examination"] + [f"- {x}" for x in report["challenges"]]
    parts += ["## Conclusion", report["conclusion"], "## Sources"] + report["sources"]
    return "\n\n".join(parts)


def annual_candidates(pool):
    directional = [
        "revenue_yoy",
        "net_income_yoy",
        "operating_income_yoy",
        "net_margin",
        "operating_margin",
        "free_cash_flow",
    ]
    latest = max((r["end"] for r in pool if r["kind"] == "annual"), default="")
    rows = [
        r
        for r in pool
        if r["kind"] == "annual" and r["end"] == latest and r["metric"] in directional
    ]
    return sorted(rows, key=lambda r: directional.index(r["metric"]))


def research(engine, ticker, as_of, selector: Selector | None = None, *, single=False):
    if selector is None and os.getenv("OLLAMA_MODEL"):
        from live import live_research

        return live_research(engine, ticker, as_of, single=single)
    selector = selector or Rules()
    sid = str(uuid4())
    base = dict(
        session_id=sid, trace_id=sid, provider=selector.name, model=selector.model
    )
    parent = emit(**base, event_type="research_started")
    args = {}
    refs = set()
    with engine.begin() as conn:
        pool = evidence(conn, ticker, as_of)
        if not pool:
            emit(
                **base,
                parent_event_id=parent,
                event_type="error",
                status="error",
                error_type="NoEvidence",
            )
            raise LookupError("No financial evidence available by selected date")
        annual = annual_candidates(pool)
        for role in ("Analyst",) if single else ("Bull", "Bear"):
            start = perf_counter()
            candidates = [
                r
                for r in annual
                if role == "Analyst"
                or (r["value"] > 0 if role == "Bull" else r["value"] < 0)
            ]
            claim = None
            usage = {}
            try:
                if candidates:
                    index, usage = selector.choose(role, candidates)
                    r = candidates[index]
                    claim = Claim.model_validate(r)
                    status = (
                        "Unchecked"
                        if single
                        else verify_numeric_claim(conn, claim, as_of, pool)
                    )
                    statement = f"{r['metric'].replace('_',' ')} was {r['value']:,.2f} {r['unit']} for the year ending {r['end']}."
                    refs.update(r["sources"])
                else:
                    statement = f"The available annual metrics provide insufficient directional evidence for the {role.lower()} thesis."
                    status = "Insufficient evidence"
            except Exception as exc:
                emit(
                    **base,
                    parent_event_id=parent,
                    event_type="agent_run",
                    agent_name=role,
                    status="error",
                    error_type=type(exc).__name__,
                    latency_ms=(perf_counter() - start) * 1000,
                )
                raise
            emit(
                **base,
                parent_event_id=parent,
                event_type="agent_run",
                agent_name=role,
                latency_ms=(perf_counter() - start) * 1000,
                evidence_ids=claim.evidence_ids if claim else [],
                **usage,
            )
            args[role] = dict(
                statement=statement,
                claim=claim.model_dump() if claim else None,
                status=status,
            )
        challenges = []
        for challenger, opponent in (
            [] if single else [("Bear", "Bull"), ("Bull", "Bear")]
        ):
            target = args[opponent]
            counter = args[challenger]
            if target["claim"] and counter["claim"]:
                challenge = (
                    f"{challenger} challenges a one-sided reading. {counter['statement']} "
                    f"{opponent} retains its verified observation and qualifies its thesis "
                    "because the two measures moved in different directions."
                )
            elif target["claim"]:
                challenge = (
                    f"{challenger} asks whether the observed {target['claim']['metric']} "
                    "establishes a durable trend. The evidence covers a historical period. "
                    f"{opponent} limits its thesis to that period."
                )
            else:
                challenge = (
                    f"{challenger} asks which cited measure supports the opposing position. "
                    f"{opponent} leaves its position unresolved."
                )
            challenges.append(challenge)
        report = dict(
            id=sid,
            ticker=ticker,
            as_of=as_of,
            provider=selector.name,
            summary=f"Reviewed {len(pool)} dated facts and derived metrics available by the cutoff.",
            arguments=args,
            challenges=challenges,
            sources=sorted(refs),
            conclusion="Inconclusive on future performance. Filing excerpts cover selected accounting measures. Valuation, competitive conditions and forward guidance are outside the evidence set.",
        )
        report["markdown"] = report_text(report)
        put(conn, sessions, dict(id=sid, ticker=ticker, result=json.dumps(report)))
    emit(**base, parent_event_id=parent, event_type="research_completed")
    return report
