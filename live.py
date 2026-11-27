"""Local LLM research. Only the compact filing packet is exposed to the model."""

import json
import os
import re
from datetime import date
from decimal import Decimal
from time import perf_counter
from uuid import uuid4
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, create_model

from pipeline import put, sessions
from research import (
    Claim,
    annual_candidates,
    emit,
    evidence,
    report_text,
    verify_numeric_claim,
)

QUESTION = "Which annual financial measure supports a strength, which supports a downside, and what uncertainty remains? Abstain when the supplied measures do not support a side."
RUNNER_MODEL = "Qwen2.5-1.5B-Instruct-Q4_K_M"
RUNNER = "http://127.0.0.1:18536"
ADDRESS = "http://127.0.0.1:11434"
QUANTITY = re.compile(
    r"\d|[%$€£]|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|double[ds]?|half|twice|percent)\b",
    re.I,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Finding(Output):
    metric: str = Field(
        description="Copy a supplied metric name exactly. Use none for abstention."
    )
    value: str
    interpretation: str = Field(min_length=1, max_length=320)


class Both(Output):
    strength: Finding
    downside: Finding


class Note(Output):
    text: str = Field(min_length=1, max_length=400)
    metrics: list[str] = Field(max_length=2)


class Synthesis(Output):
    metrics: list[str] = Field(max_length=2)
    limitation: Literal["future performance", "causes of change", "valuation"]


def finding_schema(name, choices):
    return create_model(
        name,
        __base__=Finding,
        metric=(
            Literal[tuple(choices + (["none"] if "none" not in choices else []))],
            ...,
        ),
    )


class ModelFailure(RuntimeError):
    pass


def installed_models():
    with httpx.Client(trust_env=False, timeout=2) as client:
        models = []
        try:
            health = client.get(RUNNER + "/v1/models")
            if health.is_success and any(
                x.get("id") == RUNNER_MODEL for x in health.json().get("data", [])
            ):
                models.append(RUNNER_MODEL)
        except (httpx.HTTPError, ValueError):
            pass
        try:
            response = client.get(ADDRESS + "/api/tags")
            response.raise_for_status()
            models.extend(
                r["name"]
                for r in response.json()["models"]
                if not r["name"].endswith(":cloud")
            )
        except (httpx.HTTPError, ValueError):
            pass
        return models


def complete(model, prompt, schema):
    system = "Interpret only the historical financial evidence provided. Use no numbers in narrative prose. Do not forecast or recommend trades. Return the specified JSON."
    if model == RUNNER_MODEL:
        with httpx.Client(trust_env=False, timeout=150) as client:
            response = client.post(
                RUNNER + "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "research",
                            "schema": schema.model_json_schema(),
                        },
                    },
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 320,
                },
            )
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"],
            data["model"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
    if "cloud" in model.lower():
        raise ValueError("Choose an installed local model")
    with httpx.Client(trust_env=False, timeout=150) as client:
        info = client.post(ADDRESS + "/api/show", json={"model": model})
        info.raise_for_status()
        details = info.json()
        if (
            details.get("remote_model")
            or details.get("remote_host")
            or not details.get("model_info")
        ):
            raise ValueError("A locally installed model with local weights is required")
        response = client.post(
            ADDRESS + "/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": schema.model_json_schema(),
                "messages": [
                    {
                        "role": "system",
                        "content": "Interpret only the supplied historical financial evidence. No forecasts, investment advice or invented context. Write short sentences. Use no quantities in prose. Return the required JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 4096,
                    "num_predict": 320,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
    return (
        data["message"]["content"],
        data["model"],
        data.get("prompt_eval_count"),
        data.get("eval_count"),
    )


def qualitative(value):
    if QUANTITY.search(value):
        raise ValueError("Quantitative prose requires a structured claim")


def live_research(engine, ticker, as_of, *, single=False, model=None, case_id=None):
    date.fromisoformat(as_of)
    model = model or os.getenv("OLLAMA_MODEL") or RUNNER_MODEL
    sid = str(uuid4())
    meta = dict(
        trace_id=sid,
        session_id=sid,
        provider=("llama.cpp-local" if model == RUNNER_MODEL else "ollama-local"),
        model=model,
        prompt_version="2.0",
        evaluation_case_id=case_id,
    )
    parent = emit(**meta, event_type="research_started")
    start = perf_counter()
    with engine.connect() as conn:
        pool = evidence(conn, ticker, as_of)
    if not pool:
        emit(
            **meta,
            parent_event_id=parent,
            event_type="error",
            status="error",
            error_type="NoEvidence",
        )
        raise LookupError("No filing evidence by the selected cutoff")
    candidates = annual_candidates(pool)
    packet = [
        dict(
            row=i,
            metric=r["metric"],
            value=f"{r['value']:.2f}",
            unit=r["unit"],
            period=r["end"],
            filed=r["filed"],
        )
        for i, r in enumerate(candidates, 1)
    ]
    context = (
        QUESTION
        + "\nIssuer "
        + ticker
        + " as of "
        + as_of
        + ". Evidence "
        + json.dumps(packet, separators=(",", ":"))
    )
    audit = []
    calls = []

    def ask(role, instruction, schema, background=None):
        t = perf_counter()
        fields = dict(
            meta,
            parent_event_id=parent,
            event_type="agent_run",
            agent_name=role,
            evidence_ids=list(
                dict.fromkeys(fid for r in candidates for fid in r["evidence_ids"])
            ),
        )
        try:
            raw, identity, ti, to = complete(
                model,
                (context if background is None else background) + "\n" + instruction,
                schema,
            )
            fields.update(
                model=identity,
                input_tokens=ti,
                output_tokens=to,
                token_usage_source=(
                    "provider" if ti is not None or to is not None else None
                ),
                estimated_cost_usd=0,
            )
            parsed = schema.model_validate_json(raw)
        except Exception as exc:
            fields.update(status="error", error_type=type(exc).__name__)
            raise ModelFailure(
                f"{role} failed ({type(exc).__name__}). Check the local model service or select offline mode."
            ) from exc
        finally:
            fields["latency_ms"] = (perf_counter() - t) * 1000
            emit(**fields)
            calls.append(
                {
                    k: fields.get(k)
                    for k in (
                        "agent_name",
                        "model",
                        "latency_ms",
                        "input_tokens",
                        "output_tokens",
                        "estimated_cost_usd",
                    )
                }
            )
        return parsed

    refs = set()

    def check(f, side):
        index = next((p["row"] for p in packet if p["metric"] == f.metric), 0)
        abstained = f.metric.lower() == "none"
        valid = bool(index) and f.value.strip() == packet[index - 1]["value"]
        attempt = dict(
            side=side,
            row=index,
            metric=f.metric,
            proposed_value=f.value,
            valid_citation=bool(index),
            numeric_correct=None if abstained else valid,
        )
        audit.append(attempt)
        try:
            if abstained:
                if f.value.strip() not in ("", "0", "N/A", "null"):
                    raise ValueError("Value supplied without evidence")
                return dict(
                    statement="Insufficient directional evidence.",
                    claim=None,
                    status="Insufficient evidence",
                )
            if not attempt["valid_citation"]:
                raise ValueError("Unknown evidence row")
            r = candidates[index - 1]
            valid = Decimal(f.value) == Decimal(packet[index - 1]["value"])
            attempt["numeric_correct"] = valid
            if not valid or (r["value"] <= 0 if side == "Bull" else r["value"] >= 0):
                raise ValueError("Value or direction disagrees with evidence")
            claim = Claim.model_validate(r)
            with engine.connect() as conn:
                if verify_numeric_claim(conn, claim, as_of, pool) != "Supported":
                    raise ValueError("Evidence failed database verification")
            refs.update(r["sources"])
            # Only a source-checkable figure becomes a factual statement.
            # A model-authored follow-up can appear explicitly as a question.
            interpretation = f.interpretation.strip()
            if (
                QUANTITY.search(interpretation)
                or not interpretation.endswith("?")
                or re.search(r"[.!]", interpretation)
            ):
                attempt["prose_withheld"] = True
                interpretation = ""
            return dict(
                statement=f"{r['metric'].replace('_',' ')}: {r['value']:,.2f} {r['unit']} ({r['end']}). {interpretation}".strip(),
                claim=claim.model_dump(),
                status="Supported",
                row=index,
            )
        except (ArithmeticError, ValueError) as exc:
            attempt["reason"] = str(exc)
            emit(
                **meta,
                parent_event_id=parent,
                event_type="tool_call",
                tool_name="verify_numeric_claim",
                status="invalid",
                error_type=type(exc).__name__,
                evidence_ids=(
                    candidates[index - 1]["evidence_ids"]
                    if attempt["valid_citation"]
                    else []
                ),
            )
            return dict(
                statement="Proposed finding withheld after evidence verification.",
                claim=None,
                status="Unverifiable",
            )

    instruction = "For a strength choose a positive row; for a downside choose a negative row. Prefer year-over-year changes. Copy the row's value field verbatim. Copy the exact metric name as the citation. When absent use metric none, empty value, and state insufficient evidence. In interpretation, ask one short follow-up research question about the selected metric. Do not repeat numbers or assert new facts."
    eligible = {
        side: [
            r["metric"]
            for r in candidates
            if (r["value"] > 0 if side == "strength" else r["value"] < 0)
        ]
        or ["none"]
        for side in ("strength", "downside")
    }
    instruction += (
        " Eligible metric names by side: "
        + json.dumps(eligible)
        + ". If the only permitted name is none, return metric none and an empty value."
    )
    bull_schema = finding_schema("BullFinding", eligible["strength"])
    bear_schema = finding_schema("BearFinding", eligible["downside"])
    both_schema = create_model(
        "AnalystFindings",
        __base__=Both,
        strength=(bull_schema, ...),
        downside=(bear_schema, ...),
    )
    try:
        if single:
            both = ask(
                "Analyst",
                "Give both a strength and a downside. " + instruction,
                both_schema,
            )
            args = {
                "Bull": check(both.strength, "Bull"),
                "Bear": check(both.downside, "Bear"),
            }
        else:
            # Neither initial call receives the opposing answer.
            bull = ask(
                "Bull",
                "Give the strongest historical strength. " + instruction,
                bull_schema,
            )
            bear = ask(
                "Bear",
                "Give the principal historical downside. " + instruction,
                bear_schema,
            )
            args = {"Bull": check(bull, "Bull"), "Bear": check(bear, "Bear")}
        accepted = {
            a["claim"]["metric"]: a["row"] for a in args.values() if a.get("row")
        }
        positions = json.dumps(
            {
                side: dict(
                    metric=a["claim"]["metric"] if a["claim"] else "none",
                    direction=(
                        ("positive" if a["claim"]["value"] > 0 else "negative")
                        if a["claim"]
                        else "insufficient evidence"
                    ),
                )
                for side, a in args.items()
            }
        )

        def note(role, task):
            opponent = "Bear" if role.startswith("Bull") else "Bull"
            target = args[opponent]["claim"]
            target_metric = target["metric"] if target else None
            task += (
                " Challenge the "
                + opponent
                + " position. "
                + (
                    "Address and cite " + target_metric + "."
                    if target_metric
                    else "Question the absence of a supported opposing thesis."
                )
            )
            ans = ask(
                role,
                task
                + " Positions: "
                + positions
                + " Cite only accepted metric names in metrics. Use no numeric quantities in text.",
                Note,
                background="Verified annual filing evidence for " + ticker + ".",
            )
            try:
                qualitative(ans.text)
                if not ans.text.strip().endswith("?") or re.search(r"[.!]", ans.text):
                    raise ValueError("Challenge must be a question")
                if not accepted:
                    return "Insufficient verified evidence for an interpretation."
                if target_metric and target_metric not in ans.metrics:
                    raise ValueError("Opposing claim not cited")
                if not set(ans.metrics) <= set(accepted) or not ans.metrics:
                    raise ValueError("Invalid note citation")
                return (
                    ans.text
                    + " [evidence "
                    + ", ".join(str(accepted[m]) for m in ans.metrics)
                    + "]"
                )
            except ValueError as exc:
                audit.append(dict(stage=role, reason=str(exc)))
                emit(
                    **meta,
                    parent_event_id=parent,
                    event_type="tool_call",
                    tool_name="verify_note",
                    status="invalid",
                    error_type="InvalidNote",
                )
                return "Interpretation withheld after verification."

        challenges = (
            []
            if single
            else [
                note(
                    side + " challenge",
                    "Ask a short critical question about the limits of the opposing finding. Speak from the "
                    + side
                    + " perspective.",
                )
                for side in ("Bull", "Bear")
            ]
        )
        synthesis = ask(
            "Analyst synthesis" if single else "Synthesis",
            "Synthesize the verified positions and challenges. Order the supported metric names by importance and select the most relevant unresolved limitation. Include each supported metric exactly once. "
            + positions
            + " Challenges: "
            + json.dumps(challenges),
            Synthesis,
            background="Issuer " + ticker,
        )
        if set(synthesis.metrics) == set(accepted) and len(synthesis.metrics) == len(
            accepted
        ):
            findings = [
                next(
                    a["claim"]
                    for a in args.values()
                    if a["claim"] and a["claim"]["metric"] == metric
                )
                for metric in synthesis.metrics
            ]
            clauses = [
                f"{c['metric'].replace('_',' ')} was {c['value']:,.2f} {c['unit']}"
                for c in findings
            ]
            conclusion = (
                "For the reporting period, " + ", while ".join(clauses) + ". "
                if clauses
                else "There is insufficient verified directional evidence. "
            )
            conclusion += f"The filing leaves {synthesis.limitation} uncertain."
        else:
            audit.append(
                dict(
                    stage="Synthesis",
                    reason="Synthesis did not cover exactly the verified metrics",
                )
            )
            emit(
                **meta,
                parent_event_id=parent,
                event_type="tool_call",
                tool_name="verify_synthesis",
                status="invalid",
                error_type="InvalidSynthesis",
            )
            observed = [a["claim"] for a in args.values() if a["claim"]]
            lines = [
                f"{c['metric'].replace('_',' ')} was {c['value']:,.2f} {c['unit']}."
                for c in observed
            ]
            conclusion = (
                "Model synthesis withheld. Verified summary: "
                + " ".join(lines)
                + " Future performance remains uncertain."
            )
        for side, item in args.items():
            if not item["claim"]:
                conclusion += (
                    f" The {side.lower()} position has insufficient verified support."
                )
        report = dict(
            id=sid,
            ticker=ticker,
            as_of=as_of,
            provider=("llama.cpp-local" if model == RUNNER_MODEL else "ollama-local"),
            model=model,
            mode="single_llm" if single else "bull_bear_llm",
            summary=QUESTION,
            arguments=args,
            challenges=challenges,
            conclusion=conclusion,
            sources=sorted(refs),
            audit=audit,
            calls=calls,
            evidence_packet=[
                dict(p, evidence_ids=r["evidence_ids"], sources=r["sources"])
                for p, r in zip(packet, candidates)
            ],
        )
        report["markdown"] = (
            report_text(report)
            + "\n\n## Evidence register\n"
            + "\n".join(
                f"- [{p['row']}] {p['metric']} ({p['period']}) · "
                + ", ".join(p["evidence_ids"])
                for p in report["evidence_packet"]
            )
        )
        with engine.begin() as conn:
            put(conn, sessions, dict(id=sid, ticker=ticker, result=json.dumps(report)))
    except Exception as exc:
        emit(
            **meta,
            parent_event_id=parent,
            event_type="error",
            status="error",
            error_type=type(exc).__name__,
            latency_ms=(perf_counter() - start) * 1000,
        )
        raise
    emit(
        **meta,
        parent_event_id=parent,
        event_type="research_completed",
        latency_ms=(perf_counter() - start) * 1000,
    )
    return report
