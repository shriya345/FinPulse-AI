import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from streamlit.testing.v1 import AppTest

import api
from evaluate import evaluate
from pipeline import (
    ROOT,
    database,
    facts,
    get_financial_facts,
    get_financial_metrics,
    load,
    period,
    runs,
)
from research import Claim, Event, evidence, verify_numeric_claim


@pytest.fixture
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path/'test.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("TELEMETRY_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    engine = database(url)
    for path in sorted((ROOT / "snapshots").glob("*.json")):
        raw = path.read_text()
        x = json.loads(raw)
        load(engine, x["ticker"], raw, x["url"], x["capture"])
    return engine


def test_normalization():
    assert [
        period(s, "2024-12-31") for s in ("", "2024-01-01", "2024-10-01", "2024-07-01")
    ] == ["instant", "annual", "quarter", "cumulative"]


def test_reingestion_and_failure(db):
    p = next((ROOT / "snapshots").glob("AAPL*.json"))
    raw = p.read_text()
    with db.connect() as c:
        count = c.scalar(select(func.count()).select_from(facts))
    load(db, "AAPL", raw, json.loads(raw)["url"], "excerpt")
    with pytest.raises(ValueError):
        load(db, "NVDA", raw, "invalid", "fixture")
    with db.connect() as c:
        assert c.scalar(select(func.count()).select_from(facts)) == count
        assert (
            c.scalar(
                select(func.count()).select_from(runs).where(runs.c.status == "failed")
            )
            == 1
        )


def test_future_and_missing(db):
    with db.connect() as c:
        assert not get_financial_facts(c, "AAPL", "1990-01-01")
        assert not get_financial_metrics(c, "UNKNOWN", "2024-12-31")


def test_verification(db):
    with db.connect() as c:
        pool = evidence(c, "AAPL", "2024-12-31")
        r = next(
            x
            for x in pool
            if x["metric"] == "net_margin"
            and x["end"] == "2024-09-28"
            and x["kind"] == "annual"
        )
        claim = Claim.model_validate(r)
        for update, expected in [
            ({}, "Supported"),
            ({"value": -1}, "Contradicted"),
            ({"ticker": "MSFT"}, "Contradicted"),
            ({"evidence_ids": []}, "Insufficient evidence"),
            ({"evidence_ids": ["missing"]}, "Unverifiable"),
        ]:
            assert (
                verify_numeric_claim(c, claim.model_copy(update=update), "2024-12-31")
                == expected
            )


def test_reference_answers(db):
    for result in evaluate(db).values():
        assert result["correct"] == result["total"] == 12


def test_research_api(db, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "engine", db)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    result = client.post(
        "/research", json={"ticker": "AAPL", "as_of": "2024-12-31"}
    ).json()
    assert set(result["arguments"]) == {"Bull", "Bear"}
    assert len(result["challenges"]) == 2
    assert client.get("/research/" + result["id"]).json() == result
    events = [
        Event.model_validate_json(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert all(e.input_tokens is None for e in events)
    assert events[-1].trace_id == events[0].trace_id
    assert client.get("/companies/ZZZ/metrics").status_code == 404


def test_sql(db):
    with db.connect() as c:
        for file in sorted((ROOT / "sql").glob("[0-9]*.sql")):
            for stmt in file.read_text().split(";"):
                if stmt.strip():
                    c.exec_driver_sql(stmt)


def test_dashboard(db, monkeypatch):
    monkeypatch.setenv("FINPULSE_OFFLINE", "1")
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.button[0].click().run()
    assert not app.exception
    assert "Cross-examination" in [s.value for s in app.subheader]


def test_full_companyfacts_and_ratios(tmp_path, monkeypatch):
    import httpx

    import pipeline

    engine = database(f"sqlite:///{tmp_path/'refresh.db'}")
    data = {}

    def fact(tag, value, start=None, end="2024-12-31"):
        row = dict(
            val=value,
            end=end,
            filed="2025-02-01",
            form="10-K",
            accn="0000320193-25-000001",
        )
        if start:
            row["start"] = start
        data.setdefault(tag, {"units": {"USD": []}})["units"]["USD"].append(row)

    for tag, value in [
        ("NetIncomeLoss", 20),
        ("OperatingIncomeLoss", 30),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", 100),
    ]:
        fact(tag, value, "2024-01-01")
    for tag, old, new in [("Assets", 80, 120), ("StockholdersEquity", 40, 60)]:
        fact(tag, old, end="2023-12-31")
        fact(tag, new)
    fact("Liabilities", 60)
    calls = []

    def reply(url, headers, timeout, **options):
        calls.append(headers)
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "cik": 320193,
                "entityName": "Test issuer",
                "facts": {"us-gaap": data},
            },
        )

    monkeypatch.setenv("SEC_USER_AGENT", "Fixture qa@example.org")
    monkeypatch.setattr(pipeline.httpx, "get", reply)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    pipeline.refresh(engine, "AAPL")
    pipeline.refresh(engine, "AAPL")
    assert len(calls) == 1 and "qa@example.org" in calls[0]["User-Agent"]
    with engine.connect() as conn:
        m = {
            r["metric"]: r["value"]
            for r in get_financial_metrics(conn, "AAPL", "2025-02-01")
        }
        assert m == {
            "return_on_assets": 20,
            "return_on_equity": 40,
            "liabilities_to_equity": 1,
            "net_margin": 20,
            "operating_margin": 30,
        }


def test_live_round_and_evidence_gate(db, monkeypatch, tmp_path):
    import live

    calls = []

    def model(name, prompt, schema):
        calls.append((schema, prompt))
        if issubclass(schema, live.Finding):
            raw = (
                {
                    "metric": "revenue_yoy",
                    "value": "999.99",
                    "interpretation": "Sales rose.",
                }
                if len(calls) == 1
                else {
                    "metric": "net_income_yoy",
                    "value": "-3.36",
                    "interpretation": "Earnings fell.",
                }
            )
        elif schema is live.Synthesis:
            raw = {"metrics": ["net_income_yoy"], "limitation": "future performance"}
        else:
            raw = {
                "text": "Does the earnings trend persist?",
                "metrics": ["net_income_yoy"],
            }
        return json.dumps(raw), "test-local", 10, 5

    monkeypatch.setattr(live, "complete", model)
    out = live.live_research(db, "AAPL", "2024-12-31", model="test-local")
    assert len(calls) == 5 and out["arguments"]["Bull"]["claim"] is None
    assert out["arguments"]["Bear"]["status"] == "Supported"
    assert all("net_income_yoy" in prompt for _, prompt in calls)
    assert out["audit"][0]["numeric_correct"] is False
    assert "999.99" not in out["markdown"]
    events = [
        Event.model_validate_json(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert (
        len(
            [
                e
                for e in events
                if e.event_type == "agent_run"
                and e.input_tokens == 10
                and e.model == "test-local"
            ]
        )
        == 5
    )
    assert any(e.status == "invalid" for e in events)
    assert len({e.session_id for e in events}) == 1


def test_invalid_live_output_keeps_usage_and_offline_access(db, monkeypatch, tmp_path):
    import live

    monkeypatch.setattr(live, "complete", lambda *args: ("broken", "test-local", 9, 2))
    monkeypatch.setattr(api, "engine", db)
    client = TestClient(api.app)
    assert (
        client.post("/research", json={"ticker": "AAPL", "mode": "live"}).status_code
        == 503
    )
    events = [
        Event.model_validate_json(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert any(e.status == "error" and e.output_tokens == 2 for e in events)
    assert (
        client.post(
            "/research", json={"ticker": "AAPL", "mode": "deterministic"}
        ).json()["provider"]
        == "deterministic"
    )


def test_unknown_note_citation_and_quantitative_prose(db, monkeypatch):
    import live

    def model(name, prompt, schema):
        if issubclass(schema, live.Both):
            strength = dict(
                metric="revenue_yoy",
                value="2.02",
                interpretation="Stock will rise 500%.",
            )
            downside = dict(
                metric="none", value="", interpretation="Insufficient evidence."
            )
            raw = dict(strength=strength, downside=downside)
        else:
            raw = dict(limitation="valuation", metrics=["invented"])
        return json.dumps(raw), "test-local", None, None

    monkeypatch.setattr(live, "complete", model)
    r = live.live_research(db, "AAPL", "2024-12-31", single=True)
    assert "500%" not in r["markdown"]
    assert "withheld" in r["conclusion"]
    assert r["arguments"]["Bull"]["status"] == "Supported"
    assert len(r["calls"]) == 2
