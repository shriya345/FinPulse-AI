"""Transactional EDGAR ingestion and point-in-time analytics."""

import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

ROOT = Path(__file__).resolve().parent
CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810"}
TAGS = dict(
    zip(
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "NetIncomeLoss",
            "OperatingIncomeLoss",
            "Assets",
            "Liabilities",
            "StockholdersEquity",
            "NetCashProvidedByUsedInOperatingActivities",
            "PaymentsToAcquirePropertyPlantAndEquipment",
        ],
        [
            "revenue",
            "net_income",
            "operating_income",
            "assets",
            "liabilities",
            "equity",
            "operating_cash",
            "capex",
        ],
    )
)
meta = MetaData()


def col(name, typ=String, **kwargs):
    return Column(name, typ, **kwargs)


def key():
    return col("id", primary_key=True)


companies = Table(
    "companies", meta, col("ticker", primary_key=True), col("name"), col("cik")
)
sources = Table(
    "source_records",
    meta,
    key(),
    col("url"),
    col("retrieved"),
    col("capture"),
    col("raw", Text),
)
facts = Table(
    "financial_facts",
    meta,
    key(),
    Column("ticker", ForeignKey("companies.ticker")),
    Column("source", ForeignKey("source_records.id")),
    col("metric"),
    col("start"),
    col("end"),
    col("filed"),
    col("kind"),
    col("unit"),
    col("value", BigInteger),
    col("accession"),
)
Index("ix_fact_lookup", facts.c.ticker, facts.c.metric, facts.c.end, facts.c.filed)
runs = Table(
    "ingestion_runs",
    meta,
    key(),
    col("ticker"),
    col("timestamp"),
    col("status"),
    col("rows_seen", Integer),
    col("rejected", Integer),
    col("error"),
)
sessions = Table("research_sessions", meta, key(), col("ticker"), col("result", Text))
evaluations = Table("evaluation_results", meta, key(), col("result", Text))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def database(url=None):
    engine = create_engine(url or os.getenv("DATABASE_URL", "sqlite:///finpulse.db"))
    if engine.dialect.name == "sqlite":
        event.listen(
            engine, "connect", lambda db, _: db.execute("pragma foreign_keys=on")
        )
    meta.create_all(engine)
    return engine


def put(conn, table, values):
    insert = pg_insert if conn.dialect.name == "postgresql" else sqlite_insert
    conn.execute(insert(table).values(values).on_conflict_do_nothing())


def period(start, end):
    last = date.fromisoformat(end)
    if not start:
        return "instant"
    days = (last - date.fromisoformat(start)).days + 1
    if 350 <= days <= 380:
        return "annual"
    if 75 <= days <= 105:
        return "quarter"
    if 155 <= days <= 295:
        return "cumulative"
    return "other"


def load(engine, ticker, raw, url, capture, retrieved=None):
    from uuid import uuid4

    run = dict(
        id=str(uuid4()),
        ticker=ticker,
        timestamp=now(),
        rows_seen=0,
        rejected=0,
        error=None,
    )
    try:
        payload = json.loads(raw)
        if int(payload["cik"]) != int(CIKS[ticker]):
            raise ValueError("Issuer mismatch")
        source = digest(url + raw)
        concepts = (
            payload.get("facts", {}).get("us-gaap", {})
            if "facts" in payload
            else {payload["tag"]: payload}
        )
        with engine.begin() as conn:
            put(
                conn,
                companies,
                dict(ticker=ticker, cik=CIKS[ticker], name=payload["entityName"]),
            )
            put(
                conn,
                sources,
                dict(
                    id=source,
                    url=url,
                    retrieved=retrieved or now(),
                    capture=capture,
                    raw=raw,
                ),
            )
            batch = {}
            for tag, concept in concepts.items():
                if tag not in TAGS:
                    continue
                for row in concept.get("units", {}).get("USD", []):
                    if row.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                        continue
                    try:
                        start, end, filed = (
                            row.get("start") or "",
                            row["end"],
                            row["filed"],
                        )
                        kind = period(start, end)
                        if (
                            date.fromisoformat(filed) < date.fromisoformat(end)
                            or not isinstance(row["val"], int)
                            or not row["accn"]
                        ):
                            raise ValueError("Invalid observation")
                        fields = [ticker, tag, start, end, filed, row["accn"]]
                        fid = digest("|".join(fields))
                        batch[fid] = dict(
                            id=fid,
                            ticker=ticker,
                            source=source,
                            metric=TAGS[tag],
                            start=start,
                            end=end,
                            filed=filed,
                            kind=kind,
                            unit="USD",
                            value=row["val"],
                            accession=row["accn"],
                        )
                    except (ValueError, KeyError, TypeError):
                        run["rejected"] += 1
            insert = pg_insert if conn.dialect.name == "postgresql" else sqlite_insert
            if batch:
                conn.execute(
                    insert(facts).on_conflict_do_nothing(), list(batch.values())
                )
            run["rows_seen"] = len(batch)
            put(conn, runs, dict(**run, status="ok"))
    except Exception as exc:
        with engine.begin() as conn:
            put(
                conn,
                runs,
                dict(**{**run, "error": type(exc).__name__}, status="failed"),
            )
        raise


def refresh(engine, ticker):
    agent = os.environ.get("SEC_USER_AGENT", "")
    if len(agent.strip()) < 12 or agent.strip().lower() in {
        "python-httpx",
        "mozilla/5.0",
    }:
        raise ValueError(
            "Set SEC_USER_AGENT to a descriptive application and researcher contact"
        )
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIKS[ticker]}.json"
    with engine.connect() as conn:
        cached = conn.execute(
            select(sources.c.retrieved)
            .where(sources.c.url == url)
            .order_by(sources.c.retrieved.desc())
        ).scalar()
    if (
        cached
        and (
            datetime.now(timezone.utc) - datetime.fromisoformat(cached)
        ).total_seconds()
        < 3600
    ):
        return
    try:
        for attempt in range(3):
            time.sleep(1 if attempt == 0 else 2**attempt)
            try:
                # Prefer the configured HTTPS transport over a generic SOCKS fallback.
                options = {}
                if proxy := os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"):
                    options["proxy"] = proxy
                response = httpx.get(
                    url, headers={"User-Agent": agent}, timeout=45, **options
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(min(int(response.headers.get("Retry-After", "2")), 120))
                    continue
                response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
    except Exception as exc:
        from uuid import uuid4

        with engine.begin() as conn:
            put(
                conn,
                runs,
                dict(
                    id=str(uuid4()),
                    ticker=ticker,
                    timestamp=now(),
                    status="failed",
                    rows_seen=0,
                    rejected=0,
                    error=type(exc).__name__,
                ),
            )
        raise
    load(engine, ticker, response.text, url, "full companyfacts response")


def get_company_profile(conn, ticker: str):
    return (
        conn.execute(select(companies).where(companies.c.ticker == ticker))
        .mappings()
        .one_or_none()
    )


def get_source_evidence(conn, fid: str):
    return (
        conn.execute(
            select(facts, sources.c.url).join(sources).where(facts.c.id == fid)
        )
        .mappings()
        .one_or_none()
    )


def get_financial_facts(conn, ticker: str, as_of: str):
    date.fromisoformat(as_of)
    q = (
        select(facts, sources.c.url)
        .join(sources)
        .where(facts.c.ticker == ticker, facts.c.filed <= as_of)
        .order_by(facts.c.end, facts.c.filed, facts.c.accession)
    )
    latest = {}
    for row in conn.execute(q).mappings():
        latest[(row["metric"], row["start"], row["end"])] = dict(row)
    return list(latest.values())


def compare_financial_periods(current: dict, previous: dict):
    delta = lambda k: (
        date.fromisoformat(current[k]) - date.fromisoformat(previous[k])
    ).days
    return (
        current["ticker"] == previous["ticker"]
        and current["metric"] == previous["metric"]
        and current["unit"] == previous["unit"]
        and current["kind"] == previous["kind"]
        and all(
            350 <= delta(k) <= 380
            for k in (["end", "start"] if current["start"] else ["end"])
        )
    )


def get_financial_metrics(conn, ticker: str, as_of: str):
    rows = get_financial_facts(conn, ticker, as_of)
    by_key = {(r["metric"], r["start"], r["end"]): r for r in rows}
    out = []
    for r in rows:
        if r["kind"] not in {"quarter", "annual", "instant"}:
            continue

        def add(name, value, unit, *deps):
            out.append(
                dict(
                    metric=name,
                    value=value,
                    unit=unit,
                    start=r["start"],
                    end=r["end"],
                    kind=r["kind"],
                    ticker=ticker,
                    evidence_ids=[d["id"] for d in deps],
                    sources=[d["url"] for d in deps],
                    filed=max(d["filed"] for d in deps),
                )
            )

        def same(metric):
            return by_key.get((metric, r["start"], r["end"]))

        if r["kind"] == "annual" and r["metric"] in {"net_income", "operating_income"}:
            previous = next(
                (p for p in reversed(rows) if compare_financial_periods(r, p)), None
            )
            if previous and previous["value"] > 0:
                add(
                    r["metric"] + "_yoy",
                    100 * (r["value"] / previous["value"] - 1),
                    "%",
                    r,
                    previous,
                )
        if r["metric"] == "revenue":
            prior = next(
                (p for p in reversed(rows) if compare_financial_periods(r, p)), None
            )
            if prior and prior["value"] > 0:
                add(
                    "revenue_yoy",
                    100 * (r["value"] / prior["value"] - 1),
                    "%",
                    r,
                    prior,
                )
            for numerator, name in [
                ("operating_income", "operating_margin"),
                ("net_income", "net_margin"),
            ]:
                n = same(numerator)
                if n and r["value"] > 0:
                    add(name, 100 * n["value"] / r["value"], "%", n, r)
        if r["metric"] == "operating_cash" and (cap := same("capex")):
            add("free_cash_flow", r["value"] - cap["value"], "USD", r, cap)
        if r["metric"] == "liabilities" and (eq := same("equity")) and eq["value"] > 0:
            add("liabilities_to_equity", r["value"] / eq["value"], "x", r, eq)
        if r["metric"] == "net_income" and r["kind"] == "annual":
            from datetime import timedelta

            before = (date.fromisoformat(r["start"]) - timedelta(days=1)).isoformat()
            for metric, name in [
                ("assets", "return_on_assets"),
                ("equity", "return_on_equity"),
            ]:
                a, b = [by_key.get((metric, "", end)) for end in (before, r["end"])]
                if a and b and min(a["value"], b["value"]) > 0:
                    add(
                        name, 200 * r["value"] / (a["value"] + b["value"]), "%", r, a, b
                    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["demo", "refresh", "init"])
    parser.add_argument("tickers", nargs="*")
    args = parser.parse_args()
    engine = database()
    if args.action == "demo":
        for file in sorted((ROOT / "snapshots").glob("*.json")):
            raw = file.read_text()
            obj = json.loads(raw)
            load(
                engine, obj["ticker"], raw, obj["url"], obj["capture"], obj["retrieved"]
            )
    if args.action == "refresh":
        for ticker in args.tickers or CIKS:
            refresh(engine, ticker)
