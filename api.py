import json
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from live import ModelFailure, live_research
from pipeline import (
    companies,
    database,
    evaluations,
    get_financial_facts,
    get_financial_metrics,
    runs,
    sessions,
)
from research import Rules, research
from rag import document_research, add_filing_evidence

app = FastAPI(title="FinPulse AI", version="1.0")
engine = database()


class Request(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    document_question: str | None = Field(default=None, min_length=1, max_length=600)
    mode: Literal["deterministic", "live", "single_live"] | None = None
    as_of: date = Field(default_factory=date.today)


@app.get("/health")
def health():
    with engine.connect() as conn:
        conn.execute(select(1))
    return {"status": "ok", "database": engine.dialect.name}


def listing(table):
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(select(table)).mappings()]


@app.get("/companies")
def issuers():
    return listing(companies)


def financial(ticker, cutoff, tool):
    with engine.connect() as conn:
        if not conn.execute(
            select(companies.c.ticker).where(companies.c.ticker == ticker)
        ).scalar():
            raise HTTPException(404, "Issuer unavailable")
        return tool(conn, ticker, cutoff.isoformat())


@app.get("/companies/{ticker}/financials")
def financials(ticker: str, as_of: date | None = None):
    return financial(ticker, as_of or date.today(), get_financial_facts)


@app.get("/companies/{ticker}/metrics")
def metrics(ticker: str, as_of: date | None = None):
    return financial(ticker, as_of or date.today(), get_financial_metrics)


@app.post("/research")
def create(req: Request):
    try:
        if req.mode in ("live", "single_live"):
            result = live_research(
                engine,
                req.ticker,
                req.as_of.isoformat(),
                single=req.mode == "single_live",
            )
        else:
            result = research(
                engine,
                req.ticker,
                req.as_of.isoformat(),
                Rules() if req.mode == "deterministic" else None,
            )
        return (
            add_filing_evidence(engine, result, req.document_question)
            if req.document_question
            else result
        )
    except ModelFailure as exc:
        raise HTTPException(503, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/research/{session_id}")
def session(session_id: str):
    with engine.connect() as conn:
        raw = conn.execute(
            select(sessions.c.result).where(sessions.c.id == session_id)
        ).scalar()
    if raw is None:
        raise HTTPException(404, "Session unavailable")
    return json.loads(raw)


@app.get("/pipeline/runs")
def pipeline_runs():
    return listing(runs)


@app.get("/evaluations")
def evaluation_results():
    return [json.loads(r["result"]) for r in listing(evaluations)]


class DocumentRequest(BaseModel):
    ticker: Literal["AAPL", "MSFT", "NVDA"]
    question: str = Field(min_length=1, max_length=600)
    as_of: date = Field(default_factory=date.today)
    method: Literal["BM25", "Semantic", "Hybrid", "Hybrid + reranker"] = "Hybrid"
    perspective: Literal["Analyst", "Bull", "Bear"] = "Analyst"
    generate: bool = True


@app.post("/research/documents")
def documents(req: DocumentRequest):
    try:
        return document_research(
            req.question,
            req.ticker,
            req.as_of.isoformat(),
            req.method,
            perspective=req.perspective,
            generate=req.generate,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(
            503, "Document index or local retrieval model unavailable"
        ) from exc
