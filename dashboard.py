import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select

from live import ModelFailure, installed_models, live_research
from pipeline import (
    companies,
    database,
    evaluations,
    facts,
    get_financial_facts,
    get_financial_metrics,
    runs,
)
from research import Rules, research
from rag import document_research, add_filing_evidence
from rag_docs import ROOT, ITEMS
from rag_retrieve import METHODS

st.set_page_config(page_title="FinPulse AI", layout="wide")
st.title("FinPulse AI")
engine = database()
with engine.connect() as conn:
    issuers = dict(conn.execute(select(companies.c.ticker, companies.c.name)).all())
if not issuers:
    st.info("Load the included SEC excerpts with python pipeline.py demo.")
    st.stop()
ticker = st.sidebar.selectbox(
    "Company", list(issuers), format_func=lambda t: f"{t} · {issuers[t]}"
)
cutoff = st.sidebar.date_input("Filings available by", date.today()).isoformat()
company, debate, report, ops, documents = st.tabs(
    [
        "Company intelligence",
        "Bull vs Bear",
        "Research report",
        "Pipeline & evaluation",
        "Document Research",
    ]
)
with engine.connect() as conn:
    data = get_financial_facts(conn, ticker, cutoff)
    metrics = get_financial_metrics(conn, ticker, cutoff)
with company:
    st.header(issuers[ticker])
    if data:
        st.caption(
            f"Latest included filing {max(r['filed'] for r in data)} · Latest reporting date {max(r['end'] for r in data)}"
        )
        annual = pd.DataFrame(
            [
                r
                for r in data
                if r["kind"] == "annual"
                and r["metric"] in {"revenue", "net_income", "operating_income"}
            ]
        )
        if not annual.empty:
            cols = st.columns(3)
            for col, (metric, group) in zip(cols, annual.groupby("metric")):
                row = group.sort_values("end").iloc[-1]
                col.metric(metric.replace("_", " ").title(), f"${row.value/1e9:,.2f}B")
                col.caption(f"Year ended {row.end}")
            annual["USD billions"] = annual.value / 1e9
            st.plotly_chart(
                px.bar(
                    annual,
                    x="end",
                    y="USD billions",
                    color="metric",
                    barmode="group",
                    labels={"end": "Fiscal year end", "metric": "Measure"},
                ),
                use_container_width=True,
            )
        st.dataframe(pd.DataFrame(metrics), hide_index=True)
        with st.expander("Filing evidence"):
            st.dataframe(
                pd.DataFrame(data)[
                    [
                        "metric",
                        "start",
                        "end",
                        "kind",
                        "value",
                        "unit",
                        "filed",
                        "accession",
                        "url",
                    ]
                ],
                hide_index=True,
            )
    else:
        st.info("No filings in this snapshot meet the selected cutoff.")


@st.cache_data(ttl=5)
def available_models():
    return installed_models()


models = [] if os.getenv("FINPULSE_OFFLINE") == "1" else available_models()
mode = st.sidebar.radio(
    "Research mode",
    ["Live local LLM", "Deterministic offline"],
    index=0 if models else 1,
)
model = (
    st.sidebar.selectbox("Local model", models)
    if models and mode == "Live local LLM"
    else None
)
if mode == "Live local LLM" and not model:
    st.sidebar.info("Start the local model service, then refresh this page.")
with debate:
    with st.expander("Optional filing evidence"):
        doc_question = st.text_input(
            "Filing question for the research report", max_chars=600
        ).strip()
    key = (ticker, cutoff, mode, model, doc_question)
    if st.button(
        "Run research", disabled=not data or (mode == "Live local LLM" and not model)
    ):
        st.session_state.pop("research", None)
        try:
            with st.spinner("Reading filing evidence and checking claims…"):
                generated = (
                    live_research(engine, ticker, cutoff, model=model)
                    if model
                    else research(engine, ticker, cutoff, Rules())
                )
            if doc_question:
                generated = add_filing_evidence(
                    engine, generated, doc_question, llm=model
                )
            st.session_state["research"] = (key, generated)
        except (ModelFailure, ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("research")
    result = saved[1] if saved and saved[0] == key else None
    if result:
        st.caption(
            " · ".join(filter(None, [mode, result["provider"], result.get("model")]))
        )
        for col, (side, item) in zip(st.columns(2), result["arguments"].items()):
            col.subheader(side)
            col.write(item["statement"])
            col.caption(item["status"])
            if item["claim"]:
                col.json(item["claim"], expanded=False)
        st.subheader("Cross-examination")
        for line in result["challenges"]:
            st.write(line)
with report:
    if result:
        st.markdown(result["markdown"])
        st.download_button(
            "Download report", result["markdown"], f"{ticker}-research.md"
        )
    else:
        st.info("Run research in the Bull vs Bear tab.")
with ops:
    with engine.connect() as conn:
        st.metric(
            "Stored financial observations",
            conn.execute(select(func.count()).select_from(facts)).scalar(),
        )
        st.dataframe(
            pd.read_sql(select(runs).order_by(runs.c.timestamp.desc()), conn),
            hide_index=True,
        )
        for raw in conn.execute(select(evaluations.c.result)).scalars():
            st.json(json.loads(raw), expanded=False)
    log = Path(os.getenv("TELEMETRY_PATH", "telemetry.jsonl"))
    if log.exists():
        st.dataframe(pd.read_json(log, lines=True).tail(50), hide_index=True)

with documents:
    st.header("Document Research")
    st.caption(f"{ticker} annual filing · available by {cutoff}")
    question = st.text_input(
        "Question about the filing",
        max_chars=600,
        placeholder="Which activities drove growth in Services?",
    )
    method = st.selectbox("Retrieval method", METHODS, index=2)
    perspective = st.selectbox("Research perspective", ["Analyst", "Bull", "Bear"])
    doc_key = (ticker, cutoff, question.strip(), method, perspective, model)
    if st.button("Research the filing", disabled=not question.strip()):
        st.session_state.pop("document_result", None)
        try:
            with st.spinner("Finding filing passages…"):
                doc = document_research(
                    question,
                    ticker,
                    cutoff,
                    method,
                    perspective=perspective,
                    llm=model,
                    generate=mode == "Live local LLM",
                )
            st.session_state["document_result"] = (doc_key, doc)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Document research could not run: {exc}")
    cached = st.session_state.get("document_result")
    if cached and cached[0] == doc_key:
        doc = cached[1]
        st.caption(
            f"Retrieval {doc['retrieval_ms']:.1f} ms · Total {doc['end_to_end_ms']/1000:.2f} s"
        )
        st.write(doc["answer"])
        for i, c in enumerate(doc["citations"], 1):
            st.markdown(
                f"[{i}. {ticker} 10-K · Item {c['section']} · filed {c['filed']}]({c['url']})"
            )
            st.caption(c["accession"])
            st.text(c["excerpt"])
        for r in doc["passages"]:
            with st.expander(
                f"Item {r['section']} · {ITEMS[r['section']]} · score {r['score']:.4f}"
            ):
                st.write(r["text"])
                st.caption(f"{r['id']} · {r['accession']} · filed {r['filed']}")
with ops:
    comparison = ROOT / "eval/document-summary.json"
    if comparison.exists():
        st.subheader("Document retrieval evaluation")
        st.caption(
            "20 questions · retrieval metrics over 14 answerable questions · top three chunks"
        )
        st.dataframe(
            pd.DataFrame(json.loads(comparison.read_text())["all"])[
                [
                    "method",
                    "precision_at_3",
                    "recall_at_3",
                    "mrr_at_3",
                    "correct",
                    "queries",
                ]
            ].rename(
                columns={
                    "method": "Method",
                    "precision_at_3": "Precision@3",
                    "recall_at_3": "Recall@3",
                    "mrr_at_3": "MRR@3",
                    "correct": "Correct answers",
                    "queries": "Questions",
                }
            ),
            hide_index=True,
        )
