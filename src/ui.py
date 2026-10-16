"""Shared Streamlit UI: session state, theming, sidebar, global filters, KPI cards, model caches.

All user data lives in st.session_state (per browser session) - nothing is shared between users."""
from __future__ import annotations

import html
import time
from datetime import date

import pandas as pd
import streamlit as st

from . import analytics, config, data_loader, storage
from .sla_risk import SLAModelUnavailable, train_breach_model
from .ticket_classifier import InsufficientDataError, TrainResult, train_classifier

FILTER_KEYS = ["f_category", "f_priority", "f_status", "f_product", "f_channel"]
FILTER_COLS = {"f_category": "category", "f_priority": "priority", "f_status": "status", "f_product": "product", "f_channel": "channel"}
THEMES = {
    "dark": dict(bg="#0b1220", panel="#111a2e", panel2="#16223b", border="#243250", text="#e6edf7", muted="#94a3b8", blue="#3b82f6", cyan="#22d3ee", violet="#a78bfa", red="#f87171"),
    "light": dict(bg="#f4f7fb", panel="#ffffff", panel2="#eef3fb", border="#d6deea", text="#0f172a", muted="#52627a", blue="#2563eb", cyan="#0891b2", violet="#7c3aed", red="#dc2626"),
}


# ---------------------------------------------------------------- session / dataset
def init_session() -> None:
    ss = st.session_state
    if "dataset_hash" in ss:
        return
    t0 = time.perf_counter()
    raw, df, stats = data_loader.load_sample()
    ss.update(raw_df=raw, df=df, dataset_name="Demo dataset (synthetic)", is_demo=True, dataset_hash=data_loader.dataframe_hash(df),
              validation_stats=stats, validation_issues=[], theme="dark", low_conf=config.LOW_CONFIDENCE_DEFAULT,
              sla_hours=dict(config.DEFAULT_SLA_HOURS), _memo={}, load_seconds=time.perf_counter() - t0)
    storage.init_db()


def set_dataset(raw: pd.DataFrame, df: pd.DataFrame, name: str, stats: dict, issues: list[dict], is_demo: bool = False) -> None:
    ss = st.session_state
    ss.update(raw_df=raw, df=df, dataset_name=name, is_demo=is_demo, dataset_hash=data_loader.dataframe_hash(df),
              validation_stats=stats, validation_issues=issues, _memo={})
    for k in FILTER_KEYS + ["f_dates"]:
        ss.pop(k, None)


def reset_to_demo() -> None:
    raw, df, stats = data_loader.load_sample()
    set_dataset(raw, df, "Demo dataset (synthetic)", stats, [], is_demo=True)


def memo(name: str, fn):
    """Per-session, per-dataset cache (cleared automatically when the dataset changes)."""
    store = st.session_state.setdefault("_memo", {})
    key = (st.session_state["dataset_hash"], name)
    if key not in store:
        store[key] = fn()
    return store[key]


def dark() -> bool:
    return st.session_state.get("theme", "dark") == "dark"


# ---------------------------------------------------------------- models
def get_classifier(task: str = "category", auto: bool = True) -> tuple[TrainResult | None, str | None]:
    """Return (result, message). Trains lazily once per dataset; message explains why not."""
    df = st.session_state["df"]
    if task not in df or df[task].notna().sum() == 0:
        return None, f"No '{task}' labels in this dataset, so a {task} model cannot be trained."
    key = f"clf_{task}"
    store = st.session_state.setdefault("_memo", {})
    mk = (st.session_state["dataset_hash"], key)
    if mk in store:
        return store[mk]
    if auto and len(df) > config.AUTO_TRAIN_MAX_ROWS and not st.session_state.get(f"force_{key}"):
        return None, f"Large dataset ({len(df):,} rows): training is not automatic. Start it from the Model Performance page."
    try:
        with st.spinner(f"Training {task} classifier (TF-IDF + Logistic Regression)…"):
            res = train_classifier(df["text"], df[task], task=task)
        out = (res, None)
        storage.log_model_run(st.session_state["dataset_name"], len(df), task, "LogReg+TFIDF", {**res.metrics, "ece": res.ece})
    except InsufficientDataError as exc:
        out = (None, str(exc))
    store[mk] = out
    return out


def get_breach_model():
    df = st.session_state["df"]

    def build():
        try:
            return train_breach_model(df), None
        except SLAModelUnavailable as exc:
            return None, str(exc)
    return memo("breach_model", build)


# ---------------------------------------------------------------- theme + chrome
def apply_theme() -> None:
    t = THEMES[st.session_state.get("theme", "dark")]
    vars_css = ":root{" + "".join(f"--{k}:{v};" for k, v in t.items()) + "}"
    css = config.STYLES_PATH.read_text(encoding="utf-8") if config.STYLES_PATH.exists() else ""
    st.markdown(f"<style>{vars_css}\n{css}</style>", unsafe_allow_html=True)


def sidebar_status() -> None:
    ss = st.session_state
    sb = st.sidebar
    sb.markdown("### SupportIQ AI")
    badge = '<span class="siq-badge demo">DEMO DATA</span>' if ss["is_demo"] else '<span class="siq-badge ok">YOUR DATA</span>'
    sb.markdown(f"{badge}<br><span class='muted'>{html.escape(ss['dataset_name'])} · {len(ss['df']):,} tickets</span>", unsafe_allow_html=True)
    store = ss.get("_memo", {})
    clf = store.get((ss["dataset_hash"], "clf_category"))
    if clf and clf[0]:
        sb.markdown(f'<span class="siq-badge ok">Classifier ready</span> <span class="muted">macro-F1 {clf[0].metrics["macro_f1"]:.2f}</span>', unsafe_allow_html=True)
    elif clf:
        sb.markdown('<span class="siq-badge warn">Classifier unavailable</span>', unsafe_allow_html=True)
    else:
        sb.markdown('<span class="siq-badge warn">Classifier not trained yet</span>', unsafe_allow_html=True)
    sb.toggle("Light theme", value=ss["theme"] == "light", key="theme_toggle", on_change=lambda: ss.update(theme="light" if ss["theme_toggle"] else "dark"))


def global_filters() -> pd.DataFrame:
    """Render sidebar filters (state persists across pages) and return the filtered frame."""
    ss = st.session_state
    for k, v in ss.pop("_pending", {}).items():   # set by chart-click handlers before widgets exist
        ss[k] = v
    df = ss["df"]
    opts = analytics.filter_options(df)
    with st.sidebar.expander("Filters", expanded=False):
        if "created_at" in df and df["created_at"].notna().any():
            lo, hi = df["created_at"].min().date(), df["created_at"].max().date()
            st.date_input("Created between", value=(lo, hi), min_value=lo, max_value=hi, key="f_dates")
        sel = {}
        for key, col in FILTER_COLS.items():
            if col in opts:
                ss[key] = [v for v in ss.get(key, []) if v in opts[col]]
                sel[col] = st.multiselect(col.title(), opts[col], key=key, placeholder="All")
        if st.button("Reset filters", width="stretch"):
            for k in FILTER_KEYS + ["f_dates"]:
                ss.pop(k, None)
            st.rerun()
    dr = ss.get("f_dates")
    dr = tuple(dr) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None
    out = analytics.apply_filters(df, dr, **sel)
    active = len(out) != len(df)
    st.sidebar.markdown(f"<span class='muted'>{len(out):,} of {len(df):,} tickets match{' (filtered)' if active else ''}</span>", unsafe_allow_html=True)
    return out


def set_filter_from_chart(key: str, value: str) -> None:
    """Apply a chart click as a filter (takes effect on the next run, before widgets render)."""
    if st.session_state.get(key) != [value]:
        st.session_state["_pending"] = {key: [value]}
        st.rerun()


# ---------------------------------------------------------------- components
def page_header(title: str, subtitle: str) -> None:
    ss = st.session_state
    badge = '<span class="siq-badge demo">DEMO DATA · synthetic</span>' if ss["is_demo"] else '<span class="siq-badge ok">Your dataset</span>'
    st.markdown(f'<div class="siq-hero">{badge}<h1>{html.escape(title)}</h1><div class="muted">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    cls = {"warn": "warn", "model": "model", "rule": "rule"}.get(kind, "")
    st.markdown(f'<div class="siq-note {cls}">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpi_cards(items: list[tuple]) -> None:
    """items: (label, value|None, sub, tooltip). None -> 'Not available'."""
    cards = []
    for label, value, sub, tip in items:
        v = f'<div class="val">{html.escape(str(value))}</div>' if value is not None else '<div class="val na">Not available</div>'
        cards.append(f'<div class="siq-card"><div class="lbl" title="{html.escape(tip, quote=True)}">{html.escape(label)} ⓘ</div>{v}<div class="sub">{sub}</div></div>')
    st.markdown('<div class="siq-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def empty_state(title: str, detail: str) -> None:
    st.markdown(f'<div class="siq-card" style="text-align:center;padding:28px"><div class="val na">{html.escape(title)}</div><div class="sub">{html.escape(detail)}</div></div>', unsafe_allow_html=True)


def today() -> date:
    return date.today()


def page_context() -> pd.DataFrame:
    """Called at the top of every page: makes pages runnable standalone and returns the filtered frame."""
    init_session()
    if "filtered" not in st.session_state:
        apply_theme()
    return st.session_state.get("filtered", st.session_state["df"])
