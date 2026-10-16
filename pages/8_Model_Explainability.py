import plotly.express as px
import streamlit as st

from src import analytics, charts, explainability, storage, ticket_classifier, ui

df = ui.page_context()
full = st.session_state["df"]
ui.page_header("Model Performance & Explainability", "Honest hold-out evaluation of the TF-IDF classifiers. Metrics appear only after a valid evaluation.")
dk = ui.dark()
task = st.radio("Target", ["category", "priority"], horizontal=True, format_func=str.title)
key = f"clf_{task}"
if len(full) > ui.config.AUTO_TRAIN_MAX_ROWS and (st.session_state["dataset_hash"], key) not in st.session_state.get("_memo", {}):
    ui.note(f"Large dataset ({len(full):,} rows). Training takes a while on small instances.", "warn")
    if st.button(f"Train {task} model", type="primary"):
        st.session_state[f"force_{key}"] = True
        st.rerun()
res, msg = ui.get_classifier(task)
if res is None:
    ui.empty_state("Model not available", msg or "")
    st.stop()

ui.note(f"Evaluation: stratified hold-out split, {res.n_train:,} train / {res.n_test:,} test tickets (80/20, seed 42). TF-IDF is fitted on the training portion only. "
        "Logistic Regression is fixed as the primary model in advance, so test scores are not selected by peeking.", "model")
for w in res.warnings:
    ui.note(w, "warn")
m = res.metrics
ui.kpi_cards([("Accuracy", f"{m['accuracy']:.3f}", f"n={res.n_test}", "Share of test tickets classified correctly."),
              ("Macro-F1", f"{m['macro_f1']:.3f}", "unweighted over classes", "Average F1 across classes; every class counts equally."),
              ("Weighted-F1", f"{m['weighted_f1']:.3f}", "weighted by support", "Average F1 weighted by class frequency."),
              ("Macro precision", f"{m['macro_precision']:.3f}", "", "Average precision across classes."),
              ("Macro recall", f"{m['macro_recall']:.3f}", "", "Average recall across classes."),
              ("Calibration error", f"{res.ece:.3f}", "ECE, lower is better", "Expected calibration error of the top-class probability on the test set."),
              ("Prediction latency", f"{res.latency_ms:.1f} ms", "per ticket, measured", "Median-style average over single-ticket predictions on this server."),
              ("Training time", f"{res.train_seconds:.1f} s", "LogReg + SVM", "Wall-clock seconds to fit both models on this server.")])
if st.session_state["is_demo"]:
    ui.note("These scores are for SYNTHETIC template-generated demo tickets and are higher than you should expect on real, messy support data.", "warn")

t1, t2, t3, t4 = st.tabs(["Per-class & confusion", "Errors & confidence", "Feature importance", "Model comparison & history"])
with t1:
    st.dataframe(res.per_class.round(3), width="stretch", hide_index=True)
    st.plotly_chart(charts.confusion(res.confusion, res.labels, dk), width="stretch")
    with st.expander("Classification report"):
        st.code(res.report_text)
with t2:
    tf = res.test_frame
    err = tf[tf["true"] != tf["predicted"]]
    st.markdown(f"**{len(err)} misclassified of {len(tf)} test tickets**")
    st.dataframe(err.sort_values("confidence", ascending=False).rename(columns={"text": "ticket text"}).round({"confidence": 3}), width="stretch", hide_index=True)
    tf = tf.assign(outcome=(tf["true"] == tf["predicted"]).map({True: "correct", False: "wrong"}))
    fig = px.histogram(tf, x="confidence", color="outcome", nbins=20, barmode="overlay", title="Confidence distribution (test set)", color_discrete_map={"correct": "#22d3ee", "wrong": "#f87171"})
    st.plotly_chart(charts.style(fig, dk, 320), width="stretch")
    st.caption("A well-behaved model puts most wrong predictions at low confidence; high-confidence errors are the most worth reviewing (table is sorted by that).")
with t3:
    n = st.slider("Terms per class", 5, 20, 10)
    tt = explainability.top_terms_per_class(res.pipeline, n)
    cls = st.selectbox("Class", res.labels)
    d = tt[tt["class"] == cls].sort_values("weight")
    fig = px.bar(d, x="weight", y="term", orientation="h", title=f"Top word n-grams for “{cls}” (logistic-regression coefficients)")
    fig.update_traces(marker_color=charts.COLORWAY[2]); fig.update_layout(yaxis_title=None)
    st.plotly_chart(charts.style(fig, dk, 380, False), width="stretch")
    st.download_button("Download all top terms (CSV)", analytics.to_csv_bytes(tt), "top_terms.csv", "text/csv")
with t4:
    st.dataframe(res.comparison.round(3), width="stretch", hide_index=True)
    st.caption("Linear SVM is shown for comparison only; it does not produce probabilities, so the app deploys Logistic Regression.")
    runs = storage.recent_model_runs(10)
    if runs:
        st.markdown("**Recent training runs (metadata only, stored in local SQLite)**")
        st.dataframe([{"when (UTC)": r["created_utc"], "dataset": r["dataset"], "rows": r["n_rows"], "task": r["task"], "macro-F1": round(r["metrics"].get("macro_f1", float("nan")), 3)} for r in runs], width="stretch", hide_index=True)
