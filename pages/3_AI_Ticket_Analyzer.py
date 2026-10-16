import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics, charts, data_loader, explainability, rules, ticket_classifier, ui
from src.data_validation import ValidationError, detect_columns
from src.preprocessing import build_text, build_text_series
from src.sla_risk import build_features, tier

df = ui.page_context()
full = st.session_state["df"]
ui.page_header("AI Ticket Analyzer", "Classify a ticket, get a priority recommendation, and see which words drove the prediction.")
dk = ui.dark()
clf, msg = ui.get_classifier("category")
if clf is None:
    ui.empty_state("Category model not available", msg or "")
    ui.note("The analyzer needs a trained category model. Load a dataset with at least 60 labelled tickets.", "warn")
    st.stop()
prio, _ = ui.get_classifier("priority")

SAMPLES = {
    "Payment problem": ("Charged twice for my subscription", "I was billed twice this month and need a refund for the duplicate payment."),
    "Production outage": ("Service is down for all users", "Our whole team cannot reach the service and production is blocked. Urgent, this is an outage."),
    "Login trouble": ("Cannot log in after password reset", "The reset email arrived but my new password is rejected and the account is locked."),
    "API error": ("API returns 401 unauthorized", "Valid key but every call fails with 401 since yesterday; webhook delivery also stopped."),
    "Feature idea": ("Request: dark mode", "Would be great to have a dark theme for the dashboard."),
}
def _load_sample():
    s, d = SAMPLES[st.session_state["an_sample"]]
    st.session_state["an_subject"], st.session_state["an_desc"] = s, d
st.session_state.setdefault("an_subject", "")
st.session_state.setdefault("an_desc", "")

tab1, tab2 = st.tabs(["Single ticket", "Batch (CSV)"])
with tab1:
    cs, cb = st.columns([3, 1])
    cs.selectbox("Sample tickets", list(SAMPLES), key="an_sample")
    cb.button("Load sample", on_click=_load_sample, width="stretch")
    with st.form("analyze"):
        subject = st.text_input("Subject", key="an_subject")
        desc = st.text_area("Description", key="an_desc", height=130)
        c1, c2, c3 = st.columns(3)
        prods = ["(not specified)"] + (sorted(full["product"].dropna().unique()) if "product" in full else [])
        chans = ["(not specified)"] + (sorted(full["channel"].dropna().unique()) if "channel" in full else [])
        product = c1.selectbox("Product (optional)", prods)
        channel = c2.selectbox("Channel (optional)", chans)
        customer = c3.text_input("Customer ID (optional)")
        go_ = st.form_submit_button("Analyze ticket", type="primary")
    text = build_text(subject, desc)
    if go_ and not text:
        st.error("Enter a subject or description first.")
    elif go_:
        low = st.session_state["low_conf"]
        pr = ticket_classifier.predict(clf.pipeline, [text], low)
        cat, conf, is_low = pr.loc[0, "predicted"], float(pr.loc[0, "confidence"]), bool(pr.loc[0, "low_confidence"])
        scores = ticket_classifier.class_scores(clf.pipeline, text)
        sig = rules.urgency_signals(text)
        ml_prio = None
        if prio is not None:
            ml_prio = ticket_classifier.predict(prio.pipeline, [text], 0)["predicted"].iloc[0]
        rec, why = rules.combine_priority(ml_prio, sig["rule_priority"])

        k1, k2, k3 = st.columns(3)
        with k1:
            ui.kpi_cards([("Predicted category", cat, "ML model", "Logistic regression on TF-IDF word + character n-grams.")])
        with k2:
            ui.kpi_cards([("Confidence", f"{conf:.0%}", "LOW – verify manually" if is_low else "above threshold", f"Softmax score, not a guarantee. Calibration error on the test set: {clf.ece:.3f}. Low-confidence threshold: {low:.2f}.")])
        with k3:
            ui.kpi_cards([("Recommended priority", rec, "model + keyword rule" if rec else "", why)])
        if is_low:
            ui.note(f"Low confidence (<{low:.0%}). Treat the category as a suggestion and review it manually.", "warn")
        st.markdown("**How the priority was decided**")
        ui.note(f"Model prediction: {ml_prio or 'not available (no priority labels / model)'}. Keyword rule: {sig['rule_priority'] or 'no urgency keywords found'}"
                f"{' (' + ', '.join(sig['strong'] + sig['weak']) + ')' if sig['strong'] or sig['weak'] else ''}. {why}", "rule")

        a, b = st.columns(2)
        with a:
            fig = px.bar(scores.head(6).sort_values("score"), x="score", y="class", orientation="h", title="Class scores (top 6)")
            fig.update_traces(marker_color=charts.COLORWAY[0]); fig.update_layout(xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
            st.plotly_chart(charts.style(fig, dk, 300, False), width="stretch")
        with b:
            ex = explainability.explain_prediction(clf.pipeline, text, cat, 8)
            if ex.empty:
                st.info("No known terms from the training vocabulary were found in this text.")
            else:
                fig = px.bar(ex.sort_values("contribution"), x="contribution", y="term", orientation="h", title=f"Terms pushing toward “{cat}”")
                fig.update_traces(marker_color=charts.COLORWAY[2]); fig.update_layout(yaxis_title=None, xaxis_title="TF-IDF × coefficient")
                st.plotly_chart(charts.style(fig, dk, 300, False), width="stretch")

        st.markdown("**SLA breach risk**")
        bm, bmsg = ui.get_breach_model()
        if bm is None:
            ui.note(f"SLA breach probability not available: {bmsg}", "warn")
        else:
            row = pd.DataFrame([{"text": text, "priority": rec or "Unknown", "category": cat, "product": None if product.startswith("(") else product,
                                 "channel": None if channel.startswith("(") else channel, "created_at": pd.Timestamp.now()}])
            p = float(bm.pipeline.predict_proba(build_features(row))[0, 1])
            ui.note(f"ML estimate: {p:.0%} ({tier(p)} risk) — from a model trained on {bm.n_train:,} past tickets; test PR-AUC {bm.metrics['pr_auc']:.2f} vs {bm.metrics['baseline_pr_auc']:.2f} for random guessing. "
                    "Probabilities are not calibrated guarantees and use the current time as the creation time.", "model")

        st.markdown("**Recommended next actions** (transparent business rules, not model output)")
        for a_ in rules.recommended_actions(cat, rec, sig["strong"]):
            st.markdown(f"- {a_}")
        with st.expander("Limitations"):
            st.markdown("- Model predictions depend on the vocabulary and label quality of the training data; unfamiliar wording lowers accuracy.\n"
                        "- Confidence is a model score, not a probability of being correct.\n- Keyword rules are simple string matches and can miss or over-trigger.\n"
                        "- Customer ID is shown for context only and is not used by any model.")

with tab2:
    st.write("Upload a CSV with a **subject** and/or **description** column (other columns are kept in the output).")
    up = st.file_uploader("CSV file", type=["csv"], key="batch_up")
    if up is not None:
        try:
            raw, _ = data_loader.read_csv_bytes(up.getvalue())
            mp = detect_columns(raw.columns)
            if not (mp["subject"] or mp["description"]):
                st.error("Could not find a subject or description column. Rename a text column to 'subject' or 'description'.")
            else:
                txt = build_text_series(pd.DataFrame({"subject": raw[mp["subject"]] if mp["subject"] else "", "description": raw[mp["description"]] if mp["description"] else ""}))
                with st.spinner(f"Predicting {len(raw):,} tickets…"):
                    out = ticket_classifier.predict(clf.pipeline, txt.tolist(), st.session_state["low_conf"])
                res = pd.concat([raw.reset_index(drop=True), out.rename(columns={"predicted": "predicted_category"})], axis=1)
                st.success(f"Predicted {len(res):,} tickets; {int(res['low_confidence'].sum()):,} flagged low-confidence.")
                st.dataframe(res.head(200), width="stretch", hide_index=True)
                st.download_button("Download predictions (CSV)", analytics.to_csv_bytes(res), "predictions.csv", "text/csv")
        except ValidationError as exc:
            st.error(str(exc))
