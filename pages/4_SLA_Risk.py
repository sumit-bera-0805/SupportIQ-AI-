import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics, charts, sla_risk, ui

df = ui.page_context()
full = st.session_state["df"]
ui.page_header("SLA Risk Intelligence", "Deterministic deadline monitoring plus a separately-labelled machine-learning breach model.")
dk = ui.dark()
if df.empty:
    ui.empty_state("No tickets match the current filters", "Widen or reset the filters in the sidebar.")
    st.stop()

has_dl = "sla_deadline" in df and df["sla_deadline"].notna().any()
with st.expander("SLA configuration", expanded=not has_dl):
    st.caption("Hours allowed per priority. Used ONLY for tickets without a dataset deadline; such deadlines are marked as policy assumptions.")
    cols = st.columns(4)
    for c, p in zip(cols, ["Critical", "High", "Medium", "Low"]):
        st.session_state["sla_hours"][p] = c.number_input(f"{p} (hours)", 1, 24 * 30, int(st.session_state["sla_hours"][p]), key=f"sla_{p}")
    use_policy = st.checkbox("Apply the policy to tickets without a dataset deadline", value=not has_dl)
    frac = st.slider("'At risk' when remaining time is below this share of the SLA window", 0.05, 0.75, 0.25, 0.05)

as_of = sla_risk.as_of_time(full)
mon = sla_risk.monitor(df, as_of, st.session_state["sla_hours"], frac, use_policy)
summ = sla_risk.breach_summary(mon)
if as_of is None or (mon["sla_state"] == "Unknown").all():
    ui.empty_state("SLA monitoring not available", "Map created_at and either sla_deadline or priority (for policy deadlines).")
    st.stop()
st.caption(f"Reference time (“now”) = latest timestamp in the data: {as_of:%d %b %Y %H:%M} UTC")
if summ["policy_deadlines"]:
    ui.note(f"{summ['policy_deadlines']:,} ticket(s) use POLICY deadlines from the configuration above, not real SLA data.", "warn")
cnt = mon["sla_state"].value_counts()
ui.kpi_cards([
    ("Breach rate", None if summ["rate"] is None else f"{summ['rate'] * 100:.1f}%", f"{summ['n_known']:,} tickets with outcomes", analytics.KPI_DEFINITIONS["SLA breach rate"]),
    ("Open & overdue", f"{int(cnt.get('Breached (open, overdue)', 0)):,}", "already breached", "Unresolved tickets past their deadline."),
    ("At risk (rule)", f"{int(cnt.get('At risk', 0)):,}", "rule-based, not ML", "Unresolved, not yet overdue, with less than the configured share of the window left."),
    ("Resolved late", f"{int(cnt.get('Breached (resolved late)', 0)):,}", "historical", "Resolved after the deadline."),
    ("Met", f"{int(cnt.get('Met', 0)):,}", "historical", "Resolved on or before the deadline."),
])

t1, t2, t3 = st.tabs(["Deadline monitor (rules)", "Breach prediction (ML)", "Breach drivers"])
with t1:
    ui.note("Everything on this tab is deterministic rule-based monitoring. No machine learning is involved.", "rule")
    live = mon[mon["sla_state"].isin(["Breached (open, overdue)", "At risk", "On track"])].copy()
    if live.empty:
        ui.empty_state("No open tickets with deadlines", "Nothing to monitor in the current filter.")
    else:
        a, b, c = st.columns(3)
        states = a.multiselect("State", ["Breached (open, overdue)", "At risk", "On track"], default=["Breached (open, overdue)", "At risk"])
        sort_by = b.selectbox("Sort by", ["hours_remaining", "priority", "age_hours", "deadline"])
        asc = c.toggle("Ascending", value=True)
        v = live[live["sla_state"].isin(states)]
        if sort_by == "priority":
            v = v.assign(_p=v["priority"].map({p: i for i, p in enumerate(["Critical", "High", "Medium", "Low"])})).sort_values("_p", ascending=asc).drop(columns="_p")
        else:
            v = v.sort_values(sort_by, ascending=asc, na_position="last")
        v = v.assign(recommended_action=v["sla_state"].map(lambda s: sla_risk.recommended_action(s)))
        show = [c for c in ["ticket_id", "subject", "priority", "status", "sla_state", "deadline", "deadline_source", "hours_remaining", "age_hours", "recommended_action"] if c in v]
        st.dataframe(v[show], width="stretch", hide_index=True, column_config={"hours_remaining": st.column_config.NumberColumn(format="%.1f"), "age_hours": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("Download monitored tickets (CSV)", analytics.to_csv_bytes(v[show]), "sla_monitor.csv", "text/csv")

with t2:
    bm, bmsg = ui.get_breach_model()
    if bm is None:
        ui.empty_state("ML breach model not trained", bmsg or "")
        st.caption("A trained model requires historical outcomes: sla_deadline + resolved_at + created_at for at least 100 resolved tickets, with enough breached and met examples. Rule-based monitoring (left tab) remains available.")
    else:
        ui.note(f"Machine-learning model (logistic regression on creation-time features only). Trained on tickets created before {bm.split_date:%d %b %Y}; evaluated chronologically on the {bm.n_test:,} tickets after it. "
                f"Train n={bm.n_train:,}.", "model")
        m = bm.metrics
        ui.kpi_cards([("PR-AUC", f"{m['pr_auc']:.3f}", f"random baseline {m['baseline_pr_auc']:.3f}", "Area under the precision-recall curve on the future test period. Compare with the baseline (= breach prevalence)."),
                      ("Precision @0.5", f"{m['precision']:.3f}", "of flagged, truly breached", "Share of predicted breaches that really breached."),
                      ("Recall @0.5", f"{m['recall']:.3f}", "of breaches, caught", "Share of real breaches the model flagged."),
                      ("F1 @0.5", f"{m['f1']:.3f}", "", "Harmonic mean of precision and recall at a 0.5 threshold."),
                      ("Brier score", f"{m['brier']:.3f}", "lower is better", "Mean squared error of the predicted probabilities.")])
        for n_ in bm.notes:
            st.caption(n_)
        cal = bm.calibration
        fig = px.line(cal, x="mean_predicted", y="observed_rate", markers=True, title="Calibration (test period, quantile bins)")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#94a3b8"))
        fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed breach rate")
        st.plotly_chart(charts.style(fig, dk, 320, False), width="stretch")
        scored = sla_risk.score_open_tickets(bm, mon)
        st.subheader("Open tickets ranked by predicted breach risk")
        if scored.empty:
            st.info("No open, not-yet-overdue tickets to score in the current filter.")
        else:
            a, b = st.columns(2)
            tiers = a.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
            sort_by = b.selectbox("Sort by", ["breach_probability", "hours_remaining", "age_hours", "priority"], key="ml_sort")
            v = scored[scored["risk_tier"].isin(tiers)]
            v = v.sort_values(sort_by, ascending=sort_by in ("hours_remaining", "priority"), na_position="last")
            v = v.assign(recommended_action=[sla_risk.recommended_action(s, t) for s, t in zip(v["sla_state"], v["risk_tier"])])
            show = [c for c in ["ticket_id", "subject", "priority", "breach_probability", "risk_tier", "hours_remaining", "age_hours", "top_factors", "recommended_action"] if c in v]
            st.dataframe(v[show], width="stretch", hide_index=True, column_config={"breach_probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
            st.download_button("Download ML risk scores (CSV)", analytics.to_csv_bytes(v[show]), "sla_ml_risk.csv", "text/csv")

with t3:
    known = mon[mon["sla_state"].isin(["Met", "Breached (resolved late)", "Breached (open, overdue)"])].assign(breached=lambda d: d["sla_state"].str.startswith("Breached"))
    for col in ("priority", "category", "channel", "product"):
        if col in known and known[col].notna().any():
            g = known.groupby(col)["breached"].agg(["mean", "size"]).reset_index().rename(columns={"mean": "breach_rate", "size": "tickets"})
            fig = px.bar(g.sort_values("breach_rate"), x="breach_rate", y=col, orientation="h", title=f"Breach rate by {col}", hover_data=["tickets"])
            fig.update_traces(marker_color=charts.COLORWAY[0]); fig.update_layout(xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
            st.plotly_chart(charts.style(fig, dk, 280, False), width="stretch")
