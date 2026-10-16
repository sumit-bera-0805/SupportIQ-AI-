import pandas as pd
import streamlit as st

from src import config, data_loader, ui
from src.data_validation import ValidationError, detect_columns, validate_and_clean

ui.page_context()
ui.page_header("Data Import & Configuration", "Upload your own tickets, map columns, review data quality, and tune app settings.")
ss = st.session_state

c1, c2 = st.columns([2, 1])
with c2:
    st.markdown("**Current dataset**")
    st.write(f"{ss['dataset_name']} — {len(ss['df']):,} rows")
    st.download_button("Download sample CSV", config.SAMPLE_PATH.read_bytes(), "sample_tickets.csv", "text/csv")
    if st.button("Reset to demo dataset", disabled=ss["is_demo"], width="stretch"):
        ui.reset_to_demo()
        st.rerun()
    st.caption(f"Limits: {config.MAX_UPLOAD_MB} MB, {config.MAX_ROWS:,} rows. Recommended for a 512 MB free instance: ≤ 20,000 rows.")
with c1:
    st.markdown("**1 · Upload a CSV**")
    st.caption("Drag and drop a comma-separated file with a header row. Data stays in your browser session and is never sent to third parties.")
    up = st.file_uploader("Ticket CSV", type=["csv"], key="main_up")

if up is not None:
    try:
        raw, skipped = data_loader.read_csv_bytes(up.getvalue())
    except ValidationError as exc:
        st.error(str(exc))
        st.stop()
    st.markdown("**2 · Preview and column mapping**")
    st.dataframe(raw.head(10), width="stretch", hide_index=True)
    auto = detect_columns(raw.columns)
    st.caption("Columns were matched automatically by name. Adjust any that are wrong.")
    options = ["(not mapped)"] + list(raw.columns)
    mapping = {}
    cols = st.columns(3)
    for i, canon in enumerate(config.CANONICAL_COLUMNS):
        default = options.index(auto[canon]) if auto[canon] in options else 0
        choice = cols[i % 3].selectbox(canon, options, index=default, key=f"map_{canon}_{up.file_id}")
        mapping[canon] = None if choice == "(not mapped)" else choice
    policy = st.radio("Missing-value handling", ["keep", "label_unknown", "drop_incomplete"], horizontal=True,
                      format_func={"keep": "Keep missing as missing (recommended)", "label_unknown": "Label missing priority/status/product/channel 'Unknown'", "drop_incomplete": "Drop rows missing category, priority or created_at"}.get)
    if st.button("Validate and analyse", type="primary"):
        try:
            with st.spinner("Validating…"):
                res = validate_and_clean(raw, mapping, policy, skipped)
            ss["_candidate"] = (raw, res, up.name)
        except ValidationError as exc:
            ss.pop("_candidate", None)
            st.error(str(exc))
    cand = ss.get("_candidate")
    if cand and cand[2] == up.name:
        _, res, name = cand
        st.markdown("**3 · Data quality report**")
        s = res.stats
        a, b, c, d = st.columns(4)
        a.metric("Rows read", f"{s['rows_in']:,}"); b.metric("Rows kept", f"{s['rows_out']:,}")
        c.metric("Duplicate IDs removed", s.get("duplicate_ids_removed", 0)); d.metric("Empty-text rows removed", s.get("empty_text_removed", 0))
        for it in res.issues:
            {"error": st.error, "warning": st.warning}.get(it["level"], st.info)(it["message"])
        miss = pd.Series(s["missing_counts"]).rename("missing values").to_frame()
        miss["share"] = (miss["missing values"] / s["rows_out"]).map("{:.1%}".format)
        with st.expander("Missing values per column"):
            st.dataframe(miss, width="stretch")
        if any(s.get("invalid_dates", {}).values()):
            st.caption(f"Invalid dates by column: {s['invalid_dates']}")
        st.dataframe(res.df.drop(columns=["text"]).head(10), width="stretch", hide_index=True)
        if st.button("Use this dataset", type="primary"):
            ui.set_dataset(raw, res.df, name, s, res.issues)
            ss.pop("_candidate", None)
            st.success("Dataset loaded. Open any page to explore it.")
            st.rerun()

st.divider()
st.markdown("**Configuration**")
ss["low_conf"] = st.slider("Low-confidence threshold for predictions", 0.1, 0.9, float(ss["low_conf"]), 0.05, help="Predictions with a top-class score below this are flagged for manual review.")
st.caption("SLA hour thresholds per priority are configured on the SLA Risk page. Theme is switched in the sidebar.")
if ss["validation_issues"]:
    with st.expander("Validation notes for the current dataset"):
        for it in ss["validation_issues"]:
            st.write(f"[{it['level']}] {it['message']}")
