import pandas as pd
import streamlit as st

from src import analytics, charts, clustering, similarity_search as ss_, ui

df = ui.page_context()
full = st.session_state["df"]
ui.page_header("Similar Incident Search", "Find related past tickets by keywords or by example, and explore machine-generated clusters.")
dk = ui.dark()
if "text" not in full or full["text"].fillna("").str.len().eq(0).all():
    ui.empty_state("Nothing to search", "This dataset has no ticket text.")
    st.stop()

index = ui.memo("tfidf_index", lambda: ss_.TicketIndex(full))
if index.truncated:
    st.caption(f"Index limited to the most recent {len(index.df):,} tickets to protect memory.")

tab1, tab2, tab3 = st.tabs(["Keyword search", "Similar to a ticket", "Clusters"])

def filters(prefix: str) -> set | None:
    with st.expander("Result filters"):
        sel, mask = {}, pd.Series(True, index=index.df.index)
        cols = st.columns(3)
        for c, col in zip(cols, ["category", "priority", "product"]):
            if col in index.df and index.df[col].notna().any():
                sel[col] = c.multiselect(col.title(), sorted(index.df[col].dropna().unique()), key=f"{prefix}_{col}")
                if sel[col]:
                    mask &= index.df[col].isin(sel[col])
        if "created_at" in index.df and index.df["created_at"].notna().any():
            lo, hi = index.df["created_at"].min().date(), index.df["created_at"].max().date()
            dr = st.date_input("Created between", (lo, hi), min_value=lo, max_value=hi, key=f"{prefix}_dates")
            if isinstance(dr, (tuple, list)) and len(dr) == 2:
                mask &= (index.df["created_at"] >= pd.Timestamp(dr[0])) & (index.df["created_at"] < pd.Timestamp(dr[1]) + pd.Timedelta(days=1))
    return None if mask.all() else set(index.df.loc[mask, "ticket_id"])

def semantic_toggle(key: str) -> bool:
    if ss_.semantic_available():
        return st.checkbox("Use semantic embeddings (downloads a ~90 MB model on first use)", key=key)
    st.caption("Optional semantic mode is disabled: `sentence-transformers` is not installed. TF-IDF search is used.")
    return False

def show(res: pd.DataFrame) -> None:
    if res.empty:
        st.info("No matching tickets. Try different or fewer keywords, or relax the filters.")
        return
    st.dataframe(res, width="stretch", hide_index=True, column_config={"score": st.column_config.ProgressColumn("similarity", format="%.2f", min_value=0, max_value=1)})
    st.download_button("Download results (CSV)", analytics.to_csv_bytes(res), "similar_tickets.csv", "text/csv", key=f"dl_{hash(tuple(res['ticket_id']))}")

with tab1:
    q = st.text_input("Describe the issue", placeholder="e.g. webhook not delivered 401", key="sim_q")
    allowed = filters("kw")
    k = st.slider("Results", 3, 50, 10, key="kw_k")
    sem = semantic_toggle("kw_sem")
    if q.strip():
        if sem:
            with st.spinner("Loading embedding model and encoding tickets (first run only)…"):
                sidx = ui.memo("semantic_index", lambda: ss_.SemanticIndex(full))
            show(sidx.search(q, k, allowed_ids=allowed))
        else:
            show(index.search(q, k, allowed_ids=allowed))
with tab2:
    tid = st.selectbox("Reference ticket", index.df["ticket_id"].tolist(), key="sim_tid")
    row = index.df[index.df["ticket_id"] == tid].iloc[0]
    st.markdown(f"**{row.get('subject', '')}**  \n{row.get('description', '')}")
    allowed2 = filters("st")
    k2 = st.slider("Results", 3, 50, 10, key="st_k")
    show(index.similar_to(tid, k2, allowed_ids=allowed2))
    st.caption("The reference ticket is always excluded from its own results.")
with tab3:
    ui.note("Cluster numbers are machine-generated groups from K-Means on text. They are NOT verified business categories.", "model")
    a, b = st.columns(2)
    auto = a.checkbox("Choose number of clusters automatically (silhouette)", value=True)
    kk = None if auto else b.slider("Clusters (k)", 2, 15, 6)
    try:
        with st.spinner("Clustering tickets…"):
            cr = ui.memo(f"clusters_{kk}", lambda: clustering.cluster_tickets(full, kk))
    except clustering.ClusteringUnavailable as exc:
        ui.empty_state("Clustering not available", str(exc))
        st.stop()
    ui.kpi_cards([("Clusters", str(cr.k), f"{cr.n_used:,} tickets{' (sampled)' if cr.sampled else ''}", "Number of K-Means groups."),
                  ("Silhouette", "Not available" if cr.silhouette is None else f"{cr.silhouette:.2f}", "−1 … 1, higher = tighter", "Cohesion vs separation of clusters in the reduced text space; text clusters are usually low."),
                  ("Agreement with categories (ARI)", "Not available" if cr.ari_vs_category is None else f"{cr.ari_vs_category:.2f}", "0 = random, 1 = identical", "Adjusted Rand Index between clusters and the labelled category column."),
                  ("2-D variance shown", f"{cr.explained_variance_2d:.0%}", "of reduced space", "The scatter plot is a lossy 2-D projection.")])
    st.plotly_chart(charts.scatter_clusters(cr.points, dk), width="stretch")
    st.dataframe(cr.summary, width="stretch", hide_index=True)
    pick = st.selectbox("Inspect cluster", cr.summary["cluster"].tolist())
    ids = cr.points[cr.points["cluster"] == str(pick)]["ticket_id"].head(10)
    st.dataframe(full[full["ticket_id"].isin(ids)][[c for c in ("ticket_id", "subject", "category", "priority") if c in full]], width="stretch", hide_index=True)
