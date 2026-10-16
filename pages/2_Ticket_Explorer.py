import streamlit as st

from src import analytics, ui

df = ui.page_context()
ui.page_header("Ticket Explorer", "Search, sort, page through and export the filtered tickets.")
if df.empty:
    ui.empty_state("No tickets match the current filters", "Widen or reset the filters in the sidebar.")
    st.stop()

q = st.text_input("Search subject / description", placeholder="e.g. refund, timeout, SSO", key="ex_q")
view = df
if q.strip():
    needle = q.strip().lower()
    view = df[df["text"].fillna("").str.lower().str.contains(needle, regex=False)]
show_cols = [c for c in df.columns if c not in ("text",)]
a, b, c = st.columns([2, 1, 1])
sort_col = a.selectbox("Sort by", show_cols, index=show_cols.index("created_at") if "created_at" in show_cols else 0)
asc = b.toggle("Ascending", value=False)
page_size = c.selectbox("Rows per page", [25, 50, 100, 250], index=1)
view = view.sort_values(sort_col, ascending=asc, na_position="last")
pages_n = max(1, -(-len(view) // page_size))
page_no = st.number_input(f"Page (1–{pages_n})", min_value=1, max_value=pages_n, value=1, step=1)
st.caption(f"{len(view):,} matching tickets · showing {min(page_size, len(view) - (page_no - 1) * page_size) if len(view) else 0}")
st.dataframe(view.iloc[(page_no - 1) * page_size: page_no * page_size][show_cols], width="stretch", hide_index=True)
st.download_button("Download filtered tickets (CSV)", analytics.to_csv_bytes(view[show_cols]), "filtered_tickets.csv", "text/csv", disabled=view.empty)

if not view.empty:
    st.subheader("Ticket detail")
    tid = st.selectbox("Ticket", view["ticket_id"].head(2000).tolist())
    row = view[view["ticket_id"] == tid].iloc[0]
    l, r = st.columns([1, 2])
    with l:
        for col in show_cols:
            if col not in ("subject", "description"):
                st.markdown(f"**{col}:** {row[col] if row[col] == row[col] else '—'}")
    with r:
        st.markdown(f"**{row.get('subject', '')}**")
        st.write(row.get("description", ""))
