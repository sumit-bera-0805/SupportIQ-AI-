import plotly.express as px
import streamlit as st

from src import analytics, charts, ui

df = ui.page_context()
ui.page_header("Customer & Product Insights", "Who contacts support, about what, and where service is slowest.")
dk = ui.dark()
if df.empty:
    ui.empty_state("No tickets match the current filters", "Widen or reset the filters in the sidebar.")
    st.stop()

st.subheader("Customers")
cs = analytics.customer_summary(df)
if cs is None:
    ui.empty_state("Customer insights not available", "Map a customer_id column to enable them.")
else:
    l, r = st.columns(2)
    with l:
        fig = px.bar(cs.sort_values("tickets"), x="tickets", y="customer_id", orientation="h", title="Top customers by ticket volume")
        fig.update_traces(marker_color=charts.COLORWAY[0]); fig.update_layout(yaxis_title=None, xaxis_title="Tickets")
        st.plotly_chart(charts.style(fig, dk, 420, False), width="stretch")
    with r:
        per = df.dropna(subset=["customer_id"]).groupby("customer_id").size()
        ui.kpi_cards([("Customers", f"{per.size:,}", "with ≥1 ticket", "Distinct customer_id values."),
                      ("Repeat contacts", f"{(per >= 2).mean() * 100:.0f}%", "customers with 2+ tickets", "Share of customers who raised more than one ticket in this filter."),
                      ("Top-10 share", f"{per.sort_values(ascending=False).head(10).sum() / per.sum() * 100:.0f}%", "of all tickets", "Tickets from the ten busiest customers.")])
        st.dataframe(cs, width="stretch", hide_index=True)

st.subheader("Products")
if "product" in df and df["product"].notna().any():
    a, b = st.columns(2)
    if "category" in df and df["category"].notna().any():
        m = df.dropna(subset=["product", "category"]).pipe(lambda d: __import__("pandas").crosstab(d["product"], d["category"]))
        a.plotly_chart(charts.heatmap(m, "Product × category", dk), width="stretch")
    if "resolution_hours" in df and df["resolution_hours"].notna().any():
        g = df.groupby("product")["resolution_hours"].median().reset_index()
        fig = px.bar(g.sort_values("resolution_hours"), x="resolution_hours", y="product", orientation="h", title="Median resolution time by product (hours)")
        fig.update_traces(marker_color=charts.COLORWAY[1]); fig.update_layout(yaxis_title=None, xaxis_title="Hours")
        b.plotly_chart(charts.style(fig, dk, 340, False), width="stretch")
else:
    ui.empty_state("Product insights not available", "Map a product column to enable them.")

st.subheader("Channels")
if "channel" in df and df["channel"].notna().any():
    cols = [c for c in ("first_response_hours", "resolution_hours") if c in df and df[c].notna().any()]
    if cols:
        g = df.groupby("channel")[cols].median().reset_index().melt("channel", var_name="metric", value_name="median_hours")
        fig = px.bar(g, x="channel", y="median_hours", color="metric", barmode="group", title="Median response / resolution hours by channel")
        fig.update_layout(xaxis_title=None, yaxis_title="Hours")
        st.plotly_chart(charts.style(fig, dk), width="stretch")
    else:
        ui.empty_state("Channel timing not available", "Needs first_response_at / resolved_at.")
else:
    ui.empty_state("Channel insights not available", "Map a channel column to enable them.")
