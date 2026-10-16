import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics, charts, forecasting, ui

df = ui.page_context()
ui.page_header("Volume Forecasting", "Baseline forecasts validated with chronological backtests - no black boxes.")
dk = ui.dark()
if "created_at" not in df or df["created_at"].notna().sum() == 0:
    ui.empty_state("Forecasting not available", "This dataset has no valid created_at timestamps.")
    st.stop()

a, b, c = st.columns(3)
freq = a.radio("Granularity", ["D", "W", "M"], horizontal=True, format_func=forecasting.FREQ_LABELS.get, key="fc_freq")
horizon = b.number_input("Horizon (periods)", 1, 60, forecasting.DEFAULT_HORIZON[freq], key=f"fc_h_{freq}")
y = forecasting.aggregate_counts(df["created_at"], freq)
methods = ["Auto (best backtest)"] + list(forecasting.available_methods(freq, len(y)))
method = c.selectbox("Method", methods)
try:
    res = forecasting.forecast_volume(df["created_at"], freq, int(horizon), method)
except forecasting.InsufficientHistory as exc:
    ui.empty_state("Not enough history to forecast", str(exc))
    st.markdown(f"**Minimum history:** {forecasting.MIN_PERIODS[freq]} complete {forecasting.FREQ_LABELS[freq].lower()} periods · **available:** {len(y)}")
    st.stop()

best = res.backtest[res.backtest["method"] == res.method].iloc[0]
ui.kpi_cards([("Method used", res.method, f"{res.n_folds} backtest fold(s)", "Chosen by lowest MAE across chronological rolling-origin backtests (unless you picked a method)."),
              ("Backtest MAE", f"{best['MAE']:.2f}", "tickets / period", "Mean absolute error over held-out future periods."),
              ("Backtest RMSE", f"{best['RMSE']:.2f}", "tickets / period", "Root mean squared error; penalises large misses."),
              ("WAPE", "Not available" if pd.isna(best["WAPE %"]) else f"{best['WAPE %']:.1f}%", "sum|error| / sum actual", "Weighted absolute percentage error; stable when some periods have few tickets."),
              ("Forecast peak", f"{res.forecast['forecast'].max():.1f}", "tickets / period", "Highest forecast value in the horizon.")])
st.plotly_chart(charts.forecast_chart(res.history, res.forecast, f"{forecasting.FREQ_LABELS[freq]} ticket volume: history vs forecast", dk), width="stretch")
st.caption("Interval = forecast ± 1.96 × backtest RMSE: an approximate empirical interval that assumes errors are roughly symmetric and stable.")

l, r = st.columns(2)
with l:
    st.subheader("Backtest comparison")
    st.dataframe(res.backtest.round(2), width="stretch", hide_index=True)
    st.caption("Each method is scored on the same chronological hold-out windows; training data always precedes the evaluated window.")
with r:
    st.subheader("Insights")
    for i in res.insights:
        st.markdown(f"- {i}")
    st.subheader("Capacity planning")
    cap = st.number_input("Tickets one agent can handle per period", 1.0, 500.0, 10.0 if freq == "D" else 50.0)
    peak, upper = res.forecast["forecast"].max(), res.forecast["upper"].max()
    st.markdown(f"Agents needed at forecast peak: **{forecasting.agents_needed(peak, cap)}** · at upper bound: **{forecasting.agents_needed(upper, cap)}**")

st.subheader("Seasonality")
s1, s2 = st.columns(2)
daily = forecasting.aggregate_counts(df["created_at"], "D")
if len(daily) >= 14:
    dow = daily.groupby(daily.index.dayofweek).mean().reindex(range(7)).rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))).reset_index()
    dow.columns = ["weekday", "avg_tickets"]
    fig = px.bar(dow, x="weekday", y="avg_tickets", title="Average tickets by weekday")
    fig.update_traces(marker_color=charts.COLORWAY[1]); fig.update_layout(xaxis_title=None, yaxis_title="Tickets / day")
    s1.plotly_chart(charts.style(fig, dk, 280, False), width="stretch")
monthly = forecasting.aggregate_counts(df["created_at"], "M", drop_partial_last=False)
if len(monthly) >= 2:
    fig = px.bar(monthly.reset_index(), x="index", y="tickets", title="Tickets by month (last month may be partial)")
    fig.update_traces(marker_color=charts.COLORWAY[0]); fig.update_layout(xaxis_title=None, yaxis_title="Tickets")
    s2.plotly_chart(charts.style(fig, dk, 280, False), width="stretch")
out = res.forecast.reset_index().rename(columns={"index": "period_start"})
st.download_button("Download forecast (CSV)", analytics.to_csv_bytes(out), "forecast.csv", "text/csv")
