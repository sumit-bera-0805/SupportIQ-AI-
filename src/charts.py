"""Plotly figure builders with one consistent, theme-aware style."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORWAY = ["#3b82f6", "#22d3ee", "#a78bfa", "#60a5fa", "#34d399", "#f59e0b", "#f472b6", "#94a3b8"]
PRIORITY_COLORS = {"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#3b82f6", "Low": "#64748b"}


def style(fig: go.Figure, dark: bool = True, height: int = 340, legend: bool = True) -> go.Figure:
    txt = "#e6edf7" if dark else "#0f172a"
    grid = "rgba(148,163,184,.18)"
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=txt, size=12), colorway=COLORWAY, height=height, margin=dict(l=10, r=10, t=36, b=10),
                      legend=dict(orientation="h", y=-0.2, title=None) if legend else dict(), showlegend=legend,
                      title=dict(font=dict(size=14), x=0.0), hoverlabel=dict(font_size=12))
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    return fig


def bar_h(df: pd.DataFrame, cat: str, val: str, title: str, dark: bool) -> go.Figure:
    d = df.sort_values(val)
    fig = px.bar(d, x=val, y=cat, orientation="h", title=title, text=val)
    fig.update_traces(marker_color=COLORWAY[0], textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, xaxis_title="Tickets")
    return style(fig, dark, legend=False)


def priority_bar(df: pd.DataFrame, dark: bool) -> go.Figure:
    fig = px.bar(df, x="priority", y="tickets", color="priority", color_discrete_map=PRIORITY_COLORS, title="Priority distribution", text="tickets",
                 category_orders={"priority": ["Critical", "High", "Medium", "Low"]})
    fig.update_layout(xaxis_title=None, yaxis_title="Tickets")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return style(fig, dark, legend=False)


def trend_line(df: pd.DataFrame, title: str, dark: bool, by: str | None = None) -> go.Figure:
    fig = px.line(df, x="created_at", y="tickets", color=by, title=title, markers=len(df) < 60)
    fig.update_layout(xaxis_title=None, yaxis_title="Tickets", hovermode="x unified")
    return style(fig, dark, legend=bool(by))


def stacked_bar(df: pd.DataFrame, x: str, color: str, title: str, dark: bool) -> go.Figure:
    g = df.dropna(subset=[x, color]).groupby([x, color]).size().rename("tickets").reset_index()
    fig = px.bar(g, x=x, y="tickets", color=color, title=title, barmode="stack")
    fig.update_layout(xaxis_title=None, yaxis_title="Tickets")
    return style(fig, dark)


def heatmap(m: pd.DataFrame, title: str, dark: bool, fmt: str = ".0f", colorbar: str = "Tickets") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=m.values, x=[str(c) for c in m.columns], y=[str(i) for i in m.index], colorscale="Blues", text=m.values,
                               texttemplate="%{text:" + fmt + "}", colorbar=dict(title=colorbar), hovertemplate="%{y} / %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(title=title)
    return style(fig, dark, legend=False, height=max(300, 34 * len(m) + 120))


def histogram(values: pd.Series, title: str, xlabel: str, dark: bool, nbins: int = 40, clip_q: float = 0.98) -> go.Figure:
    v = values.dropna()
    if len(v):
        v = v[v <= v.quantile(clip_q)]
    fig = px.histogram(v, nbins=nbins, title=title)
    fig.update_traces(marker_color=COLORWAY[1])
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Tickets", bargap=0.04)
    return style(fig, dark, legend=False)


def confusion(cm: np.ndarray, labels: list[str], dark: bool) -> go.Figure:
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, text=cm, texttemplate="%{text}",
                               colorbar=dict(title="Row share"), hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{text}<extra></extra>"))
    fig.update_layout(title="Confusion matrix (test set; colour = share of true class)", xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed")
    return style(fig, dark, legend=False, height=max(380, 52 * len(labels) + 140))


def forecast_chart(history: pd.Series, fc: pd.DataFrame, title: str, dark: bool, show_last: int = 120) -> go.Figure:
    h = history.tail(show_last)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Actual", mode="lines", line=dict(color=COLORWAY[0], width=2)))
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["upper"]) + list(fc["lower"][::-1]), fill="toself",
                             fillcolor="rgba(167,139,250,.20)", line=dict(width=0), name="95% interval (approx.)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast", mode="lines+markers", line=dict(color=COLORWAY[2], width=2, dash="dash")))
    cut = fc.index[0]
    fig.add_shape(type="line", x0=cut, x1=cut, y0=0, y1=1, yref="paper", line=dict(color="#94a3b8", width=1, dash="dot"))
    fig.add_annotation(x=cut, y=1, yref="paper", text="forecast starts", showarrow=False, xanchor="left", font=dict(size=11, color="#94a3b8"))
    fig.update_layout(title=title, yaxis_title="Tickets", xaxis_title=None, hovermode="x unified")
    return style(fig, dark)


def scatter_clusters(pts: pd.DataFrame, dark: bool) -> go.Figure:
    fig = px.scatter(pts, x="x", y="y", color="cluster", hover_data={"ticket_id": True, "subject": True, "category": True, "x": False, "y": False},
                     title="Ticket clusters (PCA projection of TF-IDF/SVD space)", opacity=0.75)
    fig.update_layout(xaxis_title="PC 1", yaxis_title="PC 2")
    return style(fig, dark)
