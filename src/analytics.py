"""Filtering, KPI computation and aggregation. Every metric returns None when the data
cannot support it (the UI shows 'Not available'); nothing is invented."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRIORITY_ORDER, UNRESOLVED_STATUSES
from .sla_risk import as_of_time, breach_summary, is_unresolved, monitor

KPI_DEFINITIONS = {
    "Total tickets": "Count of tickets in the validated working copy after the active filters are applied.",
    "Open / unresolved": "Tickets whose status is Open, In Progress or Pending. If there is no status column: tickets with no resolved_at timestamp.",
    "High-priority share": "Share of tickets with a known priority that are High or Critical. Tickets with missing priority are excluded from the denominator.",
    "Avg first response": "Mean hours between created_at and first_response_at, only for tickets where both are valid and ordered.",
    "SLA breach rate": "Breached / (met + breached) among tickets with a dataset SLA deadline. Breached = resolved after the deadline, or still open past it (as of the latest timestamp in the data).",
    "Classifier macro-F1": "Macro-averaged F1 on a stratified 20% hold-out test set; the model never saw these tickets during training.",
    "Median resolution": "Median hours from created_at to resolved_at for resolved tickets with valid timestamps.",
    "Volume vs prior period": "Tickets in the last 30 days of data vs the 30 days before. Shown only when the data covers 60+ days.",
}


def filter_options(df: pd.DataFrame) -> dict:
    opts = {}
    for c in ("category", "priority", "status", "product", "channel"):
        if c in df and df[c].notna().any():
            vals = sorted(df[c].dropna().unique().tolist())
            if c == "priority":
                vals = [p for p in PRIORITY_ORDER if p in vals] + [v for v in vals if v not in PRIORITY_ORDER]
            opts[c] = vals
    return opts


def apply_filters(df: pd.DataFrame, date_range=None, **selected) -> pd.DataFrame:
    """selected: column -> list of allowed values (empty/None = no filter)."""
    mask = pd.Series(True, index=df.index)
    if date_range and "created_at" in df and len(date_range) == 2 and all(date_range):
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        mask &= (df["created_at"] >= start) & (df["created_at"] < end)
    for col, vals in selected.items():
        if vals and col in df:
            mask &= df[col].isin(vals)
    return df[mask]


def kpis(df: pd.DataFrame) -> dict:
    """Return raw KPI values (None = not available)."""
    n = len(df)
    out: dict = {"total": n}
    out["open"] = int(is_unresolved(df).sum()) if ("status" in df or "resolved_at" in df) and n else None
    if "priority" in df and df["priority"].notna().any():
        known = df["priority"].dropna()
        out["high_priority_pct"] = float(known.isin(["High", "Critical"]).mean() * 100)
    else:
        out["high_priority_pct"] = None
    out["avg_first_response_h"] = float(df["first_response_hours"].mean()) if "first_response_hours" in df and df["first_response_hours"].notna().any() else None
    out["median_resolution_h"] = float(df["resolution_hours"].median()) if "resolution_hours" in df and df["resolution_hours"].notna().any() else None
    if "sla_deadline" in df and df["sla_deadline"].notna().any():
        s = breach_summary(monitor(df, as_of_time(df), allow_policy_deadlines=False))
        out["sla_breach_rate"] = s["rate"] * 100 if s["rate"] is not None else None
        out["sla_known"] = s["n_known"]
    else:
        out["sla_breach_rate"] = None
        out["sla_known"] = 0
    out["volume_change_pct"] = volume_change(df)
    return out


def volume_change(df: pd.DataFrame, days: int = 30) -> float | None:
    if "created_at" not in df or df["created_at"].notna().sum() == 0:
        return None
    end = df["created_at"].max()
    if (end - df["created_at"].min()).days < 2 * days:
        return None
    cur = df["created_at"] > end - pd.Timedelta(days=days)
    prev = (df["created_at"] > end - pd.Timedelta(days=2 * days)) & ~cur
    if prev.sum() == 0:
        return None
    return float((cur.sum() - prev.sum()) / prev.sum() * 100)


def volume_trend(df: pd.DataFrame, freq: str = "D", by: str | None = None) -> pd.DataFrame:
    if "created_at" not in df:
        return pd.DataFrame()
    d = df.dropna(subset=["created_at"])
    rule = {"D": "D", "W": "W-SUN", "M": "MS"}[freq]
    if by and by in d:
        g = d.groupby([pd.Grouper(key="created_at", freq=rule), by]).size().rename("tickets").reset_index()
    else:
        g = d.groupby(pd.Grouper(key="created_at", freq=rule)).size().rename("tickets").reset_index()
    return g


def category_priority_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    if "category" not in df or "priority" not in df:
        return None
    d = df.dropna(subset=["category", "priority"])
    if d.empty:
        return None
    m = pd.crosstab(d["category"], d["priority"])
    cols = [p for p in PRIORITY_ORDER if p in m.columns] + [c for c in m.columns if c not in PRIORITY_ORDER]
    return m[cols]


def count_by(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    if col not in df or df[col].notna().sum() == 0:
        return None
    return df[col].value_counts().rename_axis(col).reset_index(name="tickets")


def customer_summary(df: pd.DataFrame, top: int = 15) -> pd.DataFrame | None:
    if "customer_id" not in df or df["customer_id"].notna().sum() == 0:
        return None
    g = df.dropna(subset=["customer_id"]).groupby("customer_id")
    out = g.size().rename("tickets").to_frame()
    if "priority" in df:
        out["high_priority"] = g["priority"].apply(lambda s: int(s.isin(["High", "Critical"]).sum()))
    if "resolution_hours" in df:
        out["median_resolution_h"] = g["resolution_hours"].median()
    return out.sort_values("tickets", ascending=False).head(top).reset_index()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV export (UTF-8, ISO timestamps, no index). Guards against spreadsheet formula injection."""
    safe = df.copy()
    for c in safe.select_dtypes(include=["object", "string"]).columns:
        safe[c] = safe[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return safe.to_csv(index=False).encode("utf-8")


def format_hours(h: float | None) -> str:
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return "Not available"
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} d"
