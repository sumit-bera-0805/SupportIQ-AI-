"""Ticket-volume forecasting with transparent baselines and chronological backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FREQ_LABELS = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
MIN_PERIODS = {"D": 42, "W": 16, "M": 12}     # minimum history to forecast at all
SEASON = {"D": 7, "W": 0, "M": 12}            # 0 = no seasonal method for this granularity
MA_WINDOW = {"D": 7, "W": 4, "M": 3}
DEFAULT_HORIZON = {"D": 14, "W": 8, "M": 3}
_PANDAS_FREQ = {"D": "D", "W": "W-SUN", "M": "MS"}


class InsufficientHistory(ValueError):
    pass


def aggregate_counts(created: pd.Series, freq: str = "D", drop_partial_last: bool = True) -> pd.Series:
    """Ticket counts per period. Periods with no tickets between first and last are real zeros.
    The final period is dropped for weekly/monthly data when it is not yet complete."""
    s = created.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    pf = _PANDAS_FREQ[freq]
    if freq == "D":
        idx = s.dt.floor("D")
        full = pd.date_range(idx.min(), idx.max(), freq="D")
        counts = idx.value_counts().reindex(full, fill_value=0).sort_index()
    elif freq == "W":
        per = s.dt.to_period("W-SUN")
        counts = per.value_counts().sort_index()
        full = pd.period_range(counts.index.min(), counts.index.max(), freq="W-SUN")
        counts = counts.reindex(full, fill_value=0)
        counts.index = counts.index.start_time
        if drop_partial_last and len(counts) > 1 and s.max() < (counts.index[-1] + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)).normalize():
            counts = counts.iloc[:-1]
    else:
        per = s.dt.to_period("M")
        counts = per.value_counts().sort_index()
        full = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
        counts = counts.reindex(full, fill_value=0)
        counts.index = counts.index.start_time
        if drop_partial_last and len(counts) > 1 and s.max().to_period("M").end_time.normalize() > s.max().normalize():
            counts = counts.iloc[:-1]
    counts.name = "tickets"
    return counts.astype(float)


def _ma(y: np.ndarray, h: int, freq: str) -> np.ndarray:
    return np.full(h, y[-MA_WINDOW[freq]:].mean())


def _snaive(y: np.ndarray, h: int, freq: str) -> np.ndarray:
    s = SEASON[freq]
    return np.resize(y[-s:], h)


def _savg(y: np.ndarray, h: int, freq: str) -> np.ndarray:
    """Seasonal average: mean of each weekday over the last 4 weeks (daily only)."""
    s = SEASON[freq]
    last = y[-s * 4:].reshape(4, s).mean(axis=0)
    return np.resize(last, h)


def available_methods(freq: str, n: int) -> dict:
    m = {"Moving average": _ma}
    s = SEASON[freq]
    if s and n >= 2 * s:
        m["Seasonal naive"] = _snaive
        if freq == "D" and n >= 4 * s:
            m["Seasonal average (4 wks)"] = _savg
    return m


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    nz = actual > 0
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE %": float(np.mean(np.abs(err[nz] / actual[nz])) * 100) if nz.any() else np.nan,
            "WAPE %": float(np.abs(err).sum() / actual.sum() * 100) if actual.sum() > 0 else np.nan}


@dataclass
class ForecastResult:
    freq: str
    history: pd.Series
    forecast: pd.DataFrame      # index=period start; forecast, lower, upper
    method: str
    backtest: pd.DataFrame      # one row per method
    n_folds: int
    insights: list[str]
    residual_rmse: float


def backtest(y: pd.Series, h: int, freq: str, folds: int = 3) -> tuple[pd.DataFrame, dict, int]:
    vals = y.to_numpy(dtype=float)
    methods = available_methods(freq, len(vals))
    min_train = max(MIN_PERIODS[freq] // 2, 2 * SEASON[freq] if SEASON[freq] else 4)
    usable = [f for f in range(folds) if len(vals) - h * (folds - f) >= min_train]
    if not usable:
        raise InsufficientHistory("Not enough history to backtest this horizon. Choose a shorter horizon or coarser granularity.")
    rows, errors = [], {}
    for name, fn in methods.items():
        acts, preds = [], []
        for f in usable:
            origin = len(vals) - h * (folds - f)
            acts.append(vals[origin:origin + h]); preds.append(fn(vals[:origin], h, freq))
        a, p = np.concatenate(acts), np.concatenate(preds)
        rows.append({"method": name, **_metrics(a, p)})
        errors[name] = a - p
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), errors, len(usable)


def forecast_volume(created: pd.Series, freq: str = "D", horizon: int | None = None, method: str = "Auto (best backtest)") -> ForecastResult:
    y = aggregate_counts(created, freq)
    n = len(y)
    if n < MIN_PERIODS[freq]:
        raise InsufficientHistory(f"{FREQ_LABELS[freq]} forecasting needs at least {MIN_PERIODS[freq]} complete periods of history; this data has {n}. "
                                  "Upload a longer history or choose a coarser granularity.")
    h = horizon or DEFAULT_HORIZON[freq]
    bt, errors, folds = backtest(y, h, freq)
    chosen = bt.iloc[0]["method"] if method.startswith("Auto") or method not in errors else method
    fn = available_methods(freq, n)[chosen]
    pred = np.maximum(fn(y.to_numpy(), h, freq), 0)
    rmse = float(np.sqrt(np.mean(errors[chosen] ** 2)))
    step = pd.DateOffset(months=1) if freq == "M" else pd.Timedelta(days=7 if freq == "W" else 1)
    idx = [y.index[-1] + step * (i + 1) for i in range(h)]
    fc = pd.DataFrame({"forecast": pred, "lower": np.maximum(pred - 1.96 * rmse, 0), "upper": pred + 1.96 * rmse}, index=pd.DatetimeIndex(idx))
    insights = _insights(y, fc, freq)
    return ForecastResult(freq, y, fc, chosen, bt, folds, insights, rmse)


def _insights(y: pd.Series, fc: pd.DataFrame, freq: str) -> list[str]:
    out = []
    top = y.sort_values(ascending=False).head(3)
    out.append("Busiest historical periods: " + ", ".join(f"{d:%d %b %Y} ({int(v)})" for d, v in top.items()) + ".")
    if freq == "D" and len(y) >= 14:
        by_dow = y.groupby(y.index.dayofweek).mean()
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        out.append(f"Highest-volume weekday on average: {names[int(by_dow.idxmax())]} ({by_dow.max():.1f} tickets/day); lowest: {names[int(by_dow.idxmin())]} ({by_dow.min():.1f}).")
    pk = fc["forecast"].idxmax()
    out.append(f"Forecast peak: {fc['forecast'].max():.1f} tickets in the period starting {pk:%d %b %Y} (95% interval up to {fc.loc[pk, 'upper']:.0f}).")
    recent = y.tail(MA_WINDOW[freq]).mean()
    out.append(f"Forecast average {fc['forecast'].mean():.1f} per period vs {recent:.1f} over the last {MA_WINDOW[freq]} observed periods.")
    return out


def agents_needed(peak_tickets: float, tickets_per_agent: float) -> int:
    """Capacity helper: ceil(peak volume / per-agent throughput)."""
    if tickets_per_agent <= 0:
        raise ValueError("tickets_per_agent must be positive")
    return int(np.ceil(peak_tickets / tickets_per_agent))
