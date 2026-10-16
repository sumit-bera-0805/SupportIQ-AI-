"""SLA monitoring.

Two clearly separated layers:
  1. RULE-BASED deadline monitoring (deterministic; uses dataset deadlines, or - only when
     the dataset has none - a configurable per-priority policy that is flagged as an assumption).
  2. A MACHINE-LEARNING breach model, trained only when the dataset contains real historical
     SLA outcomes (sla_deadline + resolved_at). Features are limited to what is known when a
     ticket is created, so there is no target leakage."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_SLA_HOURS, UNRESOLVED_STATUSES

CAT_FEATURES = ["priority", "category", "product", "channel", "weekday", "hour_bucket"]
MIN_LABELED = 100
MIN_PER_CLASS = 15


class SLAModelUnavailable(ValueError):
    """Raised when there is not enough genuine historical SLA outcome data."""


def as_of_time(df: pd.DataFrame) -> pd.Timestamp | None:
    cols = [c for c in ("created_at", "resolved_at", "first_response_at") if c in df]
    vals = [df[c].max() for c in cols if df[c].notna().any()]
    return max(vals) if vals else None


def is_unresolved(df: pd.DataFrame) -> pd.Series:
    """Unresolved = status in an open state; otherwise (no status column) resolved_at missing.
    Returns a boolean Series; tickets whose state cannot be determined are False."""
    if "status" in df and df["status"].notna().any():
        return df["status"].isin(UNRESOLVED_STATUSES)
    if "resolved_at" in df:
        return df["resolved_at"].isna()
    return pd.Series(False, index=df.index)


def monitor(df: pd.DataFrame, as_of: pd.Timestamp | None = None, sla_hours: dict[str, float] | None = None,
            at_risk_fraction: float = 0.25, allow_policy_deadlines: bool = True) -> pd.DataFrame:
    """Add deadline / remaining-time / state columns using deterministic rules.

    state values: Met, Breached (resolved late), Breached (open, overdue), At risk, On track, Unknown.
    ``deadline_source`` is 'dataset' for real deadlines or 'policy' for assumed ones."""
    out = df.copy()
    as_of = as_of if as_of is not None else as_of_time(out)
    sla_hours = sla_hours or DEFAULT_SLA_HOURS
    n = len(out)
    deadline = out["sla_deadline"] if "sla_deadline" in out else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    source = pd.Series(np.where(deadline.notna(), "dataset", "none"), index=out.index, dtype="object")
    if allow_policy_deadlines and "created_at" in out and "priority" in out:
        hrs = out["priority"].map(sla_hours)
        policy = out["created_at"] + pd.to_timedelta(hrs, unit="h")
        use = deadline.isna() & policy.notna()
        deadline = deadline.where(~use, policy)
        source = source.mask(use, "policy")
    out["deadline"] = deadline
    out["deadline_source"] = source
    resolved_at = out["resolved_at"] if "resolved_at" in out else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    unresolved = is_unresolved(out)
    state = pd.Series("Unknown", index=out.index, dtype="object")
    has_dl = deadline.notna()
    done = resolved_at.notna() & has_dl
    state[done & (resolved_at <= deadline)] = "Met"
    state[done & (resolved_at > deadline)] = "Breached (resolved late)"
    remaining = pd.Series(np.nan, index=out.index)
    if as_of is not None:
        live = has_dl & unresolved & resolved_at.isna()
        remaining = ((deadline - as_of).dt.total_seconds() / 3600).where(live)
        window = (deadline - out["created_at"]).dt.total_seconds() / 3600 if "created_at" in out else pd.Series(np.nan, index=out.index)
        state[live & (remaining < 0)] = "Breached (open, overdue)"
        state[live & (remaining >= 0) & (remaining <= window * at_risk_fraction)] = "At risk"
        state[live & (remaining >= 0) & ~(remaining <= window * at_risk_fraction)] = "On track"
    out["hours_remaining"] = remaining
    if as_of is not None and "created_at" in out:
        out["age_hours"] = ((as_of - out["created_at"]).dt.total_seconds() / 3600).where(unresolved)
    out["sla_state"] = state
    return out


def breach_summary(mon: pd.DataFrame) -> dict:
    """Breach rate among tickets with a known outcome (resolved late or open-and-overdue)."""
    known = mon["sla_state"].isin(["Met", "Breached (resolved late)", "Breached (open, overdue)"])
    breached = mon["sla_state"].str.startswith("Breached")
    k = int(known.sum())
    return {"n_known": k, "n_breached": int((breached & known).sum()),
            "rate": float((breached & known).sum() / k) if k else None,
            "n_at_risk": int((mon["sla_state"] == "At risk").sum()),
            "policy_deadlines": int((mon["deadline_source"] == "policy").sum())}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["text"] = df["text"].fillna("") if "text" in df else ""
    for c in ("priority", "category", "product", "channel"):
        f[c] = df[c].fillna("Unknown").astype(str) if c in df else "Unknown"
    f["weekday"] = df["created_at"].dt.dayofweek.astype("Int64").astype(str)
    f["hour_bucket"] = (df["created_at"].dt.hour // 4).astype("Int64").astype(str)
    return f


@dataclass
class BreachModel:
    pipeline: Pipeline
    metrics: dict
    calibration: pd.DataFrame
    n_train: int
    n_test: int
    prevalence_test: float
    split_date: pd.Timestamp
    notes: list[str] = field(default_factory=list)


def labelled_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a REAL dataset deadline, a resolution time and a creation time."""
    need = {"sla_deadline", "resolved_at", "created_at"}
    if not need.issubset(df.columns):
        raise SLAModelUnavailable("A trained breach model requires sla_deadline, resolved_at and created_at columns with historical outcomes. "
                                  "Rule-based deadline monitoring is still available.")
    d = df.dropna(subset=list(need)).copy()
    d["breached"] = (d["resolved_at"] > d["sla_deadline"]).astype(int)
    return d.sort_values("created_at")


def train_breach_model(df: pd.DataFrame, seed: int = 42) -> BreachModel:
    d = labelled_outcomes(df)
    if len(d) < MIN_LABELED:
        raise SLAModelUnavailable(f"Only {len(d)} resolved ticket(s) with a real SLA deadline; at least {MIN_LABELED} are required.")
    if d["breached"].value_counts().min() < MIN_PER_CLASS:
        raise SLAModelUnavailable("Too few breached (or too few met) historical tickets to learn from.")
    cut = int(len(d) * 0.75)
    train, test = d.iloc[:cut], d.iloc[cut:]  # chronological split: train on the past, test on the future
    if train["breached"].nunique() < 2 or test["breached"].nunique() < 2:
        raise SLAModelUnavailable("The chronological train or test period contains only one outcome class.")
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))])
    pipe.fit(build_features(train), train["breached"])
    proba = pipe.predict_proba(build_features(test))[:, 1]
    y = test["breached"].to_numpy()
    pred = (proba >= 0.5).astype(int)
    metrics = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
               "f1": f1_score(y, pred, zero_division=0), "pr_auc": average_precision_score(y, proba),
               "brier": brier_score_loss(y, proba), "baseline_pr_auc": float(y.mean())}
    cal = pd.DataFrame({"p": proba, "y": y})
    cal["bin"] = pd.qcut(cal["p"], q=min(5, cal["p"].nunique()), duplicates="drop")
    calibration = cal.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    notes = ["Class-weighted logistic regression: raw probabilities are over-estimated for the minority class; "
             "read the calibration table before treating them as true likelihoods."]
    return BreachModel(pipe, {k: float(v) for k, v in metrics.items()}, calibration, len(train), len(test),
                       float(y.mean()), test["created_at"].min(), notes)


def tier(p: float) -> str:
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def score_open_tickets(model: BreachModel, mon: pd.DataFrame, top_factors: int = 3, max_rows: int = 500) -> pd.DataFrame:
    """ML risk for open tickets that have a created_at and are not yet overdue."""
    cand = mon[mon["sla_state"].isin(["On track", "At risk"]) & mon["created_at"].notna()].head(max_rows).copy()
    if cand.empty:
        return cand.assign(breach_probability=[], risk_tier=[], top_factors=[])
    F = build_features(cand)
    cand["breach_probability"] = model.pipeline.predict_proba(F)[:, 1]
    cand["risk_tier"] = cand["breach_probability"].map(tier)
    pre, clf = model.pipeline.named_steps["pre"], model.pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    X = pre.transform(F).tocsr()
    factors = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        c = row.data * clf.coef_[0][row.indices]
        top = np.argsort(-c)[:top_factors]
        factors.append(", ".join(names[row.indices[j]].split("__", 1)[1].replace("_", " = ", 1) for j in top if c[j] > 0))
    cand["top_factors"] = factors
    return cand.sort_values("breach_probability", ascending=False)


def recommended_action(state: str, tier_: str | None = None) -> str:
    """Transparent rule table (not a model output)."""
    if state == "Breached (open, overdue)":
        return "Escalate to team lead now; notify the customer of a revised ETA."
    if state == "At risk" or tier_ == "High":
        return "Assign a senior owner and prioritise today; check for blockers."
    if tier_ == "Medium":
        return "Monitor closely; confirm an owner and next update time."
    return "No action needed; standard queue handling."
