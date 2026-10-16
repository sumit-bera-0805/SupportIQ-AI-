"""Column detection, mapping, validation and cleaning of uploaded ticket data.

The original upload is never modified; ``validate_and_clean`` works on a copy."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CANONICAL_COLUMNS, DATE_COLUMNS
from .preprocessing import build_text_series

SYNONYMS: dict[str, list[str]] = {
    "ticket_id": ["ticket_id", "id", "ticket", "ticket_number", "ticket_no", "case_id", "case_number", "incident_id", "issue_id", "number"],
    "subject": ["subject", "title", "summary", "ticket_subject", "short_description", "issue"],
    "description": ["description", "body", "details", "text", "message", "ticket_description", "content", "long_description"],
    "category": ["category", "type", "ticket_type", "issue_type", "topic", "label", "class", "department"],
    "priority": ["priority", "ticket_priority", "severity", "urgency"],
    "status": ["status", "ticket_status", "state"],
    "created_at": ["created_at", "created", "created_date", "creation_date", "opened_at", "open_date", "date", "timestamp", "submitted_at", "created_on"],
    "resolved_at": ["resolved_at", "resolved", "resolution_date", "closed_at", "closed_date", "resolved_date", "solved_at"],
    "first_response_at": ["first_response_at", "first_response", "first_reply_at", "first_response_date", "first_responded_at"],
    "sla_deadline": ["sla_deadline", "sla_due", "due_date", "due_at", "sla_due_date", "deadline", "sla_due_at"],
    "product": ["product", "product_name", "service", "application", "app", "module"],
    "channel": ["channel", "source", "contact_channel", "medium", "origin"],
    "customer_id": ["customer_id", "customer", "customer_name", "user_id", "requester", "requester_id", "account_id", "client_id"],
}

PRIORITY_MAP = {
    "p0": "Critical", "p1": "Critical", "critical": "Critical", "urgent": "Critical", "blocker": "Critical", "highest": "Critical", "sev1": "Critical",
    "p2": "High", "high": "High", "major": "High", "sev2": "High",
    "p3": "Medium", "medium": "Medium", "med": "Medium", "normal": "Medium", "moderate": "Medium", "sev3": "Medium",
    "p4": "Low", "low": "Low", "minor": "Low", "trivial": "Low", "lowest": "Low", "sev4": "Low",
}
STATUS_MAP = {
    "open": "Open", "new": "Open", "opened": "Open", "reopened": "Open", "unresolved": "Open",
    "in progress": "In Progress", "in_progress": "In Progress", "inprogress": "In Progress", "working": "In Progress", "assigned": "In Progress",
    "pending": "Pending", "on hold": "Pending", "on_hold": "Pending", "waiting": "Pending", "awaiting customer": "Pending",
    "resolved": "Resolved", "solved": "Resolved", "done": "Resolved", "fixed": "Resolved",
    "closed": "Closed", "complete": "Closed", "completed": "Closed",
}
_NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "nat", "<na>"}


class ValidationError(ValueError):
    """Raised when the upload cannot be used at all (e.g. no text columns)."""


@dataclass
class ValidationResult:
    df: pd.DataFrame
    issues: list[dict] = field(default_factory=list)  # {"level": error|warning|info, "message": str}
    stats: dict = field(default_factory=dict)

    def add(self, level: str, message: str) -> None:
        self.issues.append({"level": level, "message": message})


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_columns(columns) -> dict[str, str | None]:
    """Map canonical field -> source column using exact synonym matches only."""
    normalised = {_norm(c): c for c in columns}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for canon in CANONICAL_COLUMNS:
        found = None
        for syn in SYNONYMS[canon]:
            src = normalised.get(syn)
            if src is not None and src not in used:
                found = src
                break
        mapping[canon] = found
        if found:
            used.add(found)
    return mapping


def _blank_to_na(s: pd.Series) -> pd.Series:
    def fix(v):
        if isinstance(v, str):
            v = v.strip()
            return np.nan if v.lower() in _NULL_TOKENS else v
        return np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return s.astype("object").map(fix)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse to timezone-naive UTC. Unparseable values become NaT (never guessed)."""
    parsed = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None)


def normalise_priority(v: object) -> object:
    if not isinstance(v, str):
        return np.nan
    return PRIORITY_MAP.get(v.strip().lower(), v.strip().title())


def normalise_status(v: object) -> object:
    if not isinstance(v, str):
        return np.nan
    return STATUS_MAP.get(v.strip().lower(), v.strip().title())


def validate_and_clean(raw: pd.DataFrame, mapping: dict[str, str | None],
                       missing_policy: str = "keep", malformed_rows_skipped: int = 0) -> ValidationResult:
    """Validate ``raw`` using ``mapping`` and return a cleaned working copy.

    missing_policy:
      keep            - keep rows with missing optional values (default, nothing invented)
      label_unknown   - fill missing priority/status/product/channel with 'Unknown'
                        (never category, never timestamps)
      drop_incomplete - drop rows missing category, priority, or created_at (when mapped)
    """
    if missing_policy not in {"keep", "label_unknown", "drop_incomplete"}:
        raise ValueError(f"Unknown missing_policy: {missing_policy}")
    res = ValidationResult(df=pd.DataFrame())
    mapped = {k: v for k, v in mapping.items() if v and v in raw.columns}
    if "subject" not in mapped and "description" not in mapped:
        raise ValidationError("Required column missing: map at least one of 'subject' or 'description'. "
                              "Text is required for classification and search.")

    work = pd.DataFrame({canon: _blank_to_na(raw[src]) for canon, src in mapped.items()})
    rows_in = len(work)
    res.stats.update(rows_in=rows_in, malformed_rows_skipped=malformed_rows_skipped, mapped_columns=mapped)
    if malformed_rows_skipped:
        res.add("warning", f"{malformed_rows_skipped} malformed row(s) (wrong number of fields) were skipped while reading the CSV.")
    if rows_in == 0:
        raise ValidationError("The file contains no data rows.")

    text_cols = [c for c in ("subject", "description") if c in work]
    empty_text = work[text_cols].isna().all(axis=1)
    res.stats["empty_text_removed"] = int(empty_text.sum())
    if empty_text.any():
        res.add("warning", f"{int(empty_text.sum())} row(s) had neither subject nor description and were removed from the working copy.")
        work = work[~empty_text].copy()

    if "ticket_id" not in work:
        work.insert(0, "ticket_id", [f"ROW-{i + 1}" for i in range(len(work))])
        res.add("warning", "No ticket ID column mapped: row-based IDs (ROW-n) were generated. They are not real ticket IDs.")
        res.stats["duplicate_ids_removed"] = 0
    else:
        missing_id = work["ticket_id"].isna()
        if missing_id.any():
            work.loc[missing_id, "ticket_id"] = [f"ROW-{i + 1}" for i in np.flatnonzero(missing_id.to_numpy())]
            res.add("warning", f"{int(missing_id.sum())} missing ticket ID(s) were replaced with row-based IDs.")
        work["ticket_id"] = work["ticket_id"].astype(str)
        dup = work["ticket_id"].duplicated(keep="first")
        res.stats["duplicate_ids_removed"] = int(dup.sum())
        if dup.any():
            res.add("warning", f"{int(dup.sum())} duplicate ticket ID(s) found; the first occurrence of each was kept.")
            work = work[~dup].copy()

    invalid_dates: dict[str, int] = {}
    for col in DATE_COLUMNS:
        if col in work:
            was_missing = work[col].isna()
            parsed = _parse_dates(work[col])
            bad = int((parsed.isna() & ~was_missing).sum())
            invalid_dates[col] = bad
            if bad:
                res.add("warning", f"{bad} invalid value(s) in '{col}' could not be parsed and were set to missing (not guessed).")
            work[col] = parsed
    res.stats["invalid_dates"] = invalid_dates
    if "created_at" not in work:
        res.add("info", "No created_at column mapped: time-based analytics (trends, forecasting, SLA) will be unavailable.")

    if "priority" in work:
        work["priority"] = work["priority"].map(normalise_priority)
    if "status" in work:
        work["status"] = work["status"].map(normalise_status)

    def _hours(start: str, end: str, name: str) -> None:
        if start in work and end in work:
            delta = (work[end] - work[start]).dt.total_seconds() / 3600
            neg = delta < 0
            if neg.any():
                res.add("warning", f"{int(neg.sum())} ticket(s) have {end} before {start}; {name} left missing for them.")
                delta = delta.mask(neg)
            work[name] = delta
    _hours("created_at", "first_response_at", "first_response_hours")
    _hours("created_at", "resolved_at", "resolution_hours")

    if missing_policy == "label_unknown":
        for col in ("priority", "status", "product", "channel"):
            if col in work:
                work[col] = work[col].fillna("Unknown")
    elif missing_policy == "drop_incomplete":
        need = [c for c in ("category", "priority", "created_at") if c in work]
        incomplete = work[need].isna().any(axis=1) if need else pd.Series(False, index=work.index)
        res.stats["incomplete_removed"] = int(incomplete.sum())
        if incomplete.any():
            res.add("info", f"{int(incomplete.sum())} row(s) with missing {', '.join(need)} were dropped per the chosen policy.")
            work = work[~incomplete].copy()

    if len(work) == 0:
        raise ValidationError("No usable rows remain after validation.")
    work["text"] = build_text_series(work)
    work = work.reset_index(drop=True)
    res.stats["rows_out"] = len(work)
    res.stats["missing_counts"] = {c: int(work[c].isna().sum()) for c in work.columns if c != "text"}
    if "category" in work:
        labelled = int(work["category"].notna().sum())
        res.stats["labelled_category"] = labelled
        if labelled < 60:
            res.add("warning", f"Only {labelled} ticket(s) have a category label; model training needs at least 60.")
    else:
        res.add("info", "No category column mapped: the classifier cannot be trained (search, clustering and rule-based SLA monitoring still work).")
    res.df = work
    res.add("info", f"{len(work):,} of {rows_in:,} row(s) kept in the validated working copy.")
    return res
