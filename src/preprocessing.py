"""Text preparation shared by every NLP module."""
from __future__ import annotations

import re

import pandas as pd

_WS = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Return a whitespace-normalised string; missing values become ''."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return _WS.sub(" ", str(value)).strip()


def build_text(subject: object, description: object) -> str:
    """Combine subject and description. The subject is repeated once so that it
    carries slightly more weight in TF-IDF than the (usually longer) body."""
    s, d = clean_text(subject), clean_text(description)
    if s and d:
        return f"{s}. {s}. {d}"
    return s or d


def build_text_series(df: pd.DataFrame) -> pd.Series:
    subj = df["subject"] if "subject" in df else pd.Series("", index=df.index)
    desc = df["description"] if "description" in df else pd.Series("", index=df.index)
    return pd.Series([build_text(s, d) for s, d in zip(subj, desc)], index=df.index, dtype="object")
