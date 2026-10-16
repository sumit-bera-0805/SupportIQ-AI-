"""Reading CSV uploads safely and loading the bundled demo dataset."""
from __future__ import annotations

import csv
import hashlib
import io

import pandas as pd

from .config import MAX_ROWS, MAX_UPLOAD_MB, SAMPLE_PATH
from .data_validation import ValidationError, detect_columns, validate_and_clean


def read_csv_bytes(data: bytes, max_mb: int = MAX_UPLOAD_MB, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, int]:
    """Parse CSV bytes as strings. Returns (raw dataframe, malformed rows skipped).

    Raises ValidationError for oversized, empty, or unreadable files."""
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValidationError(f"File is {len(data) / 1024 / 1024:.1f} MB; the limit for this deployment is {max_mb} MB.")
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise ValidationError(f"Could not parse the file as CSV: {exc}") from exc
    rows = [r for r in rows if any(c.strip() for c in r)]  # ignore blank lines
    if not rows:
        raise ValidationError("The uploaded file is empty.")
    header, body = [h.strip() for h in rows[0]], rows[1:]
    seen: dict[str, int] = {}
    for i, h in enumerate(header):  # make duplicate / blank header names unique so mapping stays unambiguous
        h = h or f"unnamed_{i}"
        seen[h] = seen.get(h, 0) + 1
        header[i] = h if seen[h] == 1 else f"{h}_{seen[h]}"
    width = len(header)
    good, skipped = [], 0
    for r in body:
        if len(r) > width:          # more fields than headers: cannot be mapped reliably
            skipped += 1
        else:
            good.append(r + [""] * (width - len(r)))   # short rows = trailing empty values
    df = pd.DataFrame(good, columns=header, dtype="object")
    if df.empty or len(df.columns) < 2:
        raise ValidationError("The file has no data rows or fewer than two columns. Is it a comma-separated CSV with a header row?")
    if len(df) > max_rows:
        raise ValidationError(f"File has {len(df):,} rows; the limit for this deployment is {max_rows:,}.")
    return df, skipped


def dataframe_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (raw, validated working copy, stats) for the bundled demo dataset."""
    raw = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)
    result = validate_and_clean(raw, detect_columns(raw.columns))
    return raw, result.df, result.stats
