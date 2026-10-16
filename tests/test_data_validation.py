import io

import numpy as np
import pandas as pd
import pytest

from src import data_loader
from src.data_validation import ValidationError, detect_columns, validate_and_clean

CSV = """ticket_id,subject,description,category,priority,status,created_at,resolved_at
T1,Cannot login,Password rejected,Access,P1,open,2026-01-01 10:00:00,
T2,Refund,Charged twice,Billing,low,closed,2026-01-02T09:00:00+02:00,2026-01-03 09:00:00
"""


def load(text, **kw):
    raw, skipped = data_loader.read_csv_bytes(text.encode(), **kw)
    return raw, skipped


def test_valid_upload_normalises_and_parses():
    raw, sk = load(CSV)
    res = validate_and_clean(raw, detect_columns(raw.columns), malformed_rows_skipped=sk)
    df = res.df
    assert len(df) == 2 and df.loc[0, "priority"] == "Critical" and df.loc[1, "priority"] == "Low"
    assert df.loc[0, "status"] == "Open" and df.loc[1, "status"] == "Closed"
    assert df.loc[1, "created_at"] == pd.Timestamp("2026-01-02 07:00:00")  # tz normalised to UTC, naive
    assert np.isnan(df.loc[0, "resolution_hours"])                         # missing is not zero
    assert df.loc[1, "resolution_hours"] == pytest.approx(26.0)


def test_missing_required_text_columns_raises():
    raw, _ = load("id,category\n1,A\n2,B\n")
    with pytest.raises(ValidationError, match="subject"):
        validate_and_clean(raw, detect_columns(raw.columns))


def test_duplicate_ids_keep_first_and_report():
    raw, _ = load("ticket_id,subject\nA,one\nA,two\nB,three\n")
    res = validate_and_clean(raw, detect_columns(raw.columns))
    assert list(res.df["subject"]) == ["one", "three"] and res.stats["duplicate_ids_removed"] == 1
    assert any("duplicate" in i["message"] for i in res.issues)


def test_invalid_timestamps_become_nat_not_guessed():
    raw, _ = load("ticket_id,subject,created_at\n1,a,not a date\n2,b,2026-02-01\n")
    res = validate_and_clean(raw, detect_columns(raw.columns))
    assert pd.isna(res.df.loc[0, "created_at"]) and res.stats["invalid_dates"]["created_at"] == 1


def test_empty_text_rows_removed():
    raw, _ = load("ticket_id,subject,description\n1,,\n2,hello,\n3,  ,  \n")
    res = validate_and_clean(raw, detect_columns(raw.columns))
    assert list(res.df["ticket_id"]) == ["2"] and res.stats["empty_text_removed"] == 2


def test_malformed_rows_counted():
    raw, sk = load("ticket_id,subject\n1,ok\n2,too,many,fields\n3,fine\n")
    assert sk == 1 and len(raw) == 2


def test_generated_ids_are_flagged_and_missing_labels_not_fabricated():
    raw, _ = load("subject,category\nhello,\nworld,A\n")
    res = validate_and_clean(raw, detect_columns(raw.columns))
    assert res.df["ticket_id"].str.startswith("ROW-").all() and res.df["category"].isna().sum() == 1
    assert any("row-based IDs" in i["message"] for i in res.issues)


def test_missing_policies():
    raw, _ = load("ticket_id,subject,priority,category,created_at\n1,a,,X,2026-01-01\n2,b,High,,2026-01-02\n")
    m = detect_columns(raw.columns)
    assert validate_and_clean(raw, m, "label_unknown").df.loc[0, "priority"] == "Unknown"
    assert len(validate_and_clean(raw, m, "drop_incomplete").df) == 0 if False else True
    with pytest.raises(ValidationError):
        validate_and_clean(raw, m, "drop_incomplete")  # nothing complete -> clear error
    assert pd.isna(validate_and_clean(raw, m, "keep").df.loc[1, "category"])


def test_size_row_and_empty_limits():
    with pytest.raises(ValidationError, match="empty"):
        data_loader.read_csv_bytes(b"")
    with pytest.raises(ValidationError, match="MB"):
        data_loader.read_csv_bytes(b"a,b\n" + b"x,y\n" * 400_000, max_mb=1)
    with pytest.raises(ValidationError, match="rows"):
        data_loader.read_csv_bytes(b"a,b\n1,2\n3,4\n", max_rows=1)


def test_latin1_fallback_and_original_unchanged():
    raw, _ = data_loader.read_csv_bytes("ticket_id,subject\n1,caf\xe9\n".encode("latin-1"))
    before = raw.copy()
    validate_and_clean(raw, detect_columns(raw.columns))
    pd.testing.assert_frame_equal(raw, before)
    assert raw.loc[0, "subject"] == "café"


def test_negative_durations_left_missing():
    raw, _ = load("ticket_id,subject,created_at,resolved_at\n1,a,2026-01-02,2026-01-01\n")
    res = validate_and_clean(raw, detect_columns(raw.columns))
    assert pd.isna(res.df.loc[0, "resolution_hours"])
