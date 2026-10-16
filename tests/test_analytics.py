import numpy as np
import pandas as pd
import pytest

from src import analytics, clustering, forecasting, rules, storage
from src import similarity_search as sim


def test_filters_combine(sample):
    out = analytics.apply_filters(sample, (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31")), category=["Bug / Defect"], priority=["High", "Critical"])
    assert len(out) and out["category"].eq("Bug / Defect").all() and out["priority"].isin(["High", "Critical"]).all()
    assert out["created_at"].between("2026-01-01", "2026-02-01").all()
    assert len(analytics.apply_filters(sample, None, category=[])) == len(sample)  # empty selection = no filter


def test_kpis_values_and_not_available():
    df = pd.DataFrame({"ticket_id": ["1", "2", "3"], "text": ["a", "b", "c"]})
    k = analytics.kpis(df)
    assert k["total"] == 3 and k["high_priority_pct"] is None and k["sla_breach_rate"] is None and k["avg_first_response_h"] is None and k["volume_change_pct"] is None
    df2 = df.assign(priority=["High", "Low", None], status=["Open", "Closed", "Pending"])
    k2 = analytics.kpis(df2)
    assert k2["high_priority_pct"] == pytest.approx(50.0) and k2["open"] == 2   # missing priority excluded from denominator


def test_kpis_on_sample_match_manual(sample):
    k = analytics.kpis(sample)
    assert k["total"] == len(sample) and k["open"] == int(sample["status"].isin(["Open", "In Progress", "Pending"]).sum())
    assert k["avg_first_response_h"] == pytest.approx(sample["first_response_hours"].mean())


def test_matrix_and_trend(sample):
    m = analytics.category_priority_matrix(sample)
    assert m.values.sum() == sample[["category", "priority"]].dropna().shape[0]
    assert analytics.volume_trend(sample, "M")["tickets"].sum() == sample["created_at"].notna().sum()


def test_csv_export_roundtrip_and_formula_guard():
    df = pd.DataFrame({"a": ["=cmd()", "ok"], "b": [1, 2]})
    out = pd.read_csv(__import__("io").BytesIO(analytics.to_csv_bytes(df)))
    assert out.loc[0, "a"].startswith("'=") and out.loc[1, "a"] == "ok" and list(out.columns) == ["a", "b"]


def test_forecast_insufficient_history():
    created = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D"))
    with pytest.raises(forecasting.InsufficientHistory, match="at least"):
        forecasting.forecast_volume(created, "D")
    with pytest.raises(forecasting.InsufficientHistory):
        forecasting.forecast_volume(pd.Series(dtype="datetime64[ns]"), "D")


def test_forecast_valid_and_backtest_is_chronological(sample):
    r = forecasting.forecast_volume(sample["created_at"], "D", 14)
    assert len(r.forecast) == 14 and (r.forecast["lower"] <= r.forecast["forecast"]).all() and (r.forecast["forecast"] <= r.forecast["upper"]).all()
    assert r.forecast.index[0] == r.history.index[-1] + pd.Timedelta(days=1)
    assert r.backtest["MAE"].is_monotonic_increasing and r.method == r.backtest.iloc[0]["method"]


def test_seasonal_average_exact():
    y = np.tile([10, 10, 10, 10, 10, 0, 0], 8).astype(float)
    assert forecasting._savg(y, 7, "D").tolist() == [10, 10, 10, 10, 10, 0, 0]
    assert forecasting.agents_needed(25, 10) == 3
    with pytest.raises(ValueError):
        forecasting.agents_needed(5, 0)


def test_zero_count_days_are_real_zeros():
    s = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-04"]))
    c = forecasting.aggregate_counts(s, "D")
    assert c.tolist() == [2, 0, 0, 1]


def test_similarity_excludes_self_and_ranks(sample):
    ix = sim.TicketIndex(sample)
    tid = sample["ticket_id"].iloc[50]
    res = ix.similar_to(tid, 10)
    assert tid not in set(res["ticket_id"]) and len(res) == 10 and res["score"].is_monotonic_decreasing
    q = ix.search("webhook not delivered", 5)
    assert q["matching_terms"].str.contains("webhook").all() and list(q.columns) == sim.RESULT_COLUMNS
    assert ix.search("   ").empty and ix.search("zzzzqqqq").empty
    only = ix.search("payment refund", 20, allowed_ids=set(sample.loc[sample["category"] == "Billing & Payments", "ticket_id"]))
    assert only["category"].eq("Billing & Payments").all()
    with pytest.raises(KeyError):
        ix.similar_to("nope")


def test_clustering_requires_enough_rows_and_runs(sample):
    with pytest.raises(clustering.ClusteringUnavailable):
        clustering.cluster_tickets(sample.head(10))
    r = clustering.cluster_tickets(sample, k=5)
    assert r.k == 5 and r.summary["size"].sum() == len(sample) and len(r.points) == len(sample)


def test_rules_are_deterministic():
    s = rules.urgency_signals("Production is blocked, service is down for all users")
    assert s["rule_priority"] == "Critical"
    assert rules.urgency_signals("Request: dark mode")["rule_priority"] is None
    p, why = rules.combine_priority("Low", "Critical")
    assert p == "Critical" and "keyword rule" in why


def test_storage_migrations_and_no_op_when_unwritable(tmp_path):
    db = tmp_path / "t.db"
    assert storage.init_db(db) and storage.schema_version(db) == len(storage.MIGRATIONS)
    assert storage.init_db(db) and storage.schema_version(db) == len(storage.MIGRATIONS)   # idempotent
    storage.log_model_run("d", 10, "category", "m", {"macro_f1": 0.9}, db)
    assert storage.recent_model_runs(5, db)[0]["metrics"]["macro_f1"] == 0.9
    assert storage.init_db("/proc/nonexistent/x.db") is False
