import numpy as np
import pandas as pd
import pytest

from src import sla_risk as sr

T = pd.Timestamp


def frame():
    return pd.DataFrame({
        "ticket_id": list("ABCDEF"),
        "created_at": [T("2026-01-01 00:00")] * 6,
        "priority": ["High", "High", "High", "High", "Low", "Low"],
        "status": ["Resolved", "Resolved", "Open", "Open", "Open", "Open"],
        "resolved_at": [T("2026-01-01 05:00"), T("2026-01-01 10:00"), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "sla_deadline": [T("2026-01-01 08:00"), T("2026-01-01 08:00"), T("2026-01-01 03:00"), T("2026-01-01 20:00"), T("2026-01-01 11:00"), pd.NaT],
    })


def test_states_and_remaining_hours():
    m = sr.monitor(frame(), as_of=T("2026-01-01 12:00"), allow_policy_deadlines=False).set_index("ticket_id")
    assert m.loc["A", "sla_state"] == "Met"
    assert m.loc["B", "sla_state"] == "Breached (resolved late)"
    assert m.loc["C", "sla_state"] == "Breached (open, overdue)" and m.loc["C", "hours_remaining"] == pytest.approx(-9)
    assert m.loc["D", "sla_state"] == "On track" and m.loc["D", "hours_remaining"] == pytest.approx(8)  # 8h left of 20h window > 25%
    assert m.loc["E", "sla_state"] == "Breached (open, overdue)"
    assert m.loc["F", "sla_state"] == "Unknown" and np.isnan(m.loc["F", "hours_remaining"])  # missing deadline -> unknown, not guessed


def test_at_risk_threshold():
    f = frame()
    m = sr.monitor(f, as_of=T("2026-01-01 17:00"), at_risk_fraction=0.25, allow_policy_deadlines=False).set_index("ticket_id")
    assert m.loc["D", "sla_state"] == "At risk"   # 3h left of 20h window


def test_policy_deadlines_are_flagged():
    m = sr.monitor(frame(), as_of=T("2026-01-01 12:00"), sla_hours={"High": 8, "Low": 72}).set_index("ticket_id")
    assert m.loc["F", "deadline_source"] == "policy" and m.loc["F", "deadline"] == T("2026-01-04 00:00")
    assert m.loc["A", "deadline_source"] == "dataset"
    assert sr.breach_summary(m.reset_index())["policy_deadlines"] == 1


def test_breach_summary_rate():
    m = sr.monitor(frame(), as_of=T("2026-01-01 12:00"), allow_policy_deadlines=False)
    s = sr.breach_summary(m)
    assert s["n_known"] == 4 and s["n_breached"] == 3 and s["rate"] == pytest.approx(0.75)  # A met, B late, C+E overdue; D live; F unknown


def test_model_unavailable_without_outcomes():
    with pytest.raises(sr.SLAModelUnavailable, match="requires"):
        sr.train_breach_model(frame().drop(columns=["sla_deadline"]))
    with pytest.raises(sr.SLAModelUnavailable, match="at least"):
        sr.train_breach_model(frame())


def test_model_is_chronological_and_reports_baseline(sample):
    bm = sr.train_breach_model(sample)
    d = sr.labelled_outcomes(sample)
    assert bm.n_train + bm.n_test == len(d)
    assert d.iloc[: bm.n_train]["created_at"].max() <= bm.split_date            # train strictly precedes test
    assert 0 <= bm.metrics["pr_auc"] <= 1 and bm.metrics["baseline_pr_auc"] == pytest.approx(bm.prevalence_test)


def test_scoring_open_tickets(sample):
    bm = sr.train_breach_model(sample)
    mon = sr.monitor(sample)
    sc = sr.score_open_tickets(bm, mon)
    assert sc["breach_probability"].between(0, 1).all() and set(sc["risk_tier"]) <= {"High", "Medium", "Low"}
    assert sc["sla_state"].isin(["On track", "At risk"]).all()
