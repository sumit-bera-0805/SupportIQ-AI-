import numpy as np
import pandas as pd
import pytest

from src import explainability, ticket_classifier as tc


def test_insufficient_labels_raise():
    texts = pd.Series([f"ticket {i}" for i in range(30)])
    with pytest.raises(tc.InsufficientDataError, match="at least"):
        tc.train_classifier(texts, pd.Series(["a", "b"] * 15))


def test_single_class_raises():
    texts = pd.Series([f"ticket number {i}" for i in range(100)])
    with pytest.raises(tc.InsufficientDataError, match="class"):
        tc.train_classifier(texts, pd.Series(["a"] * 100))


def test_rare_classes_dropped_and_reported(sample):
    y = sample["category"].copy()
    y.iloc[:3] = "Tiny"
    res = tc.train_classifier(sample["text"], y)
    assert "Tiny" not in res.labels and "Tiny" in res.dropped_classes and any("dropped" in w for w in res.warnings)


@pytest.fixture(scope="module")
def trained(sample):
    return tc.train_classifier(sample["text"], sample["category"])


def test_metrics_are_real_and_consistent(trained):
    m = trained.metrics
    assert 0.5 < m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1
    assert trained.n_train + trained.n_test == trained.n_labeled
    assert trained.confusion.sum() == trained.n_test
    assert set(trained.per_class["class"]) == set(trained.labels)
    assert len(trained.comparison) == 2 and trained.latency_ms > 0


def test_no_train_test_text_leak_in_split(sample):
    # identical (text,label) templates exist in the demo data, so verify the split itself is disjoint by index
    from sklearn.model_selection import train_test_split
    X, y = sample["text"], sample["category"]
    a, b = train_test_split(X.index, test_size=0.2, stratify=y, random_state=42)
    assert set(a).isdisjoint(set(b))


def test_prediction_schema_and_low_confidence(trained):
    out = tc.predict(trained.pipeline, ["Payment failed with valid card", "", "zzz qqq xxx"], low_conf=0.99)
    assert list(out.columns) == ["predicted", "confidence", "low_confidence", "alternatives"]
    assert len(out) == 3 and out.loc[0, "predicted"] == "Billing & Payments"
    assert out["low_confidence"].all()                     # threshold 0.99 flags everything
    assert pd.isna(out.loc[1, "predicted"]) and out.loc[1, "low_confidence"]   # empty text -> no prediction
    assert out.loc[0, "confidence"] == pytest.approx(trained.pipeline.predict_proba(["Payment failed with valid card"]).max())



def test_explanations(trained):
    ex = explainability.explain_prediction(trained.pipeline, "refund for duplicate payment on my invoice")
    assert not ex.empty and (ex["contribution"].diff().dropna() <= 1e-12).all()
    tt = explainability.top_terms_per_class(trained.pipeline, 5)
    assert tt.groupby("class").size().eq(5).all()
