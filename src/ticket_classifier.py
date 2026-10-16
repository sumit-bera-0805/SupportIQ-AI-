"""TF-IDF (word + char n-gram) ticket classification with honest evaluation.

Primary model: Logistic Regression (interpretable, gives class probabilities).
Comparison model: Linear SVM. The primary model is fixed in advance (not chosen on the
test set) so the reported test metrics are not optimistically biased."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

MIN_LABELED = 60
MIN_PER_CLASS = 5
TEST_SIZE = 0.2
SEED = 42


class InsufficientDataError(ValueError):
    """Raised when labels are too few / too homogeneous for a trustworthy model."""


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=30000, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=40000, strip_accents="unicode")),
    ])


@dataclass
class TrainResult:
    task: str
    pipeline: Pipeline
    labels: list[str]
    n_labeled: int
    n_train: int
    n_test: int
    dropped_classes: dict[str, int]
    metrics: dict[str, float]
    per_class: pd.DataFrame
    confusion: np.ndarray
    report_text: str
    test_frame: pd.DataFrame          # text, true, predicted, confidence
    comparison: pd.DataFrame
    train_seconds: float
    latency_ms: float
    ece: float
    warnings: list[str] = field(default_factory=list)


def _expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def prepare_labeled(texts: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Drop rows with missing text/label and classes with too few examples."""
    d = pd.DataFrame({"t": texts.astype("object"), "y": labels.astype("object")})
    d = d[d["t"].notna() & (d["t"].astype(str).str.len() > 0) & d["y"].notna()]
    counts = d["y"].value_counts()
    rare = counts[counts < MIN_PER_CLASS]
    d = d[~d["y"].isin(rare.index)]
    return d["t"].astype(str), d["y"].astype(str), {str(k): int(v) for k, v in rare.items()}


def train_classifier(texts: pd.Series, labels: pd.Series, task: str = "category",
                     seed: int = SEED, test_size: float = TEST_SIZE) -> TrainResult:
    """Train + evaluate on a stratified hold-out split. Raises InsufficientDataError
    instead of reporting meaningless metrics."""
    X, y, dropped = prepare_labeled(texts, labels)
    classes = sorted(y.unique())
    if len(X) < MIN_LABELED:
        raise InsufficientDataError(f"Only {len(X)} labelled ticket(s) usable for '{task}'; at least {MIN_LABELED} are required.")
    if len(classes) < 2:
        raise InsufficientDataError(f"Only {len(classes)} class with enough examples (>= {MIN_PER_CLASS} each); at least 2 are required.")
    n_test = int(round(len(X) * test_size))
    if n_test < len(classes):
        raise InsufficientDataError("Too few labelled tickets to hold out a stratified test set for this many classes.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    warns: list[str] = []
    if dropped:
        warns.append("Classes dropped (fewer than %d examples): %s." % (MIN_PER_CLASS, ", ".join(f"{k} ({v})" for k, v in dropped.items())))
    share = y.value_counts(normalize=True)
    if share.min() < 0.05 or share.max() / share.min() > 10:
        warns.append(f"Imbalanced labels (largest/smallest class ratio {share.max() / share.min():.1f}x). Class weights are applied; prefer macro-F1 over accuracy.")
    if len(X) < 300:
        warns.append(f"Small evaluation set ({len(X_te)} test tickets): metrics have wide uncertainty.")

    feats = make_features()
    t0 = time.perf_counter()
    try:
        Xtr = feats.fit_transform(X_tr)  # vectoriser fitted on TRAIN only -> no leakage
    except ValueError as exc:
        raise InsufficientDataError(f"Text is too sparse to build features: {exc}") from exc
    Xte = feats.transform(X_te)
    lr = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")
    lr.fit(Xtr, y_tr)
    lr_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    svm = LinearSVC(C=0.5, class_weight="balanced")
    svm.fit(Xtr, y_tr)
    svm_s = time.perf_counter() - t1

    pred = lr.predict(Xte)
    proba = lr.predict_proba(Xte)
    conf = proba.max(axis=1)
    svm_pred = svm.predict(Xte)
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
    w_p, w_r, w_f, _ = precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)
    m_p, m_r, m_f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    metrics = dict(accuracy=accuracy_score(y_te, pred), macro_f1=m_f, weighted_f1=w_f,
                   macro_precision=m_p, macro_recall=m_r, weighted_precision=w_p, weighted_recall=w_r)
    comparison = pd.DataFrame([
        {"model": "Logistic Regression (primary)", "accuracy": accuracy_score(y_te, pred), "macro_f1": m_f,
         "weighted_f1": w_f, "train_seconds": lr_s, "probabilities": "yes"},
        {"model": "Linear SVM", "accuracy": accuracy_score(y_te, svm_pred), "macro_f1": f1_score(y_te, svm_pred, average="macro", zero_division=0),
         "weighted_f1": f1_score(y_te, svm_pred, average="weighted", zero_division=0), "train_seconds": svm_s, "probabilities": "no (decision scores only)"},
    ])
    pipe = Pipeline([("features", feats), ("clf", lr)])
    sample = list(X_te.iloc[: min(30, len(X_te))])
    t2 = time.perf_counter()
    for doc in sample:
        pipe.predict_proba([doc])
    latency_ms = (time.perf_counter() - t2) / max(len(sample), 1) * 1000

    test_frame = pd.DataFrame({"text": X_te.values, "true": y_te.values, "predicted": pred, "confidence": conf})
    return TrainResult(
        task=task, pipeline=pipe, labels=classes, n_labeled=len(X), n_train=len(X_tr), n_test=len(X_te),
        dropped_classes=dropped, metrics={k: float(v) for k, v in metrics.items()}, per_class=per_class,
        confusion=confusion_matrix(y_te, pred, labels=classes),
        report_text=classification_report(y_te, pred, labels=classes, zero_division=0),
        test_frame=test_frame, comparison=comparison, train_seconds=lr_s + svm_s, latency_ms=float(latency_ms),
        ece=_expected_calibration_error(conf, (pred == y_te.values).astype(float)), warnings=warns)


def predict(pipeline: Pipeline, texts, low_conf: float = 0.45, top_k: int = 3) -> pd.DataFrame:
    """Batch prediction. ``confidence`` is the model's softmax probability for the top
    class (a score, not a guarantee; see ECE on the evaluation page)."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    cols = ["predicted", "confidence", "low_confidence", "alternatives"]
    if not texts:
        return pd.DataFrame(columns=cols)
    proba = pipeline.predict_proba(texts)
    classes = np.asarray(pipeline.classes_)
    order = np.argsort(-proba, axis=1)
    top = order[:, 0]
    alts = []
    for i in range(len(texts)):
        alts.append("; ".join(f"{classes[j]} ({proba[i, j]:.2f})" for j in order[i, 1:top_k]))
    conf = proba[np.arange(len(texts)), top]
    empty = np.array([not t.strip() for t in texts])
    out = pd.DataFrame({"predicted": classes[top], "confidence": conf, "low_confidence": (conf < low_conf) | empty, "alternatives": alts})
    out.loc[empty, ["predicted", "confidence", "alternatives"]] = [np.nan, np.nan, ""]
    return out


def class_scores(pipeline: Pipeline, text: str) -> pd.DataFrame:
    proba = pipeline.predict_proba([text])[0]
    return pd.DataFrame({"class": pipeline.classes_, "score": proba}).sort_values("score", ascending=False).reset_index(drop=True)
