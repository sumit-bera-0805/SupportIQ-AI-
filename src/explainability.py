"""Native explainability from linear-model coefficients (no SHAP dependency needed):
contribution of a term to a prediction = TF-IDF value x class coefficient."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _word_names(pipeline) -> tuple[np.ndarray, np.ndarray]:
    names = pipeline.named_steps["features"].get_feature_names_out()
    word_mask = np.array([n.startswith("word__") for n in names])
    return np.array([n.replace("word__", "", 1) for n in names]), word_mask


def _class_coef(clf, class_idx: int) -> np.ndarray:
    coef = clf.coef_
    if coef.shape[0] == 1:  # binary problem: single row describes classes_[1]
        return coef[0] if class_idx == 1 else -coef[0]
    return coef[class_idx]


def top_terms_per_class(pipeline, n: int = 10) -> pd.DataFrame:
    """Highest-weight word n-grams per class (global importance)."""
    clf = pipeline.named_steps["clf"]
    names, mask = _word_names(pipeline)
    rows = []
    for i, cls in enumerate(clf.classes_):
        w = _class_coef(clf, i)
        idx = np.flatnonzero(mask)
        top = idx[np.argsort(-w[idx])[:n]]
        rows += [{"class": cls, "term": names[j], "weight": float(w[j])} for j in top]
    return pd.DataFrame(rows)


def explain_prediction(pipeline, text: str, class_name: str | None = None, n: int = 8) -> pd.DataFrame:
    """Terms that pushed the model toward ``class_name`` (default: predicted class)."""
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    if class_name is None:
        class_name = classes[int(np.argmax(pipeline.predict_proba([text])[0]))]
    ci = classes.index(class_name)
    X = pipeline.named_steps["features"].transform([text]).tocsr()
    names, mask = _word_names(pipeline)
    w = _class_coef(clf, ci)
    idx = X.indices
    contrib = X.data * w[idx]
    keep = mask[idx]
    df = pd.DataFrame({"term": names[idx][keep], "contribution": contrib[keep]})
    return df.sort_values("contribution", ascending=False).head(n).reset_index(drop=True)
