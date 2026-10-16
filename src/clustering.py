"""K-Means clustering of ticket text. Cluster ids are machine-generated groups, NOT verified categories."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer

from .config import CLUSTER_MAX_ROWS

MIN_ROWS = 50


class ClusteringUnavailable(ValueError):
    pass


@dataclass
class ClusterResult:
    points: pd.DataFrame          # ticket_id, cluster, x, y, subject, category
    summary: pd.DataFrame         # cluster, size, keywords, example
    k: int
    silhouette: float | None
    ari_vs_category: float | None
    n_used: int
    sampled: bool
    explained_variance_2d: float


def cluster_tickets(df: pd.DataFrame, k: int | None = None, max_rows: int = CLUSTER_MAX_ROWS, seed: int = 42) -> ClusterResult:
    d = df[df["text"].fillna("").str.len() > 0]
    if len(d) < MIN_ROWS:
        raise ClusteringUnavailable(f"Clustering needs at least {MIN_ROWS} tickets with text; found {len(d)}.")
    sampled = len(d) > max_rows
    if sampled:
        d = d.sample(max_rows, random_state=seed)
    d = d.reset_index(drop=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.6, sublinear_tf=True, max_features=20000, stop_words="english")
    try:
        X = vec.fit_transform(d["text"])
    except ValueError as exc:
        raise ClusteringUnavailable(f"Text is too sparse to cluster: {exc}") from exc
    n_comp = max(2, min(50, X.shape[1] - 1, len(d) - 1))
    Z = Normalizer().fit_transform(TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X))
    rng = np.random.default_rng(seed)
    if k is None:
        cands = range(3, min(10, max(3, len(d) // 25)) + 1)
        sub = rng.choice(len(d), size=min(len(d), 2000), replace=False)
        best, best_s = 3, -1.0
        for c in cands:
            labels = KMeans(c, n_init=3, random_state=seed).fit_predict(Z[sub])
            s = silhouette_score(Z[sub], labels) if len(set(labels)) > 1 else -1
            if s > best_s:
                best, best_s = c, s
        k = best
    km = KMeans(k, n_init=5, random_state=seed).fit(Z)
    labels = km.labels_
    sil = float(silhouette_score(Z, labels, sample_size=min(len(d), 2000), random_state=seed)) if len(set(labels)) > 1 else None
    names = np.asarray(vec.get_feature_names_out())
    rows = []
    for c in range(k):
        m = labels == c
        mean = np.asarray(X[m].mean(axis=0)).ravel()
        kw = ", ".join(names[np.argsort(-mean)[:6]])
        centre = km.cluster_centers_[c]
        ex = d.loc[m, "subject"].iloc[int(np.argmax(Z[m] @ centre))] if "subject" in d else d.loc[m, "text"].iloc[0]
        rows.append({"cluster": c, "size": int(m.sum()), "keywords": kw, "example": ex})
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    xy = pca.transform(Z)
    pts = pd.DataFrame({"ticket_id": d["ticket_id"], "cluster": labels.astype(str), "x": xy[:, 0], "y": xy[:, 1],
                        "subject": d["subject"] if "subject" in d else d["text"].str[:80],
                        "category": d["category"] if "category" in d else np.nan})
    ari = None
    if "category" in d and d["category"].notna().sum() > 10:
        lab = d["category"].notna()
        ari = float(adjusted_rand_score(d.loc[lab, "category"], labels[lab.to_numpy()]))
    return ClusterResult(pts, pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True), k, sil, ari,
                         len(d), sampled, float(pca.explained_variance_ratio_.sum()))
