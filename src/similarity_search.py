"""TF-IDF cosine retrieval (sparse, top-k only - never an all-pairs matrix) with an
optional, lazily-loaded Sentence-Transformers semantic mode."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import SIMILARITY_MAX_ROWS

RESULT_COLUMNS = ["ticket_id", "score", "matching_terms", "category", "priority", "product", "created_at", "subject"]


class TicketIndex:
    def __init__(self, df: pd.DataFrame, max_rows: int = SIMILARITY_MAX_ROWS):
        d = df[df["text"].fillna("").str.len() > 0]
        self.truncated = len(d) > max_rows
        if self.truncated:
            d = d.sort_values("created_at").tail(max_rows) if "created_at" in d else d.tail(max_rows)
        self.df = d.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=50000, stop_words="english", strip_accents="unicode")
        self.X = self.vec.fit_transform(self.df["text"]).tocsr()
        self.terms = np.asarray(self.vec.get_feature_names_out())
        self._pos = {t: i for i, t in enumerate(self.df["ticket_id"])}

    def _rank(self, scores: np.ndarray, qvec, top_k: int, exclude_id: str | None, allowed: set | None, min_score: float) -> pd.DataFrame:
        scores = scores.copy()
        if exclude_id is not None and exclude_id in self._pos:
            scores[self._pos[exclude_id]] = -1.0     # a ticket is never its own match
        if allowed is not None:
            scores[~self.df["ticket_id"].isin(allowed).to_numpy()] = -1.0
        k = min(top_k, len(scores))
        if k == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        top = [i for i in top if scores[i] > min_score]
        rows = self.df.iloc[top].copy()
        rows["score"] = scores[top]
        q_idx = qvec.indices
        q_w = dict(zip(q_idx, qvec.data))
        match = []
        for i in top:
            d = self.X.getrow(i)
            common = [(j, q_w[j] * w) for j, w in zip(d.indices, d.data) if j in q_w]
            common.sort(key=lambda t: -t[1])
            match.append(", ".join(self.terms[j] for j, _ in common[:5]))
        rows["matching_terms"] = match
        for c in RESULT_COLUMNS:
            if c not in rows:
                rows[c] = np.nan
        return rows[RESULT_COLUMNS].reset_index(drop=True)

    def search(self, query: str, top_k: int = 10, exclude_id: str | None = None, allowed_ids: set | None = None, min_score: float = 0.0) -> pd.DataFrame:
        if not query or not query.strip():
            return pd.DataFrame(columns=RESULT_COLUMNS)
        q = self.vec.transform([query]).tocsr()
        if q.nnz == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        scores = (self.X @ q.T).toarray().ravel()
        return self._rank(scores, q.getrow(0), top_k, exclude_id, allowed_ids, min_score)

    def similar_to(self, ticket_id: str, top_k: int = 10, allowed_ids: set | None = None, min_score: float = 0.0) -> pd.DataFrame:
        if ticket_id not in self._pos:
            raise KeyError(f"Ticket {ticket_id} is not in the search index")
        q = self.X.getrow(self._pos[ticket_id])
        scores = (self.X @ q.T).toarray().ravel()
        return self._rank(scores, q, top_k, ticket_id, allowed_ids, min_score)


def semantic_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class SemanticIndex:
    """Optional embeddings search. Model is imported/loaded only when instantiated."""
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, df: pd.DataFrame, max_rows: int = 5000):
        if not semantic_available():
            raise RuntimeError("sentence-transformers is not installed; using TF-IDF instead.")
        from sentence_transformers import SentenceTransformer  # lazy import
        d = df[df["text"].fillna("").str.len() > 0].tail(max_rows).reset_index(drop=True)
        self.df = d
        self.model = SentenceTransformer(self.MODEL)
        self.emb = self.model.encode(d["text"].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def search(self, query: str, top_k: int = 10, exclude_id: str | None = None, allowed_ids: set | None = None) -> pd.DataFrame:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        if exclude_id is not None:
            scores[(self.df["ticket_id"] == exclude_id).to_numpy()] = -1
        if allowed_ids is not None:
            scores[~self.df["ticket_id"].isin(allowed_ids).to_numpy()] = -1
        top = np.argsort(-scores)[:top_k]
        rows = self.df.iloc[top].copy()
        rows["score"] = scores[top]
        rows["matching_terms"] = "(semantic match)"
        for c in RESULT_COLUMNS:
            if c not in rows:
                rows[c] = np.nan
        return rows[RESULT_COLUMNS].reset_index(drop=True)
