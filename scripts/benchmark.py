"""Measure real timings on this machine:  python scripts/benchmark.py  (paste results into the README)."""
import resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import clustering, forecasting, similarity_search as sim, sla_risk, ticket_classifier as tc
from src.data_loader import load_sample

def t(label, fn):
    s = time.perf_counter(); out = fn(); print(f"{label:<42}{time.perf_counter() - s:8.2f} s"); return out

raw, df, _ = t("load + validate demo data (%d rows)" % len(pd.read_csv("data/sample_tickets.csv")), load_sample)
res = t("train category classifier (LR + SVM)", lambda: tc.train_classifier(df["text"], df["category"]))
t("train priority classifier", lambda: tc.train_classifier(df["text"], df["priority"], "priority"))
print(f"{'single-ticket prediction latency':<42}{res.latency_ms:8.1f} ms")
t("batch predict 1,000 tickets", lambda: tc.predict(res.pipeline, df["text"].head(1000).tolist()))
bm = t("train SLA breach model", lambda: sla_risk.train_breach_model(df))
t("monitor + score open tickets", lambda: sla_risk.score_open_tickets(bm, sla_risk.monitor(df)))
ix = t("build TF-IDF search index", lambda: sim.TicketIndex(df))
t("similarity query (top-10)", lambda: ix.search("webhook not delivered", 10))
t("cluster tickets (auto k)", lambda: clustering.cluster_tickets(df))
t("forecast (daily, backtested)", lambda: forecasting.forecast_volume(df["created_at"], "D"))
print(f"{'peak RSS memory':<42}{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:8.0f} MB")
