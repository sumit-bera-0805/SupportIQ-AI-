"""Generate the bundled DEMO dataset (synthetic - not real customer data).

Deterministic (seed 42) so the demo is reproducible:  python scripts/generate_sample_data.py
Resolution times are simulated from priority, category, channel and weekday so that SLA
breach prediction has genuine (synthetic) signal. Results on this data demonstrate the
pipeline only; they say nothing about performance on real tickets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.csv"
START, END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30 23:59")

PRODUCTS = ["Cloud Storage", "Analytics Suite", "Mobile App", "Payments Hub", "Developer API"]
CHANNELS = ["Email", "Chat", "Web Form", "Phone", "Social"]
SLA_H = {"Critical": 4, "High": 8, "Medium": 24, "Low": 72}

T = {
 "Billing & Payments": (
  ["Charged twice for {p} subscription", "Invoice amount does not match my plan", "Refund not received for cancelled {p} order", "Payment failed with valid card", "Need a copy of last month's invoice", "Unexpected charge on my statement"],
  ["I was billed {n} dollars twice this month and need a refund.", "The invoice shows a higher amount than my contracted plan. Please review the billing.", "My credit card was declined at checkout although the card is valid and has funds.", "Please send a receipt and tax invoice for the last quarter.", "I cancelled but the subscription was renewed and I was charged again."]),
 "Login & Access": (
  ["Cannot log in to {p}", "Password reset email never arrives", "Account locked after failed attempts", "Two-factor code not accepted", "Single sign-on loop on {p}", "Need access permissions for new team member"],
  ["I keep getting an invalid credentials error even after resetting my password.", "The verification code never arrives so I cannot sign in to my account.", "My account was locked and I need it unlocked to continue working.", "SSO redirects me back to the login page repeatedly.", "Please grant admin access to the new analyst on our team."]),
 "Bug / Defect": (
  ["{p} crashes when saving settings", "Export button does nothing in {p}", "Wrong value displayed on summary screen", "Error message appears after update", "Button misaligned and unclickable on {p}", "Data not saving after edit"],
  ["After the latest update the application crashes whenever I click save.", "Clicking export produces no file and no error message.", "The summary page shows a wrong number compared to the detail view.", "I get an unexpected error code after updating to the new version.", "Changes I make are lost after refreshing the page. Steps to reproduce attached."]),
 "Performance & Outage": (
  ["{p} is down for all users", "Very slow response times on {p}", "Service unavailable 503 errors", "Timeouts across the whole region", "Dashboard takes minutes to load", "Intermittent outage affecting production"],
  ["Our entire team cannot reach the service and production is blocked. This is an outage.", "Pages take over a minute to load since this morning and requests time out.", "We see 503 errors on every request, customers are affected.", "The system is extremely slow and keeps timing out during peak hours.", "Latency has increased dramatically and several users report the service is down."]),
 "Feature Request": (
  ["Request: dark mode for {p}", "Please add bulk edit option", "Feature suggestion: scheduled reports", "Would like Slack notifications", "Add support for custom fields", "Request for CSV import in {p}"],
  ["It would be great if you could add a dark mode option to reduce eye strain.", "We often edit many items at once and a bulk edit feature would save hours.", "Can you add scheduled email reports? This enhancement would help our managers.", "Our team would love native notification integration. Please consider it for the roadmap.", "We need custom fields on records. Is this planned for a future release?"]),
 "Integration & API": (
  ["API returns 401 unauthorized", "Webhook not firing for {p}", "Rate limit errors on API calls", "OAuth token refresh failing", "SDK throws exception on {p} endpoint", "Third-party integration stopped syncing"],
  ["Our API key is valid but every call returns 401 unauthorized since yesterday.", "Webhooks are not delivered to our endpoint and the integration stopped syncing.", "We hit the rate limit with only a few requests per minute. Please check our quota.", "The OAuth refresh token request fails and our integration cannot authenticate.", "The client SDK raises an exception when calling the endpoint with pagination."]),
 "Data & Reporting": (
  ["Report numbers do not match export", "Missing records in {p} dashboard", "Need help building a custom report", "Duplicate rows in exported data", "Chart shows wrong date range", "Data sync delayed in {p}"],
  ["The totals in the report differ from the exported spreadsheet for the same period.", "Some records from last week are missing from the dashboard.", "How can I build a report that groups results by region and month?", "Exported file contains duplicate rows which breaks our analysis.", "The chart ignores the date filter and displays the wrong period."]),
}
W = {"Billing & Payments": .17, "Login & Access": .19, "Bug / Defect": .18, "Performance & Outage": .09, "Feature Request": .10, "Integration & API": .14, "Data & Reporting": .13}
FILLER = ["Thanks in advance.", "Please advise.", "This is affecting our work.", "Let me know if you need more details.", "Happy to jump on a call.", "", "", ""]
CRIT_WORDS = ["urgent", "asap", "production is blocked", "all users affected", "critical"]
CAT_RESOLVE = {"Performance & Outage": 1.5, "Integration & API": 1.35, "Bug / Defect": 1.2, "Data & Reporting": 1.1, "Billing & Payments": 0.85, "Login & Access": 0.75, "Feature Request": 1.0}
CH_RESOLVE = {"Phone": 0.8, "Chat": 0.85, "Email": 1.05, "Web Form": 1.1, "Social": 1.2}
CH_FRT = {"Chat": 0.15, "Phone": 0.1, "Email": 2.0, "Web Form": 2.5, "Social": 3.0}


def main() -> None:
    days = pd.date_range(START.normalize(), END.normalize(), freq="D")
    base = 4.0 + 1.6 * np.linspace(0, 1, len(days))
    wk = np.array([1.35, 1.3, 1.2, 1.15, 1.0, 0.45, 0.35])[days.dayofweek]
    lam = base * wk
    for spike_start in (pd.Timestamp("2026-02-10"), pd.Timestamp("2026-06-22")):  # two incident spikes
        lam = lam + np.where((days >= spike_start) & (days < spike_start + pd.Timedelta(days=3)), 6, 0)
    counts = rng.poisson(lam)
    cats = list(W)
    customers = [f"CUST-{i:04d}" for i in range(1, 281)]
    cust_p = rng.dirichlet(np.full(len(customers), 0.35))
    rows = []
    for day, c in zip(days, counts):
        for _ in range(c):
            cat = rng.choice(cats, p=np.array(list(W.values())) / sum(W.values()))
            if day in pd.date_range("2026-02-10", periods=3) and rng.random() < .6:
                cat = "Performance & Outage"
            subj_t, desc_t = T[cat]
            prod = rng.choice(PRODUCTS)
            subject = rng.choice(subj_t).format(p=prod, n=int(rng.integers(20, 400)))
            desc = rng.choice(desc_t).format(p=prod, n=int(rng.integers(20, 400)))
            urgent = rng.random() < (0.35 if cat == "Performance & Outage" else 0.05)
            if urgent:
                desc += " " + rng.choice(CRIT_WORDS).capitalize() + "."
            desc = (desc + " " + rng.choice(FILLER)).strip()
            r = rng.random()
            if r < 0.14:  # ambiguous: body written about a different topic than the subject
                other = rng.choice([c for c in cats if c != cat]); desc = rng.choice(T[other][1]).format(p=prod, n=int(rng.integers(20, 400)))
            elif r < 0.19:  # noisy label: ticket text belongs to another category (imperfect human labelling)
                other = rng.choice([c for c in cats if c != cat])
                subject = rng.choice(T[other][0]).format(p=prod, n=int(rng.integers(20, 400))); desc = rng.choice(T[other][1]).format(p=prod, n=int(rng.integers(20, 400)))
            if rng.random() < 0.07:  # vague tickets that blur category boundaries
                desc = rng.choice(["Something is not working as expected.", "Need help with my account.", "Please look into this issue."])
            score = {"Performance & Outage": 2.2, "Integration & API": 0.9, "Bug / Defect": 0.7, "Billing & Payments": 0.6}.get(cat, 0.0) + (1.5 if urgent else 0) + rng.normal(0, .7)
            prio = "Critical" if score > 2.7 else "High" if score > 1.4 else "Medium" if score > 0.35 else "Low"
            ch = rng.choice(CHANNELS, p=[.32, .25, .2, .13, .10])
            created = day + pd.Timedelta(seconds=int(rng.integers(7 * 3600, 20 * 3600)))
            sla = created + pd.Timedelta(hours=SLA_H[prio])
            frt = max(0.02, rng.lognormal(np.log(CH_FRT[ch]), 0.6))
            mult = CAT_RESOLVE[cat] * CH_RESOLVE[ch] * (1.25 if created.dayofweek >= 5 else 1.0) * (1.3 if created.hour >= 17 else 1.0)
            res_h = SLA_H[prio] * 0.55 * mult * rng.lognormal(0, 0.6)
            resolved = created + pd.Timedelta(hours=res_h)
            first = created + pd.Timedelta(hours=frt)
            rows.append(dict(ticket_id=None, subject=subject, description=desc, category=cat, priority=prio, status="Resolved",
                             created_at=created, resolved_at=resolved, first_response_at=first, sla_deadline=sla,
                             product=prod, channel=ch, customer_id=rng.choice(customers, p=cust_p)))
    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{100000 + i}" for i in range(len(df))]
    asof = END
    recent = df["created_at"] > asof - pd.Timedelta(days=6)
    backlog = (rng.random(len(df)) < 0.012) | (recent & (rng.random(len(df)) < 0.30))  # some tickets simply remain open
    open_mask = (df["resolved_at"] > asof) | backlog
    df.loc[open_mask, "resolved_at"] = pd.NaT
    df.loc[open_mask, "status"] = rng.choice(["Open", "In Progress", "Pending"], size=int(open_mask.sum()), p=[.4, .45, .15])
    unanswered = (df["first_response_at"] > asof)
    df.loc[unanswered, "first_response_at"] = pd.NaT
    df.loc[open_mask & df["first_response_at"].notna() & (df["status"] == "Open"), "status"] = "In Progress"
    for c in ("created_at", "resolved_at", "first_response_at", "sla_deadline"):
        df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
