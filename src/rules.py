"""Deterministic business rules (NOT machine learning). Kept separate from model output so the UI
can label each recommendation honestly."""
from __future__ import annotations

import re

STRONG = ["outage", "data loss", "security breach", "breach", "unauthorized access", "production is blocked", "production blocked", "all users", "entire team", "service is down", "is down", "503"]
WEAK = ["urgent", "asap", "critical", "cannot log in", "cannot access", "blocked", "crash", "timeout", "timing out", "refund", "double charged", "charged twice", "failing", "unavailable"]
_ORDER = ["Low", "Medium", "High", "Critical"]


def urgency_signals(text: str) -> dict:
    t = (text or "").lower()
    strong = [w for w in STRONG if re.search(r"\b" + re.escape(w) + r"\b", t)]
    weak = [w for w in WEAK if re.search(r"\b" + re.escape(w) + r"\b", t)]
    rule = "Critical" if strong else ("High" if len(weak) >= 2 else ("Medium" if weak else None))
    return {"strong": strong, "weak": weak, "rule_priority": rule}


def combine_priority(ml_priority: str | None, rule_priority: str | None) -> tuple[str | None, str]:
    """Recommend the higher of the ML and rule priorities, and say where it came from."""
    cands = [(p, s) for p, s in ((ml_priority, "model"), (rule_priority, "keyword rule")) if p in _ORDER]
    if not cands:
        return None, "No priority model or keyword signal available."
    best = max(cands, key=lambda c: _ORDER.index(c[0]))
    if len(cands) == 2 and cands[0][0] != cands[1][0]:
        return best[0], f"Highest of model ({ml_priority}) and keyword rule ({rule_priority}); the {best[1]} drove the result."
    return best[0], f"Based on the {best[1]} ({best[0]})."


def recommended_actions(category: str | None, priority: str | None, strong_signals: list[str]) -> list[str]:
    c = (category or "").lower()
    acts = []
    if any(k in c for k in ("outage", "performance", "incident")):
        acts += ["Check the status page / monitoring and open an incident if more than one customer is affected.", "Notify the on-call engineer."]
    elif any(k in c for k in ("billing", "payment", "invoice", "refund")):
        acts += ["Verify the charge in the billing system before replying.", "Route to the billing operations queue."]
    elif any(k in c for k in ("login", "access", "auth", "account")):
        acts += ["Verify the requester's identity before changing access.", "Check lockout / SSO status for the account."]
    elif any(k in c for k in ("bug", "defect", "error")):
        acts += ["Ask for reproduction steps, version and logs.", "Search known issues for duplicates."]
    elif any(k in c for k in ("api", "integration")):
        acts += ["Request the request ID, timestamp and endpoint.", "Check API key scope and rate-limit status."]
    elif any(k in c for k in ("feature", "request", "enhancement")):
        acts += ["Acknowledge and log the request in the product backlog.", "Link similar existing requests."]
    elif any(k in c for k in ("data", "report")):
        acts += ["Ask for the report definition and date range.", "Compare the dashboard against the raw export."]
    else:
        acts += ["Route to the team that owns the predicted category and confirm the category manually."]
    if priority in ("Critical", "High") or strong_signals:
        acts.append("Assign a named owner immediately and set a short update cadence.")
    if priority == "Critical" or strong_signals:
        acts.append("Notify the duty manager.")
    return acts
