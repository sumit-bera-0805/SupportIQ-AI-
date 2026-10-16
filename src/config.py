"""Central configuration. Environment variables override defaults (useful on Render)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "sample_tickets.csv"
STYLES_PATH = ROOT / "assets" / "styles.css"
DB_PATH = Path(os.getenv("SUPPORTIQ_DB", str(ROOT / "data" / "supportiq.db")))

MAX_UPLOAD_MB = int(os.getenv("SUPPORTIQ_MAX_UPLOAD_MB", "25"))
MAX_ROWS = int(os.getenv("SUPPORTIQ_MAX_ROWS", "50000"))
AUTO_TRAIN_MAX_ROWS = int(os.getenv("SUPPORTIQ_AUTO_TRAIN_MAX_ROWS", "8000"))
SIMILARITY_MAX_ROWS = 20000
CLUSTER_MAX_ROWS = 5000

LOW_CONFIDENCE_DEFAULT = 0.45
DEFAULT_SLA_HOURS = {"Critical": 4, "High": 8, "Medium": 24, "Low": 72}

CANONICAL_COLUMNS = [
    "ticket_id", "subject", "description", "category", "priority", "status",
    "created_at", "resolved_at", "first_response_at", "sla_deadline",
    "product", "channel", "customer_id",
]
DATE_COLUMNS = ["created_at", "resolved_at", "first_response_at", "sla_deadline"]
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]
UNRESOLVED_STATUSES = {"Open", "In Progress", "Pending"}
RESOLVED_STATUSES = {"Resolved", "Closed"}
