"""
CrowdVision AI audit logging.

The audit log is intentionally file-based for the current local demo. It can be
migrated to PostgreSQL later without changing the frontend contract.
"""

import json
import os
import time
from datetime import datetime

AUDIT_LOG_FILE = "data/audit_log.json"
AUDIT_LOG_LIMIT = 2000


def ensure_audit_file():
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    if not os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as file:
            json.dump([], file, indent=2)


def safe_epoch(value):
    if value in [None, "", 0]:
        return 0.0
    try:
        return float(value)
    except Exception:
        pass
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.0


def normalize_audit_record(record, fallback_index=0):
    if not isinstance(record, dict):
        record = {}

    now_epoch = time.time()
    epoch = safe_epoch(record.get("epoch")) or safe_epoch(record.get("time")) or now_epoch
    time_text = record.get("time") or datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")

    status = str(record.get("status") or "Info").title()
    severity = str(record.get("severity") or "normal").lower()

    if status.lower() in {"failed", "rejected", "blocked", "error"} and severity == "normal":
        severity = "warning"
    if status.lower() in {"critical", "danger"}:
        severity = "critical"

    return {
        "id": str(record.get("id") or f"audit_{int(epoch * 1000)}_{fallback_index}"),
        "time": str(time_text),
        "epoch": epoch,
        "actor": str(record.get("actor") or "System"),
        "action": str(record.get("action") or "System event"),
        "category": str(record.get("category") or "system").lower(),
        "source": str(record.get("source") or "CrowdVision AI"),
        "status": status,
        "severity": severity,
        "details": str(record.get("details") or "No details provided."),
        "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
    }


def read_audit_log(limit=250, category="all", status="all"):
    ensure_audit_file()
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []

    records = [normalize_audit_record(item, index) for index, item in enumerate(data)]
    records = sorted(records, key=lambda item: item.get("epoch", 0), reverse=True)

    category_value = str(category or "all").lower()
    if category_value != "all":
        records = [item for item in records if str(item.get("category", "")).lower() == category_value]

    status_value = str(status or "all").lower()
    if status_value != "all":
        records = [item for item in records if str(item.get("status", "")).lower() == status_value]

    try:
        limit = max(1, min(int(limit), AUDIT_LOG_LIMIT))
    except Exception:
        limit = 250

    return records[:limit]


def save_audit_log(records):
    ensure_audit_file()
    clean_records = [normalize_audit_record(item, index) for index, item in enumerate(records)]
    clean_records = sorted(clean_records, key=lambda item: item.get("epoch", 0), reverse=True)[:AUDIT_LOG_LIMIT]

    temp_path = f"{AUDIT_LOG_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(clean_records, file, indent=2)
    os.replace(temp_path, AUDIT_LOG_FILE)


def append_audit_log(
    action,
    category="system",
    actor="System",
    source="CrowdVision AI",
    status="Info",
    details="",
    severity="normal",
    metadata=None,
):
    now_epoch = time.time()
    item = normalize_audit_record({
        "id": f"audit_{int(now_epoch * 1000)}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": now_epoch,
        "actor": actor,
        "action": action,
        "category": category,
        "source": source,
        "status": status,
        "severity": severity,
        "details": details or action,
        "metadata": metadata or {},
    })

    records = read_audit_log(limit=AUDIT_LOG_LIMIT)
    records.insert(0, item)
    save_audit_log(records)
    return item


def summarize_audit_log(records=None):
    records = records if records is not None else read_audit_log(limit=AUDIT_LOG_LIMIT)
    categories = {}
    statuses = {}
    warnings = 0
    rejections = 0

    for item in records:
        category = str(item.get("category", "system")).lower()
        status = str(item.get("status", "Info")).title()
        categories[category] = categories.get(category, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        if str(item.get("severity", "normal")).lower() in {"warning", "critical", "high"}:
            warnings += 1
        if status.lower() in {"rejected", "blocked", "failed"}:
            rejections += 1

    latest = records[0] if records else None
    return {
        "total_records": len(records),
        "warnings": warnings,
        "rejections": rejections,
        "categories": categories,
        "statuses": statuses,
        "latest_action": latest.get("action") if latest else "None",
        "latest_time": latest.get("time") if latest else None,
    }