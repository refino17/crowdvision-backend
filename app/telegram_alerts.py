"""
CrowdVision AI Telegram alert service.

No bot token is hardcoded. Configure alerts through .env:
TELEGRAM_ALERTS_ENABLED=true
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
"""

import json
import os
import time
from datetime import datetime

try:
    import requests
except Exception:  # pragma: no cover - optional at runtime
    requests = None

try:
    from app.security_config import get_env, get_env_bool, mask_secret
except Exception:
    from security_config import get_env, get_env_bool, mask_secret

TELEGRAM_HISTORY_FILE = "data/telegram_sent.json"
TELEGRAM_HISTORY_LIMIT = 500

SEVERITY_RANK = {
    "normal": 0,
    "info": 0,
    "medium": 1,
    "warning": 1,
    "high": 2,
    "critical": 3,
    "danger": 3,
}


def telegram_enabled():
    return get_env_bool("TELEGRAM_ALERTS_ENABLED", False)


def telegram_token():
    return get_env("TELEGRAM_BOT_TOKEN", "").strip()


def telegram_chat_id():
    return get_env("TELEGRAM_CHAT_ID", "").strip()


def telegram_configured():
    return bool(telegram_enabled() and telegram_token() and telegram_chat_id())


def telegram_min_severity():
    return get_env("TELEGRAM_MIN_SEVERITY", "warning").strip().lower() or "warning"


def telegram_cooldown_seconds():
    try:
        return max(0, int(float(get_env("TELEGRAM_COOLDOWN_SECONDS", "60"))))
    except Exception:
        return 60


def get_telegram_status():
    return {
        "enabled": telegram_enabled(),
        "configured": telegram_configured(),
        "bot_token": mask_secret(telegram_token()),
        "chat_id": mask_secret(telegram_chat_id(), visible=3),
        "min_severity": telegram_min_severity(),
        "cooldown_seconds": telegram_cooldown_seconds(),
        "history_file": TELEGRAM_HISTORY_FILE,
        "requests_available": requests is not None,
    }


def ensure_history_file():
    os.makedirs(os.path.dirname(TELEGRAM_HISTORY_FILE), exist_ok=True)
    if not os.path.exists(TELEGRAM_HISTORY_FILE):
        with open(TELEGRAM_HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump([], file, indent=2)


def read_history():
    ensure_history_file()
    try:
        with open(TELEGRAM_HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history):
    ensure_history_file()
    history = sorted(history, key=lambda item: float(item.get("epoch", 0)), reverse=True)[:TELEGRAM_HISTORY_LIMIT]
    temp_path = f"{TELEGRAM_HISTORY_FILE}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    os.replace(temp_path, TELEGRAM_HISTORY_FILE)


def should_send(severity, dedupe_key):
    if not telegram_configured() or requests is None:
        return False, "Telegram is not configured or requests is unavailable."

    min_rank = SEVERITY_RANK.get(telegram_min_severity(), 1)
    severity_rank = SEVERITY_RANK.get(str(severity or "normal").lower(), 0)

    if severity_rank < min_rank:
        return False, f"Severity {severity} is below Telegram threshold {telegram_min_severity()}."

    if dedupe_key:
        now = time.time()
        cooldown = telegram_cooldown_seconds()
        for item in read_history():
            if item.get("dedupe_key") == dedupe_key and now - float(item.get("epoch", 0)) < cooldown:
                return False, "Duplicate Telegram alert suppressed by cooldown."

    return True, "Ready"


def emoji_for_severity(severity):
    severity = str(severity or "normal").lower()
    if severity in {"critical", "danger"}:
        return "🚨"
    if severity in {"high", "warning"}:
        return "⚠️"
    if severity == "medium":
        return "🟡"
    return "✅"


def format_message(title, message, severity="normal", category="system", source="CrowdVision AI", action="Monitor"):
    icon = emoji_for_severity(severity)
    return (
        f"{icon} CrowdVision AI Alert\n\n"
        f"Type: {title}\n"
        f"Category: {str(category).replace('_', ' ').title()}\n"
        f"Source: {source}\n"
        f"Severity: {str(severity).title()}\n"
        f"Message: {message}\n"
        f"Action: {action}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def send_telegram_alert(
    title,
    message,
    severity="normal",
    category="system",
    source="CrowdVision AI",
    action="Monitor",
    metadata=None,
    dedupe_key=None,
    force=False,
):
    if not force:
        ready, reason = should_send(severity, dedupe_key)
        if not ready:
            return {"sent": False, "reason": reason}
    elif not telegram_configured() or requests is None:
        return {"sent": False, "reason": "Telegram is not configured or requests is unavailable."}

    text = format_message(title, message, severity, category, source, action)
    url = f"https://api.telegram.org/bot{telegram_token()}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id(),
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        response_data = response.json()

        history = read_history()
        history.insert(0, {
            "epoch": time.time(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "severity": severity,
            "category": category,
            "source": source,
            "dedupe_key": dedupe_key or f"telegram:{category}:{title}:{source}",
            "telegram_message_id": response_data.get("result", {}).get("message_id"),
        })
        save_history(history)

        return {"sent": True, "reason": "Telegram alert sent.", "response": response_data}
    except Exception as error:
        return {"sent": False, "reason": str(error)}


def send_telegram_alert_from_record(record, force=False):
    record = record if isinstance(record, dict) else {}
    return send_telegram_alert(
        title=record.get("title", "CrowdVision Alert"),
        message=record.get("message", "CrowdVision event recorded."),
        severity=record.get("severity", "normal"),
        category=record.get("category", record.get("type", "system")),
        source=record.get("source", "CrowdVision AI"),
        action=record.get("action", "Monitor"),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
        dedupe_key=record.get("dedupe_key"),
        force=force,
    )