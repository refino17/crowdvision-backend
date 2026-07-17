import json
import os
import time
from datetime import datetime

try:
    from app.telegram_alerts import send_telegram_alert_from_record
except Exception:
    try:
        from telegram_alerts import send_telegram_alert_from_record
    except Exception:
        send_telegram_alert_from_record = None

try:
    from app.audit import append_audit_log
except Exception:
    try:
        from audit import append_audit_log
    except Exception:
        append_audit_log = None

try:
    from config import NOTIFICATION_FILE, NOTIFICATION_LIMIT, NOTIFICATION_COOLDOWN_SECONDS
except Exception:
    NOTIFICATION_FILE = "data/notifications.json"
    NOTIFICATION_LIMIT = 50
    NOTIFICATION_COOLDOWN_SECONDS = 20


def get_alert_message(density_level):
    if density_level == "Critical":
        return "DANGER: Critical crowd density!"
    if density_level == "High":
        return "WARNING: High crowd density!"
    if density_level == "Medium":
        return "NOTICE: Moderate crowd density."
    return "SAFE: Crowd level is low."


def density_to_severity(density_level):
    if density_level == "Critical":
        return "critical"
    if density_level == "High":
        return "high"
    if density_level == "Medium":
        return "medium"
    return "normal"


def ensure_notification_file():
    os.makedirs(os.path.dirname(NOTIFICATION_FILE), exist_ok=True)

    if not os.path.exists(NOTIFICATION_FILE):
        with open(NOTIFICATION_FILE, "w") as file:
            json.dump([], file, indent=2)


def load_notifications(limit=None):
    ensure_notification_file()

    try:
        with open(NOTIFICATION_FILE, "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        notifications = data
    except Exception:
        notifications = []

    notifications = sorted(
        notifications,
        key=lambda item: float(item.get("epoch", 0)),
        reverse=True
    )

    if limit is None:
        limit = NOTIFICATION_LIMIT

    return notifications[: int(limit)]


def save_notifications(notifications):
    ensure_notification_file()
    temp_path = f"{NOTIFICATION_FILE}.tmp"

    clean_notifications = sorted(
        notifications,
        key=lambda item: float(item.get("epoch", 0)),
        reverse=True
    )[: int(NOTIFICATION_LIMIT)]

    with open(temp_path, "w") as file:
        json.dump(clean_notifications, file, indent=2)

    os.replace(temp_path, NOTIFICATION_FILE)


def _recent_duplicate_exists(notifications, dedupe_key, now_epoch, cooldown_seconds):
    if not dedupe_key:
        return False

    for item in notifications:
        if item.get("dedupe_key") != dedupe_key:
            continue

        previous_epoch = float(item.get("epoch", 0))
        if now_epoch - previous_epoch < cooldown_seconds:
            return True

    return False


def append_notification(
    title,
    message,
    severity="normal",
    category="system",
    source="CrowdVision AI",
    action="Monitor",
    metadata=None,
    dedupe_key=None,
    cooldown_seconds=None,
):
    """
    Store a professional notification record for dashboard display.

    This is intentionally file-based so it works with your current local FastAPI
    project without requiring a database. It is safe for your existing project
    structure and can later be replaced by Supabase/Postgres when you deploy.
    """
    now_epoch = time.time()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if cooldown_seconds is None:
        cooldown_seconds = NOTIFICATION_COOLDOWN_SECONDS

    notifications = load_notifications(limit=NOTIFICATION_LIMIT)

    if _recent_duplicate_exists(notifications, dedupe_key, now_epoch, cooldown_seconds):
        return None

    item = {
        "id": f"cvn_{int(now_epoch * 1000)}",
        "time": now_text,
        "epoch": now_epoch,
        "title": str(title),
        "message": str(message),
        "severity": str(severity or "normal").lower(),
        "category": str(category or "system").lower(),
        "source": str(source or "CrowdVision AI"),
        "action": str(action or "Monitor"),
        "dedupe_key": dedupe_key or f"{category}:{title}",
        "metadata": metadata or {},
    }

    notifications.insert(0, item)
    save_notifications(notifications)

    if send_telegram_alert_from_record is not None:
        telegram_result = send_telegram_alert_from_record(item)
        if append_audit_log is not None and telegram_result.get("sent"):
            append_audit_log(
                action="Telegram alert sent",
                category="telegram",
                actor="CrowdVision AI",
                source=item.get("source", "CrowdVision AI"),
                status="Sent",
                severity=item.get("severity", "normal"),
                details=f"Telegram alert delivered for {item.get('title')}",
                metadata={"notification_id": item.get("id"), "title": item.get("title")}
            )

    return item


def record_density_notification(
    density_level,
    camera_profile,
    zone_count,
    occupancy,
    danger_zone,
    alert_message,
):
    severity = density_to_severity(density_level)

    if severity not in ["high", "critical"]:
        return None

    title = f"{density_level} Crowd Density"
    message = (
        f"{alert_message} Source {camera_profile} has {zone_count} people in the monitored zone "
        f"with occupancy {occupancy}. Danger zone: {danger_zone or 'None'}."
    )

    action = "Dispatch security immediately and redirect flow." if severity == "critical" else "Increase monitoring and prepare crowd-control response."

    return append_notification(
        title=title,
        message=message,
        severity=severity,
        category="density",
        source=camera_profile,
        action=action,
        metadata={
            "density_level": density_level,
            "zone_count": zone_count,
            "occupancy": occupancy,
            "danger_zone": danger_zone,
        },
        dedupe_key=f"density:{camera_profile}:{density_level}:{danger_zone}",
    )


def record_incident_notification(camera_profile, density_level, incident_data):
    if not incident_data or not incident_data.get("active"):
        return None

    danger_zone = incident_data.get("danger_zone", "None")
    duration = incident_data.get("duration", 0)
    recommendation = incident_data.get("recommendation", "Monitor")
    severity = "critical" if density_level == "Critical" else "high"

    return append_notification(
        title="Active Crowd Incident",
        message=(
            f"Incident active on {camera_profile}. Zone: {danger_zone}. "
            f"Duration: {duration}s. Recommended action: {recommendation}"
        ),
        severity=severity,
        category="incident",
        source=camera_profile,
        action=recommendation,
        metadata={
            "density_level": density_level,
            "danger_zone": danger_zone,
            "duration": duration,
        },
        dedupe_key=f"incident:{camera_profile}:{danger_zone}:{density_level}",
    )


def record_anomaly_notification(camera_profile, anomaly_data):
    if not anomaly_data or not anomaly_data.get("anomaly_detected"):
        return None

    severity_map = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "normal": "normal",
    }

    anomaly_type = anomaly_data.get("anomaly_type", "Anomaly")
    severity_label = str(anomaly_data.get("anomaly_severity", "medium")).lower()
    severity = severity_map.get(severity_label, "medium")
    score = anomaly_data.get("anomaly_score", 0)
    zone = anomaly_data.get("anomaly_zone", "None")
    recommendation = anomaly_data.get("anomaly_recommendation", "Monitor")

    return append_notification(
        title=f"{anomaly_type} Detected",
        message=(
            f"Anomaly detected on {camera_profile}. Score: {score}%. "
            f"Zone: {zone}. Recommended action: {recommendation}"
        ),
        severity=severity,
        category="anomaly",
        source=camera_profile,
        action=recommendation,
        metadata={
            "anomaly_type": anomaly_type,
            "anomaly_score": score,
            "anomaly_zone": zone,
        },
        dedupe_key=f"anomaly:{camera_profile}:{anomaly_type}:{zone}",
    )


def record_camera_health_notification(camera_profile, camera_health_data):
    if not camera_health_data or not camera_health_data.get("tamper_detected"):
        return None

    tamper_type = camera_health_data.get("tamper_type", "Camera Health Warning")
    severity_label = str(camera_health_data.get("severity", "warning")).lower()
    severity = "critical" if severity_label == "critical" else "high"
    message = camera_health_data.get("message", "Camera signal requires attention.")
    signal_quality = camera_health_data.get("signal_quality", "Unknown")

    action = "Check camera source, lens, stream, lighting, cable, or network immediately."

    return append_notification(
        title=f"Camera Integrity Alert: {tamper_type}",
        message=f"{message} Source: {camera_profile}. Signal quality: {signal_quality}.",
        severity=severity,
        category="camera_health",
        source=camera_profile,
        action=action,
        metadata=camera_health_data,
        dedupe_key=f"camera_health:{camera_profile}:{tamper_type}",
        cooldown_seconds=max(30, int(NOTIFICATION_COOLDOWN_SECONDS)),
    )


def record_engine_notification(title, message, severity="normal", source="CrowdVision AI"):
    return append_notification(
        title=title,
        message=message,
        severity=severity,
        category="engine",
        source=source,
        action="Check monitoring engine status.",
        dedupe_key=f"engine:{title}:{source}",
        cooldown_seconds=10,
    )