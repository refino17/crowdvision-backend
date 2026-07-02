from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse
from io import StringIO, BytesIO
from datetime import datetime
import pandas as pd
import os
import time
import subprocess
import signal
import sys
import json

from app.report_generator import txt_report_to_pdf
from app.config import (
    get_camera_profiles as load_camera_profiles,
    get_active_camera_profile,
    set_active_camera_profile,
    add_camera_source_profile,
    delete_camera_source_profile,
    get_performance_status,
    set_saved_performance_mode,
    resolve_performance_mode,
    LIVE_STREAM_SLEEP,
    CAMERA_STATUS_FILE,
    CAMERA_OFFLINE_AFTER_SECONDS,
    EVIDENCE_DIRS,
    NOTIFICATION_LIMIT
)

app = FastAPI(title="CrowdVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVENTS_FILE = "data/events.csv"
LIVE_FRAME_PATH = "data/live/latest_frame.jpg"
REPORTS_DIR = "data/reports"
UPLOADS_DIR = "data/uploads"

ENGINE_PROCESS = None
PROJECT_ROOT = os.getcwd()
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "app", "main.py")
ENGINE_PID_FILE = "data/engine.pid"
ENGINE_LOG_FILE = "data/engine.log"

os.makedirs("data/live", exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app.mount("/live", StaticFiles(directory="data/live"), name="live")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

for _evidence_dir in ["data/evidence", "data/snapshots", "data/alerts"]:
    os.makedirs(_evidence_dir, exist_ok=True)

    try:
        app.mount(
            f"/{_evidence_dir.replace('data/', '')}",
            StaticFiles(directory=_evidence_dir),
            name=_evidence_dir.replace("/", "_")
        )
    except RuntimeError:
        pass


def load_events():
    if not os.path.exists(EVENTS_FILE):
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_FILE)

    if df.empty:
        return df

    return df.fillna(0)


def default_tracking_summary():
    return {
        "session_unique_people": 0,
        "active_tracks": 0,
        "memory_tracks": 0,
        "reidentified_people": 0,
        "duplicates_prevented": 0,
        "raw_tracker_continuity": 0,
        "tracking_quality": "Waiting",
        "memory_seconds": 0,
        "privacy_mode": "Anonymous body tracking"
    }


def normalize_tracking_summary(status):
    if not status or not isinstance(status, dict):
        return default_tracking_summary()

    tracking = status.get("tracking")

    if not isinstance(tracking, dict):
        return default_tracking_summary()

    normalized = default_tracking_summary()

    for key in normalized.keys():
        if key in tracking:
            normalized[key] = tracking.get(key)

    integer_keys = [
        "session_unique_people",
        "active_tracks",
        "memory_tracks",
        "reidentified_people",
        "duplicates_prevented",
        "raw_tracker_continuity",
        "memory_seconds"
    ]

    for key in integer_keys:
        try:
            normalized[key] = int(float(normalized.get(key, 0)))
        except Exception:
            normalized[key] = 0

    normalized["privacy_mode"] = tracking.get("privacy_mode", "Anonymous body tracking")
    normalized["tracking_quality"] = tracking.get("tracking_quality", "Tracking active")

    return normalized


def read_camera_status_file():
    if not os.path.exists(CAMERA_STATUS_FILE):
        return None

    try:
        with open(CAMERA_STATUS_FILE, "r") as file:
            return json.load(file)
    except Exception:
        return None


def write_camera_status_file(status):
    try:
        os.makedirs(os.path.dirname(CAMERA_STATUS_FILE), exist_ok=True)

        with open(CAMERA_STATUS_FILE, "w") as file:
            json.dump(status, file, indent=4)

        return True
    except Exception as error:
        print(f"Unable to write camera status file: {error}")
        return False


def mark_camera_status_stopped(reason="Monitoring engine stopped by operator"):
    previous_status = read_camera_status_file() or {}
    performance = get_performance_status()
    now = time.time()

    stopped_status = {
        **previous_status,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at_epoch": now,
        "active_profile": get_active_camera_profile(),
        "source": previous_status.get("source", ""),
        "online": False,
        "status": "Stopped",
        "reason": reason,
        "fps": 0,
        "ai_fps": 0,
        "mode": performance.get("resolved_mode", "LOW"),
        "tracking": normalize_tracking_summary(previous_status)
    }

    write_camera_status_file(stopped_status)


def generate_live_stream():
    last_frame = None

    while True:
        try:
            if (
                os.path.exists(LIVE_FRAME_PATH)
                and os.path.getsize(LIVE_FRAME_PATH) > 0
            ):
                with open(LIVE_FRAME_PATH, "rb") as image_file:
                    frame = image_file.read()

                if frame:
                    last_frame = frame

            if last_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n\r\n"
                    + last_frame
                    + b"\r\n"
                )

        except Exception as error:
            print(f"Live stream error: {error}")

        time.sleep(LIVE_STREAM_SLEEP)


def get_forecast_message(predicted_occupancy, risk_trend):
    if predicted_occupancy >= 10 and risk_trend == "Increasing":
        return "Critical congestion likely soon. Prepare immediate response."

    if predicted_occupancy >= 8:
        return "High crowd pressure expected. Increase monitoring."

    if risk_trend == "Increasing":
        return "Crowd level is rising. Continue close observation."

    if risk_trend == "Decreasing":
        return "Crowd level is reducing. Maintain monitoring."

    return "Crowd level appears stable."


def get_saved_engine_pid():
    if not os.path.exists(ENGINE_PID_FILE):
        return None

    try:
        with open(ENGINE_PID_FILE, "r") as file:
            pid = int(file.read().strip())

        return pid
    except Exception:
        return None


def save_engine_pid(pid):
    os.makedirs("data", exist_ok=True)

    with open(ENGINE_PID_FILE, "w") as file:
        file.write(str(pid))


def clear_engine_pid():
    if os.path.exists(ENGINE_PID_FILE):
        try:
            os.remove(ENGINE_PID_FILE)
        except Exception:
            pass


def is_pid_running(pid):
    if not pid:
        return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def is_engine_running():
    global ENGINE_PROCESS

    if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
        return True

    saved_pid = get_saved_engine_pid()

    if saved_pid and is_pid_running(saved_pid):
        return True

    clear_engine_pid()
    ENGINE_PROCESS = None
    return False


def get_engine_pid():
    if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
        return ENGINE_PROCESS.pid

    saved_pid = get_saved_engine_pid()

    if saved_pid and is_pid_running(saved_pid):
        return saved_pid

    return None


def stop_pid(pid):
    if not pid:
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    for _ in range(20):
        if not is_pid_running(pid):
            return

        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


@app.get("/api/system/performance")
def get_system_performance():
    return get_performance_status()


@app.post("/api/system/performance/{mode}")
def update_performance_mode(mode: str):
    mode = mode.upper()

    if not set_saved_performance_mode(mode):
        raise HTTPException(status_code=400, detail="Invalid performance mode")

    status = get_performance_status()

    return {
        "message": f"Performance mode updated to {mode}",
        "performance": status,
        "restart_required": True
    }


@app.get("/api/engine/status")
def get_engine_status():
    running = is_engine_running()
    performance = get_performance_status()

    return {
        "running": running,
        "pid": get_engine_pid() if running else None,
        "active_profile": get_active_camera_profile(),
        "mode": performance["resolved_mode"],
        "selected_mode": performance["selected_mode"],
        "recommended_mode": performance["recommended_mode"],
        "system": performance["system"]
    }


@app.post("/api/engine/start")
def start_engine():
    global ENGINE_PROCESS

    performance = get_performance_status()

    if is_engine_running():
        return {
            "message": "AI monitoring engine is already running",
            "running": True,
            "pid": get_engine_pid(),
            "active_profile": get_active_camera_profile(),
            "mode": performance["resolved_mode"],
            "selected_mode": performance["selected_mode"],
            "recommended_mode": performance["recommended_mode"]
        }

    if not os.path.exists(MAIN_SCRIPT):
        raise HTTPException(status_code=404, detail="app/main.py not found")

    os.makedirs("data", exist_ok=True)

    env = os.environ.copy()
    env["CROWDVISION_ENGINE_MODE"] = "web"
    env["CROWDVISION_HEADLESS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDVISION_PERFORMANCE_MODE"] = performance["selected_mode"]
    env["CROWDVISION_RESOLVED_PERFORMANCE_MODE"] = performance["resolved_mode"]

    if performance["resolved_mode"] in ["LOW", "BALANCED"]:
        env["CROWDVISION_LOW_RESOURCE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    else:
        env["CROWDVISION_LOW_RESOURCE"] = "0"

    log_file = open(ENGINE_LOG_FILE, "a")

    ENGINE_PROCESS = subprocess.Popen(
        [sys.executable, MAIN_SCRIPT],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=env
    )

    save_engine_pid(ENGINE_PROCESS.pid)

    return {
        "message": f"AI monitoring engine started in {performance['resolved_mode']} mode",
        "running": True,
        "pid": ENGINE_PROCESS.pid,
        "active_profile": get_active_camera_profile(),
        "mode": performance["resolved_mode"],
        "selected_mode": performance["selected_mode"],
        "recommended_mode": performance["recommended_mode"]
    }


@app.post("/api/engine/stop")
def stop_engine():
    global ENGINE_PROCESS

    pid = get_engine_pid()

    if not pid:
        ENGINE_PROCESS = None
        clear_engine_pid()
        mark_camera_status_stopped("Monitoring engine is not running")

        return {
            "message": "AI monitoring engine is not running",
            "running": False
        }

    stop_pid(pid)

    ENGINE_PROCESS = None
    clear_engine_pid()
    mark_camera_status_stopped("Monitoring engine stopped by operator")

    return {
        "message": "AI monitoring engine stopped",
        "running": False
    }


@app.post("/api/engine/restart")
def restart_engine():
    stop_engine()
    return start_engine()


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CrowdVision AI API"
    }


@app.get("/api/video-feed")
def video_feed():
    return StreamingResponse(
        generate_live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/live-image")
def get_live_image():
    if not os.path.exists(LIVE_FRAME_PATH):
        return {
            "image": None,
            "live": False,
            "frame_age_seconds": None
        }

    status = read_camera_status_file()
    running = is_engine_running()
    now = time.time()

    frame_age = round(now - os.path.getmtime(LIVE_FRAME_PATH), 2)

    active_status_age = None

    if status and status.get("updated_at_epoch"):
        active_status_age = round(now - float(status.get("updated_at_epoch", now)), 2)

    live = bool(
        running and
        status and
        status.get("online") and
        active_status_age is not None and
        active_status_age <= CAMERA_OFFLINE_AFTER_SECONDS
    )

    return {
        "image": "http://127.0.0.1:8000/live/latest_frame.jpg",
        "live": live,
        "frame_age_seconds": frame_age
    }


@app.get("/api/reports")
def get_reports():
    reports = []

    if not os.path.exists(REPORTS_DIR):
        return {"reports": []}

    for filename in os.listdir(REPORTS_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(REPORTS_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        pdf_filename = filename.replace(".txt", ".pdf")
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

        if not os.path.exists(pdf_path):
            txt_report_to_pdf(file_path)

        report_type = "Incident" if filename.startswith("incident") else "Anomaly"

        reports.append({
            "filename": filename,
            "pdf_filename": pdf_filename,
            "type": report_type,
            "size": os.path.getsize(file_path),
            "pdf_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
            "created": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(os.path.getmtime(file_path))
            ),
            "url": f"http://127.0.0.1:8000/reports/{filename}",
            "pdf_url": f"http://127.0.0.1:8000/reports/{pdf_filename}"
        })

    reports.sort(key=lambda report: report["created"], reverse=True)

    return {"reports": reports}


@app.get("/api/reports/{filename}")
def read_report(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(REPORTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    with open(file_path, "r") as file:
        content = file.read()

    return PlainTextResponse(content)


@app.get("/api/events")
def get_events():
    df = load_events()

    if df.empty:
        return {"events": []}

    return {"events": df.tail(100).to_dict(orient="records")}


@app.get("/api/latest-event")
def get_latest_event():
    df = load_events()

    if df.empty:
        return {"latest_event": None}

    return {"latest_event": df.tail(1).to_dict(orient="records")[0]}


@app.get("/api/summary")
def get_summary():
    df = load_events()

    if df.empty:
        return {
            "total_events": 0,
            "peak_people": 0,
            "peak_occupancy": 0,
            "critical_alerts": 0,
            "high_alerts": 0,
            "latest_density": "Unknown"
        }

    return {
        "total_events": len(df),
        "peak_people": int(df["total_people"].max()),
        "peak_occupancy": int(df["peak_occupancy"].max()),
        "critical_alerts": int((df["density_level"] == "Critical").sum()),
        "high_alerts": int((df["density_level"] == "High").sum()),
        "latest_density": str(df.tail(1)["density_level"].values[0])
    }


@app.get("/api/intelligence")
def get_intelligence():
    df = load_events()

    if df.empty:
        return {
            "average_people": 0,
            "average_occupancy": 0,
            "average_zone_count": 0,
            "highest_risk_time": "None",
            "highest_risk_density": "Unknown",
            "latest_camera_profile": "None",
            "alert_rate": 0
        }

    risk_order = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }

    df = df.copy()
    df["risk_score"] = df["density_level"].map(risk_order).fillna(0)

    highest_risk_row = df.sort_values(
        by=["risk_score", "total_people", "occupancy"],
        ascending=False
    ).head(1)

    alert_count = int(df["density_level"].isin(["High", "Critical"]).sum())

    return {
        "average_people": round(float(df["total_people"].mean()), 2),
        "average_occupancy": round(float(df["occupancy"].mean()), 2),
        "average_zone_count": round(float(df["zone_count"].mean()), 2),
        "highest_risk_time": str(highest_risk_row["timestamp"].values[0]),
        "highest_risk_density": str(highest_risk_row["density_level"].values[0]),
        "latest_camera_profile": str(df.tail(1)["camera_profile"].values[0]),
        "alert_rate": round((alert_count / len(df)) * 100, 2)
    }


@app.get("/api/prediction")
def get_prediction():
    df = load_events()

    if df.empty or "predicted_people" not in df.columns:
        return {
            "current_people": 0,
            "predicted_people": 0,
            "current_occupancy": 0,
            "predicted_occupancy": 0,
            "risk_trend": "Stable",
            "forecast_message": "No prediction data available yet."
        }

    latest = df.tail(1).to_dict(orient="records")[0]

    predicted_occupancy = int(latest.get("predicted_occupancy", 0))
    risk_trend_value = str(latest.get("risk_trend", "Stable"))

    return {
        "current_people": int(latest.get("total_people", 0)),
        "predicted_people": int(latest.get("predicted_people", 0)),
        "current_occupancy": int(latest.get("occupancy", 0)),
        "predicted_occupancy": predicted_occupancy,
        "risk_trend": risk_trend_value,
        "forecast_message": get_forecast_message(predicted_occupancy, risk_trend_value)
    }


@app.get("/api/anomalies")
def get_anomalies():
    df = load_events()

    if df.empty or "anomaly_detected" not in df.columns:
        return {"anomalies": []}

    anomalies_df = df[df["anomaly_detected"].astype(str) == "True"]

    if anomalies_df.empty:
        return {"anomalies": []}

    return {"anomalies": anomalies_df.tail(50).to_dict(orient="records")}


@app.get("/api/latest-anomaly")
def get_latest_anomaly():
    df = load_events()

    if df.empty or "anomaly_detected" not in df.columns:
        return {"latest_anomaly": None}

    anomalies_df = df[df["anomaly_detected"].astype(str) == "True"]

    if anomalies_df.empty:
        return {"latest_anomaly": None}

    return {"latest_anomaly": anomalies_df.tail(1).to_dict(orient="records")[0]}


@app.get("/api/anomaly-summary")
def get_anomaly_summary():
    df = load_events()

    if df.empty or "anomaly_detected" not in df.columns:
        return {
            "total_anomalies": 0,
            "highest_anomaly_score": 0,
            "latest_anomaly_type": "Normal",
            "latest_anomaly_severity": "Normal",
            "latest_anomaly_zone": "None",
            "latest_anomaly_recommendation": "Monitor"
        }

    anomalies_df = df[df["anomaly_detected"].astype(str) == "True"]

    if anomalies_df.empty:
        return {
            "total_anomalies": 0,
            "highest_anomaly_score": 0,
            "latest_anomaly_type": "Normal",
            "latest_anomaly_severity": "Normal",
            "latest_anomaly_zone": "None",
            "latest_anomaly_recommendation": "Monitor"
        }

    latest = anomalies_df.tail(1).to_dict(orient="records")[0]

    return {
        "total_anomalies": len(anomalies_df),
        "highest_anomaly_score": int(anomalies_df["anomaly_score"].max()),
        "latest_anomaly_type": latest.get("anomaly_type", "Normal"),
        "latest_anomaly_severity": latest.get("anomaly_severity", "Normal"),
        "latest_anomaly_zone": latest.get("anomaly_zone", "None"),
        "latest_anomaly_recommendation": latest.get("anomaly_recommendation", "Monitor")
    }


@app.get("/api/incidents")
def get_incidents():
    df = load_events()

    if df.empty or "incident_active" not in df.columns:
        return {"incidents": []}

    incidents_df = df[df["incident_active"].astype(str) == "True"]

    if incidents_df.empty:
        return {"incidents": []}

    return {"incidents": incidents_df.tail(50).to_dict(orient="records")}


@app.get("/api/latest-incident")
def get_latest_incident():
    df = load_events()

    if df.empty or "incident_active" not in df.columns:
        return {"latest_incident": None}

    incidents_df = df[df["incident_active"].astype(str) == "True"]

    if incidents_df.empty:
        return {"latest_incident": None}

    return {"latest_incident": incidents_df.tail(1).to_dict(orient="records")[0]}


@app.get("/api/incident-summary")
def get_incident_summary():
    df = load_events()

    if df.empty or "incident_active" not in df.columns:
        return {
            "active_incidents": 0,
            "total_incident_events": 0,
            "longest_incident_duration": 0,
            "latest_danger_zone": "None",
            "latest_recommendation": "Monitor"
        }

    incidents_df = df[df["incident_active"].astype(str) == "True"]

    if incidents_df.empty:
        return {
            "active_incidents": 0,
            "total_incident_events": 0,
            "longest_incident_duration": 0,
            "latest_danger_zone": "None",
            "latest_recommendation": "Monitor"
        }

    latest_incident = incidents_df.tail(1).to_dict(orient="records")[0]

    return {
        "active_incidents": 1,
        "total_incident_events": len(incidents_df),
        "longest_incident_duration": int(incidents_df["incident_duration"].max()),
        "latest_danger_zone": latest_incident.get("danger_zone", "None"),
        "latest_recommendation": latest_incident.get("recommendation", "Monitor")
    }


@app.get("/api/chart-data")
def get_chart_data():
    df = load_events()

    if df.empty:
        return {"chart_data": []}

    chart_df = df.tail(50).copy()

    columns = [
        "timestamp",
        "total_people",
        "zone_count",
        "occupancy",
        "peak_occupancy",
        "line_entries",
        "line_exits",
        "density_level"
    ]

    optional_columns = [
        "predicted_people",
        "predicted_occupancy",
        "risk_trend",
        "anomaly_score",
        "anomaly_type",
        "anomaly_severity"
    ]

    for column in optional_columns:
        if column in chart_df.columns:
            columns.append(column)

    return {"chart_data": chart_df[columns].to_dict(orient="records")}


@app.get("/api/alert-distribution")
def get_alert_distribution():
    df = load_events()

    if df.empty:
        return {"alerts": []}

    counts = df["density_level"].value_counts().reset_index()
    counts.columns = ["density_level", "count"]

    return {"alerts": counts.to_dict(orient="records")}


@app.get("/api/tracking-summary")
def get_tracking_summary_endpoint():
    status = read_camera_status_file()

    return {
        "tracking": normalize_tracking_summary(status),
        "status": status
    }


def get_export_dataframe(export_type):
    df = load_events()

    if df.empty:
        return pd.DataFrame()

    if export_type == "events":
        return df

    if export_type == "incidents":
        if "incident_active" not in df.columns:
            return pd.DataFrame()

        return df[df["incident_active"].astype(str) == "True"]

    if export_type == "anomalies":
        if "anomaly_detected" not in df.columns:
            return pd.DataFrame()

        return df[df["anomaly_detected"].astype(str) == "True"]

    if export_type == "summary":
        summary_data = {
            "total_events": [len(df)],
            "peak_people": [int(df["total_people"].max())],
            "peak_occupancy": [int(df["peak_occupancy"].max())],
            "critical_alerts": [int((df["density_level"] == "Critical").sum())],
            "high_alerts": [int((df["density_level"] == "High").sum())],
            "average_people": [round(float(df["total_people"].mean()), 2)],
            "average_occupancy": [round(float(df["occupancy"].mean()), 2)],
            "exported_at": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }

        return pd.DataFrame(summary_data)

    return pd.DataFrame()


@app.get("/api/export/{export_type}/csv")
def export_csv(export_type: str):
    export_type = export_type.lower()

    if export_type not in ["events", "incidents", "anomalies", "summary"]:
        raise HTTPException(status_code=400, detail="Invalid export type")

    export_df = get_export_dataframe(export_type)

    if export_df.empty:
        raise HTTPException(status_code=404, detail="No data available for export")

    csv_buffer = StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    filename = f"crowdvision_{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.get("/api/export/{export_type}/excel")
def export_excel(export_type: str):
    export_type = export_type.lower()

    if export_type not in ["events", "incidents", "anomalies", "summary"]:
        raise HTTPException(status_code=400, detail="Invalid export type")

    export_df = get_export_dataframe(export_type)

    if export_df.empty:
        raise HTTPException(status_code=404, detail="No data available for export")

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=export_type.title())

    excel_buffer.seek(0)

    filename = f"crowdvision_{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


def get_latest_event_record():
    df = load_events()

    if df.empty:
        return None

    return df.tail(1).to_dict(orient="records")[0]


def list_evidence_files():
    evidence = []

    for directory in EVIDENCE_DIRS:
        if not os.path.exists(directory):
            continue

        for filename in os.listdir(directory):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(directory, filename)

            if not os.path.isfile(file_path):
                continue

            evidence.append({
                "filename": filename,
                "directory": directory,
                "type": "Snapshot",
                "size": os.path.getsize(file_path),
                "created": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(os.path.getmtime(file_path))
                ),
                "url": f"http://127.0.0.1:8000/{directory.replace('data/', '')}/{filename}" if directory.startswith("data/") else ""
            })

    evidence.sort(key=lambda item: item["created"], reverse=True)

    return evidence


@app.get("/api/camera-health")
def get_camera_health():
    active_profile = get_active_camera_profile()
    profiles = load_camera_profiles()
    status = read_camera_status_file()
    latest_event = get_latest_event_record()
    running = is_engine_running()
    now = time.time()

    frame_age = None
    preview_url = None

    if os.path.exists(LIVE_FRAME_PATH):
        frame_age = round(now - os.path.getmtime(LIVE_FRAME_PATH), 2)
        preview_url = "http://127.0.0.1:8000/live/latest_frame.jpg"

    active_status_age = None

    if status and status.get("updated_at_epoch"):
        active_status_age = round(now - float(status.get("updated_at_epoch", now)), 2)

    active_online = bool(
        running and
        status and
        status.get("online") and
        active_status_age is not None and
        active_status_age <= CAMERA_OFFLINE_AFTER_SECONDS
    )

    active_tracking = normalize_tracking_summary(status)
    cameras = []

    for key, profile in profiles.items():
        is_active = key == active_profile

        if is_active:
            health = "Online" if active_online else ("Starting" if running else "Stopped")

            cameras.append({
                "key": key,
                "name": profile.get("name", key),
                "source": str(profile.get("source", "")),
                "source_type": profile.get("source_type", "camera"),
                "active": True,
                "online": active_online,
                "health": health,
                "last_seen": status.get("updated_at") if status else None,
                "age_seconds": active_status_age,
                "frame_age_seconds": frame_age,
                "preview_url": preview_url,
                "people": status.get("people", latest_event.get("total_people", 0) if latest_event else 0) if status else 0,
                "density": status.get("density", latest_event.get("density_level", "Unknown") if latest_event else "Unknown") if status else "Unknown",
                "fps": status.get("fps", 0) if status else 0,
                "ai_fps": status.get("ai_fps", 0) if status else 0,
                "reason": status.get("reason", "") if status else "",
                "tracking": active_tracking
            })
        else:
            cameras.append({
                "key": key,
                "name": profile.get("name", key),
                "source": str(profile.get("source", "")),
                "source_type": profile.get("source_type", "camera"),
                "active": False,
                "online": False,
                "health": "Standby",
                "last_seen": None,
                "age_seconds": None,
                "frame_age_seconds": None,
                "preview_url": None,
                "people": 0,
                "density": "Standby",
                "fps": 0,
                "ai_fps": 0,
                "reason": "Not active",
                "tracking": default_tracking_summary()
            })

    return {
        "active_profile": active_profile,
        "engine_running": running,
        "active_online": active_online,
        "stream_live": active_online,
        "latest_frame_age_seconds": frame_age,
        "live_frame_url": preview_url,
        "tracking": active_tracking,
        "status": status,
        "cameras": cameras
    }


@app.get("/api/notifications")
def get_notifications():
    notifications = []
    latest_event = get_latest_event_record()
    camera_health = get_camera_health()
    active_camera = None

    for camera in camera_health.get("cameras", []):
        if camera.get("active"):
            active_camera = camera
            break

    if not camera_health.get("engine_running"):
        notifications.append({
            "type": "engine",
            "severity": "warning",
            "title": "AI Engine Stopped",
            "message": "Monitoring engine is not currently running.",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if active_camera and not active_camera.get("online") and camera_health.get("engine_running"):
        notifications.append({
            "type": "camera",
            "severity": "critical",
            "title": "Camera Health Warning",
            "message": f"{active_camera.get('name')} is not producing fresh frames.",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if latest_event:
        density = str(latest_event.get("density_level", "Unknown"))

        if density in ["High", "Critical"]:
            notifications.append({
                "type": "crowd",
                "severity": "critical" if density == "Critical" else "warning",
                "title": f"{density} Crowd Density",
                "message": latest_event.get("alert_message", "Crowd alert detected."),
                "time": latest_event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })

        if str(latest_event.get("incident_active", "False")) == "True":
            notifications.append({
                "type": "incident",
                "severity": "critical",
                "title": "Active Incident",
                "message": f"Danger zone: {latest_event.get('danger_zone', 'Unknown')}. {latest_event.get('recommendation', 'Monitor')}",
                "time": latest_event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })

        if str(latest_event.get("anomaly_detected", "False")) == "True":
            notifications.append({
                "type": "anomaly",
                "severity": str(latest_event.get("anomaly_severity", "warning")).lower(),
                "title": f"Anomaly: {latest_event.get('anomaly_type', 'Detected')}",
                "message": f"Score: {latest_event.get('anomaly_score', 0)}%. Zone: {latest_event.get('anomaly_zone', 'Unknown')}",
                "time": latest_event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })

    return {"notifications": notifications[:NOTIFICATION_LIMIT]}


@app.get("/api/evidence")
def get_evidence():
    return {"evidence": list_evidence_files()}


@app.get("/api/camera-profiles")
def get_camera_profiles_endpoint():
    active_profile = get_active_camera_profile()
    camera_profiles = load_camera_profiles()

    profiles = []

    for key, profile in camera_profiles.items():
        profiles.append({
            "key": key,
            "name": profile.get("name", key),
            "source": str(profile.get("source", "")),
            "source_type": profile.get("source_type", "camera"),
            "zones": len(profile.get("zones", [])),
            "active": key == active_profile
        })

    return {
        "active_profile": active_profile,
        "profiles": profiles
    }


@app.post("/api/camera-profiles/{profile_key}/activate")
def activate_camera_profile(profile_key: str):
    if profile_key not in load_camera_profiles():
        raise HTTPException(status_code=404, detail="Camera profile not found")

    updated = set_active_camera_profile(profile_key)

    if not updated:
        raise HTTPException(status_code=500, detail="Unable to update camera profile")

    return {
        "message": "Camera profile updated successfully",
        "active_profile": profile_key,
        "restart_required": True
    }


def create_source_key(prefix):
    safe_prefix = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in prefix
    ).strip("_")

    return f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def default_source_profile(name, source, source_type):
    return {
        "name": name,
        "source": source,
        "source_type": source_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zone": {
            "name": "Main Monitoring Zone",
            "x1": 40,
            "y1": 80,
            "x2": 560,
            "y2": 520
        },
        "zones": [
            {
                "name": "Main Monitoring Zone",
                "x1": 40,
                "y1": 80,
                "x2": 560,
                "y2": 520
            }
        ],
        "line": {
            "name": "Entry / Exit Line",
            "y": 300
        }
    }


@app.get("/api/sources")
def get_sources():
    return get_camera_profiles_endpoint()


@app.post("/api/sources/device")
def create_device_source(
    name: str = Form(...),
    device_index: int = Form(0),
    source_type: str = Form("webcam")
):
    key = create_source_key(source_type)
    profile = default_source_profile(name, device_index, source_type)

    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Camera device source created and selected",
        "key": key,
        "active_profile": key,
        "profile": profile,
        "restart_required": True
    }


@app.post("/api/sources/stream")
def create_stream_source(
    name: str = Form(...),
    stream_url: str = Form(...),
    source_type: str = Form("ip_camera")
):
    if not stream_url.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        raise HTTPException(
            status_code=400,
            detail="Stream URL must start with http, https, rtsp, or rtmp"
        )

    key = create_source_key(source_type)
    profile = default_source_profile(name, stream_url, source_type)

    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Stream source created and selected",
        "key": key,
        "active_profile": key,
        "profile": profile,
        "restart_required": True
    }


@app.post("/api/sources/video-upload")
async def upload_video_source(
    name: str = Form("Uploaded Video"),
    file: UploadFile = File(...)
):
    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    original_name = file.filename or "uploaded_video.mp4"
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    safe_name = "".join(
        ch if ch.isalnum() or ch in ["-", "_", "."] else "_"
        for ch in original_name
    )

    stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    file_path = os.path.join(UPLOADS_DIR, stored_name)

    with open(file_path, "wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            output_file.write(chunk)

    key = create_source_key("uploaded_video")
    profile = default_source_profile(name, file_path, "uploaded_video")

    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Video uploaded and selected successfully",
        "key": key,
        "active_profile": key,
        "filename": stored_name,
        "url": f"http://127.0.0.1:8000/uploads/{stored_name}",
        "profile": profile,
        "restart_required": True
    }


@app.delete("/api/sources/{profile_key}")
def delete_source(profile_key: str):
    active_profile = get_active_camera_profile()

    if profile_key in ["webcam", "crowd_video"]:
        raise HTTPException(status_code=400, detail="Built-in sources cannot be deleted")

    deleted = delete_camera_source_profile(profile_key)

    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")

    if active_profile == profile_key:
        set_active_camera_profile("webcam")

    return {
        "message": "Source deleted successfully",
        "deleted": profile_key,
        "active_profile": get_active_camera_profile()
    }