from datetime import datetime

incident_active = False
incident_start = None


def get_recommendation(density_level, congestion_score):
    if density_level == "Critical":
        return "Dispatch security immediately and redirect crowd flow."

    if congestion_score >= 80:
        return "Open additional exits and reduce crowd pressure."

    if density_level == "High":
        return "Increase monitoring and prepare crowd-control response."

    return "Monitor"


def analyze_incident(density_level, danger_zone, congestion_score):
    global incident_active
    global incident_start

    now = datetime.now()

    if density_level in ["High", "Critical"]:
        if not incident_active:
            incident_active = True
            incident_start = now

        duration = int((now - incident_start).total_seconds())

        return {
            "active": True,
            "duration": duration,
            "danger_zone": danger_zone,
            "recommendation": get_recommendation(density_level, congestion_score)
        }

    incident_active = False
    incident_start = None

    return {
        "active": False,
        "duration": 0,
        "danger_zone": "None",
        "recommendation": "Monitor"
    }