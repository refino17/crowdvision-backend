from collections import deque

people_history = deque(maxlen=10)
occupancy_history = deque(maxlen=10)
entries_history = deque(maxlen=10)
density_history = deque(maxlen=5)


def get_anomaly_recommendation(anomaly_type, severity):
    if severity == "Critical":
        return "Dispatch security immediately and investigate abnormal crowd behavior."
    if anomaly_type == "Crowd Surge":
        return "Increase monitoring and prepare crowd-control response."
    if anomaly_type == "Occupancy Spike":
        return "Redirect crowd flow away from the overloaded area."
    if anomaly_type == "Entry Flood":
        return "Control entry points and slow down incoming movement."
    if anomaly_type == "Risk Escalation":
        return "Prepare immediate response as crowd risk is escalating."
    if anomaly_type == "Zone Overload":
        return "Redirect people away from the overloaded zone and deploy staff to control movement."
    return "Monitor"


def calculate_anomaly_score(value, threshold):
    if threshold <= 0:
        return 0
    return max(0, min(int((value / threshold) * 100), 100))


def analyze_anomaly(total_people, occupancy, line_entries, density_level, danger_zone):
    previous_people = people_history[-1] if people_history else total_people
    previous_occupancy = occupancy_history[-1] if occupancy_history else occupancy
    previous_entries = entries_history[-1] if entries_history else line_entries

    people_jump = total_people - previous_people
    occupancy_jump = occupancy - previous_occupancy
    entry_jump = line_entries - previous_entries

    people_history.append(total_people)
    occupancy_history.append(occupancy)
    entries_history.append(line_entries)
    density_history.append(density_level)

    anomaly_detected = False
    anomaly_type = "Normal"
    anomaly_score = 0

    if people_jump >= 3:
        anomaly_detected = True
        anomaly_type = "Crowd Surge"
        anomaly_score = calculate_anomaly_score(people_jump, 5)

    elif occupancy_jump >= 2:
        anomaly_detected = True
        anomaly_type = "Occupancy Spike"
        anomaly_score = calculate_anomaly_score(occupancy_jump, 4)

    elif entry_jump >= 3:
        anomaly_detected = True
        anomaly_type = "Entry Flood"
        anomaly_score = calculate_anomaly_score(entry_jump, 5)

    elif density_level == "Critical" and "Low" in density_history:
        anomaly_detected = True
        anomaly_type = "Risk Escalation"
        anomaly_score = 75

    elif density_level == "Critical" and occupancy >= 8:
        anomaly_detected = True
        anomaly_type = "Zone Overload"
        anomaly_score = 70

    if anomaly_score >= 85:
        severity = "Critical"
    elif anomaly_score >= 65:
        severity = "High"
    elif anomaly_score >= 40:
        severity = "Medium"
    else:
        severity = "Normal"

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_type": anomaly_type,
        "anomaly_score": anomaly_score,
        "anomaly_severity": severity,
        "anomaly_zone": danger_zone if anomaly_detected else "None",
        "anomaly_recommendation": get_anomaly_recommendation(anomaly_type, severity)
    }