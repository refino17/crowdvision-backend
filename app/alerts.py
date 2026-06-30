def get_alert_message(density_level):
    if density_level == "Critical":
        return "DANGER: Critical crowd density!"
    elif density_level == "High":
        return "WARNING: High crowd density!"
    elif density_level == "Medium":
        return "NOTICE: Moderate crowd density."
    else:
        return "SAFE: Crowd level is low."