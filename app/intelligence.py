def calculate_congestion_score(
    zone_count,
    occupancy,
    active_tracking_count
):
    score = (
        zone_count * 5 +
        occupancy * 3 +
        active_tracking_count * 2
    )

    return min(score, 100)


def get_congestion_level(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 35:
        return "Medium"
    return "Low"