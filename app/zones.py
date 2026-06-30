from density import get_density_level
from intelligence import calculate_congestion_score, get_congestion_level


def get_box_center(bbox):
    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y


def get_box_bottom_center(bbox):
    x1, y1, x2, y2 = bbox

    bottom_center_x = int((x1 + x2) / 2)
    bottom_center_y = int(y2)

    return bottom_center_x, bottom_center_y


def is_point_inside_zone(point, zone):
    x, y = point

    zone_x1 = zone["x1"]
    zone_y1 = zone["y1"]
    zone_x2 = zone["x2"]
    zone_y2 = zone["y2"]

    return (
        zone_x1 <= x <= zone_x2
        and zone_y1 <= y <= zone_y2
    )


def filter_detections_inside_zone(detections, zone):
    zone_detections = []

    for detection in detections:
        foot_point = get_box_bottom_center(detection["bbox"])

        if is_point_inside_zone(foot_point, zone):
            zone_detections.append(detection)

    return zone_detections


def analyze_zones(detections, zones):
    zone_analytics = []

    for zone in zones:
        zone_detections = filter_detections_inside_zone(
            detections,
            zone
        )

        zone_count = len(zone_detections)

        tracked_ids = set()

        for detection in zone_detections:
            tracking_id = detection["tracking_id"]

            if tracking_id != "N/A":
                tracked_ids.add(tracking_id)

        tracked_count = len(tracked_ids)

        congestion_score = calculate_congestion_score(
            zone_count,
            zone_count,
            tracked_count
        )

        congestion_level = get_congestion_level(congestion_score)
        density_level = get_density_level(zone_count)

        zone_analytics.append({
            "name": zone["name"],
            "zone": zone,
            "count": zone_count,
            "tracked_count": tracked_count,
            "density": density_level,
            "congestion_score": congestion_score,
            "congestion_level": congestion_level,
            "detections": zone_detections
        })

    return zone_analytics


def get_most_dangerous_zone(zone_analytics):
    if not zone_analytics:
        return {
            "name": "None",
            "count": 0,
            "congestion_score": 0,
            "congestion_level": "Low"
        }

    return max(
        zone_analytics,
        key=lambda zone_data: zone_data["congestion_score"]
    )