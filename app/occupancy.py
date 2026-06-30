class OccupancyTracker:
    def __init__(self, missing_tolerance=8):
        self.current_ids = set()
        self.missing_counts = {}

        self.current_occupancy = 0
        self.peak_occupancy = 0

        self.missing_tolerance = missing_tolerance

    def update(self, zone_detections):
        visible_ids = set()

        for detection in zone_detections:
            tracking_id = detection["tracking_id"]

            if tracking_id == "N/A":
                continue

            visible_ids.add(tracking_id)

        for tracking_id in visible_ids:
            self.current_ids.add(tracking_id)
            self.missing_counts[tracking_id] = 0

        for tracking_id in list(self.current_ids):
            if tracking_id not in visible_ids:
                self.missing_counts[tracking_id] = (
                    self.missing_counts.get(tracking_id, 0) + 1
                )

                if self.missing_counts[tracking_id] > self.missing_tolerance:
                    self.current_ids.remove(tracking_id)
                    self.missing_counts.pop(tracking_id, None)

        self.current_occupancy = len(self.current_ids)

        if self.current_occupancy > self.peak_occupancy:
            self.peak_occupancy = self.current_occupancy

        return {
            "occupancy": self.current_occupancy,
            "peak": self.peak_occupancy
        }