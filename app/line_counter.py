class LineCrossingCounter:
    def __init__(self, line_y, cooldown_frames=20):
        self.line_y = line_y
        self.cooldown_frames = cooldown_frames

        self.previous_positions = {}
        self.last_crossing_frame = {}

        self.total_entries = 0
        self.total_exits = 0

        self.current_frame = 0

    def get_foot_y(self, bbox):
        x1, y1, x2, y2 = bbox
        return int(y2)

    def can_count_crossing(self, tracking_id):
        if tracking_id not in self.last_crossing_frame:
            return True

        frames_since_last_crossing = (
            self.current_frame - self.last_crossing_frame[tracking_id]
        )

        return frames_since_last_crossing >= self.cooldown_frames

    def update(self, detections):
        self.current_frame += 1

        for detection in detections:
            tracking_id = detection["tracking_id"]

            if tracking_id == "N/A":
                continue

            current_y = self.get_foot_y(detection["bbox"])

            if tracking_id in self.previous_positions:
                previous_y = self.previous_positions[tracking_id]

                crossed_down = previous_y < self.line_y <= current_y
                crossed_up = previous_y > self.line_y >= current_y

                if self.can_count_crossing(tracking_id):
                    if crossed_down:
                        self.total_entries += 1
                        self.last_crossing_frame[tracking_id] = self.current_frame

                    elif crossed_up:
                        self.total_exits += 1
                        self.last_crossing_frame[tracking_id] = self.current_frame

            self.previous_positions[tracking_id] = current_y

        return {
            "entries": self.total_entries,
            "exits": self.total_exits,
            "line_y": self.line_y
        }