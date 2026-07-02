class MovementAnalyzer:
    def __init__(self, movement_threshold=5, history_size=6, lost_ttl_frames=90):
        self.position_history = {}
        self.last_seen_frame = {}
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        self.lost_ttl_frames = lost_ttl_frames
        self.current_frame = 0

    def get_center_point(self, bbox):
        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    def prune_old_tracks(self):
        expired_ids = [
            tracking_id
            for tracking_id, last_seen in self.last_seen_frame.items()
            if self.current_frame - last_seen > self.lost_ttl_frames
        ]

        for tracking_id in expired_ids:
            self.position_history.pop(tracking_id, None)
            self.last_seen_frame.pop(tracking_id, None)

    def analyze(self, detections):
        self.current_frame += 1

        direction_counts = {
            "left": 0,
            "right": 0,
            "up": 0,
            "down": 0,
            "stationary": 0
        }

        for detection in detections:
            tracking_id = detection.get("stable_tracking_id", detection.get("tracking_id", "N/A"))

            if tracking_id == "N/A" or tracking_id is None:
                continue

            self.last_seen_frame[tracking_id] = self.current_frame

            current_position = self.get_center_point(detection["bbox"])

            if tracking_id not in self.position_history:
                self.position_history[tracking_id] = []

            self.position_history[tracking_id].append(current_position)

            if len(self.position_history[tracking_id]) > self.history_size:
                self.position_history[tracking_id].pop(0)

            history = self.position_history[tracking_id]

            if len(history) < 2:
                direction_counts["stationary"] += 1
                continue

            start_x, start_y = history[0]
            end_x, end_y = history[-1]

            dx = end_x - start_x
            dy = end_y - start_y

            if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
                direction_counts["stationary"] += 1
            elif abs(dx) >= abs(dy):
                if dx > 0:
                    direction_counts["right"] += 1
                else:
                    direction_counts["left"] += 1
            else:
                if dy > 0:
                    direction_counts["down"] += 1
                else:
                    direction_counts["up"] += 1

        self.prune_old_tracks()
        return direction_counts