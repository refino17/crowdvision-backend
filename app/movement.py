class MovementAnalyzer:
    def __init__(self, movement_threshold=5, history_size=6):
        self.position_history = {}
        self.movement_threshold = movement_threshold
        self.history_size = history_size

    def get_center_point(self, bbox):
        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    def analyze(self, detections):
        direction_counts = {
            "left": 0,
            "right": 0,
            "up": 0,
            "down": 0,
            "stationary": 0
        }

        active_ids = set()

        for detection in detections:
            tracking_id = detection["tracking_id"]

            if tracking_id == "N/A":
                continue

            active_ids.add(tracking_id)

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

        for tracking_id in list(self.position_history.keys()):
            if tracking_id not in active_ids:
                self.position_history.pop(tracking_id, None)

        return direction_counts