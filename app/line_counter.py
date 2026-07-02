class LineCrossingCounter:
    """
    v38.1 duplicate-safe line counter.

    This counter uses the stable anonymous tracking_id from detector.py.
    It prevents one person from repeatedly increasing entry/exit counts
    while moving around, jittering near the line, or briefly disappearing.

    It still allows real movement patterns:

        entry -> exit -> entry again

    It does not use face recognition.
    """

    def __init__(self, line_y, cooldown_frames=30, line_buffer=14, lost_ttl_frames=300):
        self.line_y = int(line_y)
        self.cooldown_frames = int(cooldown_frames)
        self.line_buffer = int(line_buffer)
        self.lost_ttl_frames = int(lost_ttl_frames)

        self.previous_positions = {}
        self.previous_sides = {}
        self.last_crossing_frame = {}
        self.last_duplicate_block_frame = {}
        self.last_seen_frame = {}

        self.last_action_by_id = {}

        self.total_entries = 0
        self.total_exits = 0

        self.counted_entry_ids = set()
        self.counted_exit_ids = set()

        self.duplicate_crossings_blocked = 0
        self.current_frame = 0

    def get_foot_y(self, bbox):
        x1, y1, x2, y2 = bbox
        return int(y2)

    def get_side(self, foot_y):
        if foot_y < self.line_y - self.line_buffer:
            return "above"

        if foot_y > self.line_y + self.line_buffer:
            return "below"

        return "on_line"

    def can_count_crossing(self, tracking_id):
        if tracking_id not in self.last_crossing_frame:
            return True

        frames_since_last_crossing = (
            self.current_frame - self.last_crossing_frame[tracking_id]
        )

        return frames_since_last_crossing >= self.cooldown_frames

    def register_duplicate_block(self, tracking_id):
        """
        Count duplicate blocking only once per cooldown window.

        The previous version could increase duplicate_crossings_blocked too often
        when a person jittered near the line. This version only records a real
        blocked duplicate attempt after the cooldown window.
        """
        last_blocked = self.last_duplicate_block_frame.get(tracking_id)

        if last_blocked is None:
            self.duplicate_crossings_blocked += 1
            self.last_duplicate_block_frame[tracking_id] = self.current_frame
            return

        frames_since_last_blocked = self.current_frame - last_blocked

        if frames_since_last_blocked >= self.cooldown_frames:
            self.duplicate_crossings_blocked += 1
            self.last_duplicate_block_frame[tracking_id] = self.current_frame

    def prune_old_tracks(self):
        expired_ids = [
            tracking_id
            for tracking_id, last_seen in self.last_seen_frame.items()
            if self.current_frame - last_seen > self.lost_ttl_frames
        ]

        for tracking_id in expired_ids:
            self.previous_positions.pop(tracking_id, None)
            self.previous_sides.pop(tracking_id, None)
            self.last_seen_frame.pop(tracking_id, None)
            self.last_crossing_frame.pop(tracking_id, None)
            self.last_duplicate_block_frame.pop(tracking_id, None)

            # Do not remove last_action_by_id immediately.
            # Keeping this memory prevents duplicate counting when the same
            # anonymous ID is recovered later in the same monitoring session.

    def update(self, detections):
        self.current_frame += 1
        active_ids = set()

        for detection in detections:
            tracking_id = detection.get(
                "stable_tracking_id",
                detection.get("tracking_id", "N/A")
            )

            if tracking_id == "N/A" or tracking_id is None:
                continue

            active_ids.add(tracking_id)
            self.last_seen_frame[tracking_id] = self.current_frame

            current_y = self.get_foot_y(detection["bbox"])
            current_side = self.get_side(current_y)

            previous_y = self.previous_positions.get(tracking_id)
            previous_side = self.previous_sides.get(tracking_id)

            if previous_y is not None and previous_side is not None:
                crossed_down = previous_side == "above" and current_side == "below"
                crossed_up = previous_side == "below" and current_side == "above"

                if crossed_down:
                    if not self.can_count_crossing(tracking_id):
                        self.register_duplicate_block(tracking_id)

                    elif self.last_action_by_id.get(tracking_id) == "entry":
                        self.register_duplicate_block(tracking_id)

                    else:
                        self.total_entries += 1
                        self.counted_entry_ids.add(tracking_id)
                        self.last_action_by_id[tracking_id] = "entry"
                        self.last_crossing_frame[tracking_id] = self.current_frame

                elif crossed_up:
                    if not self.can_count_crossing(tracking_id):
                        self.register_duplicate_block(tracking_id)

                    elif self.last_action_by_id.get(tracking_id) == "exit":
                        self.register_duplicate_block(tracking_id)

                    else:
                        self.total_exits += 1
                        self.counted_exit_ids.add(tracking_id)
                        self.last_action_by_id[tracking_id] = "exit"
                        self.last_crossing_frame[tracking_id] = self.current_frame

            self.previous_positions[tracking_id] = current_y

            # Do not replace a strong side with "on_line".
            # This reduces false crossing caused by jitter around the line.
            if current_side != "on_line":
                self.previous_sides[tracking_id] = current_side

        self.prune_old_tracks()

        return {
            "entries": self.total_entries,
            "exits": self.total_exits,
            "line_y": self.line_y,
            "active_line_ids": len(active_ids),
            "unique_entry_ids": len(self.counted_entry_ids),
            "unique_exit_ids": len(self.counted_exit_ids),
            "duplicate_crossings_blocked": self.duplicate_crossings_blocked
        }