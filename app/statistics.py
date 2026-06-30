import time


class SessionStats:
    def __init__(self):
        self.session_start_time = time.time()

        self.previous_display_frame_time = time.time()

        self.previous_ai_frame_time = time.time()
        self.ai_fps = 0

        self.total_alert_snapshots = 0
        self.processed_ai_frames = 0
        self.skipped_frames = 0

    def calculate_display_fps(self):
        current_time = time.time()

        time_difference = current_time - self.previous_display_frame_time

        if time_difference <= 0:
            return 0

        display_fps = 1 / time_difference

        self.previous_display_frame_time = current_time

        return display_fps

    def update_ai_fps(self):
        current_time = time.time()

        time_difference = current_time - self.previous_ai_frame_time

        if time_difference <= 0:
            self.ai_fps = 0
        else:
            self.ai_fps = 1 / time_difference

        self.previous_ai_frame_time = current_time
        self.processed_ai_frames += 1

        return self.ai_fps

    def get_ai_fps(self):
        return self.ai_fps

    def get_runtime_seconds(self):
        return int(time.time() - self.session_start_time)

    def add_alert_snapshot(self):
        self.total_alert_snapshots += 1

    def add_skipped_frame(self):
        self.skipped_frames += 1