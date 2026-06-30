import cv2
from ultralytics import YOLO


class PeopleDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold,
        frame_width,
        tracker_type="bytetrack.yaml"
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_width = frame_width
        self.tracker_type = tracker_type

    def resize_frame(self, frame):
        original_height, original_width = frame.shape[:2]

        if original_width <= self.frame_width:
            return frame, 1, 1

        scale = self.frame_width / original_width
        new_height = int(original_height * scale)

        resized_frame = cv2.resize(
            frame,
            (self.frame_width, new_height)
        )

        scale_x = original_width / self.frame_width
        scale_y = original_height / new_height

        return resized_frame, scale_x, scale_y

    def detect_and_track(self, frame):
        resized_frame, scale_x, scale_y = self.resize_frame(frame)

        results = self.model.track(
            resized_frame,
            persist=True,
            tracker=self.tracker_type,
            classes=[0],
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                confidence = float(box.conf[0])

                tracking_id = "N/A"

                if box.id is not None:
                    tracking_id = int(box.id[0])

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "tracking_id": tracking_id
                })

        return detections