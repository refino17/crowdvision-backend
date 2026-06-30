import cv2
import numpy as np


class CrowdHeatmap:
    def __init__(
        self,
        decay_rate=0.985,
        radius=18,
        opacity=0.18,
        blur_size=35
    ):
        self.heatmap = None
        self.decay_rate = decay_rate
        self.radius = radius
        self.opacity = opacity
        self.blur_size = self._make_odd(blur_size)

    def _make_odd(self, value):
        if value % 2 == 0:
            return value + 1
        return value

    def update(self, frame, detections):
        height, width = frame.shape[:2]

        if self.heatmap is None:
            self.heatmap = np.zeros((height, width), dtype=np.float32)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            if 0 <= foot_x < width and 0 <= foot_y < height:
                cv2.circle(
                    self.heatmap,
                    (foot_x, foot_y),
                    self.radius,
                    1,
                    -1
                )

        self.heatmap *= self.decay_rate

    def draw(self, frame):
        if self.heatmap is None:
            return frame

        blurred = cv2.GaussianBlur(
            self.heatmap,
            (self.blur_size, self.blur_size),
            0
        )

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        colored_heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

        mask = normalized > 15

        output = frame.copy()

        blended = cv2.addWeighted(
            frame,
            1 - self.opacity,
            colored_heatmap,
            self.opacity,
            0
        )

        output[mask] = blended[mask]

        return output