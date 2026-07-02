import time
import math

import cv2
import numpy as np
from ultralytics import YOLO


class AnonymousIdentityManager:
    """
    Privacy-friendly person memory.

    It does not use face recognition.
    It only links short-term track IDs using:
    - tracker ID continuity from YOLO/ByteTrack
    - body box position
    - bounding-box overlap
    - simple clothing/appearance color histogram

    Result:
    - smoother stable IDs
    - fewer duplicate counts when a person disappears briefly and returns
    - no biometric identity storage
    """

    def __init__(
        self,
        memory_seconds=18,
        max_center_distance=260,
        min_match_score=0.52,
        min_color_score=0.18,
        iou_weight=0.20,
        distance_weight=0.35,
        color_weight=0.45,
        hist_bins=16
    ):
        self.memory_seconds = memory_seconds
        self.max_center_distance = max_center_distance
        self.min_match_score = min_match_score
        self.min_color_score = min_color_score
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.color_weight = color_weight
        self.hist_bins = hist_bins

        self.next_stable_id = 1
        self.raw_to_stable = {}
        self.tracks = {}

        self.total_reidentified = 0
        self.duplicates_prevented = 0
        self.total_created = 0
        self.last_frame_active_ids = set()

    def reset(self):
        self.next_stable_id = 1
        self.raw_to_stable = {}
        self.tracks = {}
        self.total_reidentified = 0
        self.duplicates_prevented = 0
        self.total_created = 0
        self.last_frame_active_ids = set()

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter_area / float(area_a + area_b - inter_area + 1e-6)

    def get_center_distance(self, box_a, box_b):
        ax, ay = self.get_center(box_a)
        bx, by = self.get_center(box_b)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def extract_appearance(self, frame, bbox):
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        # Focus more on upper/middle body for clothing colour.
        crop_h = person_crop.shape[0]
        upper_body = person_crop[: max(1, int(crop_h * 0.75)), :]

        try:
            hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv],
                [0, 1],
                None,
                [self.hist_bins, self.hist_bins],
                [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    def compare_appearance(self, hist_a, hist_b):
        if hist_a is None or hist_b is None:
            return 0.0

        try:
            distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
            similarity = 1.0 - float(distance)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def create_track(self, raw_id, bbox, appearance, now):
        stable_id = self.next_stable_id
        self.next_stable_id += 1

        self.tracks[stable_id] = {
            "stable_id": stable_id,
            "raw_ids": set([raw_id]) if raw_id is not None else set(),
            "bbox": bbox,
            "appearance": appearance,
            "first_seen": now,
            "last_seen": now,
            "seen_count": 1,
            "active": True
        }

        if raw_id is not None:
            self.raw_to_stable[raw_id] = stable_id

        self.total_created += 1
        return stable_id, "new"

    def update_track(self, stable_id, raw_id, bbox, appearance, now, state="active"):
        track = self.tracks.get(stable_id)

        if track is None:
            return

        if raw_id is not None:
            track["raw_ids"].add(raw_id)
            self.raw_to_stable[raw_id] = stable_id

        old_appearance = track.get("appearance")
        if old_appearance is not None and appearance is not None:
            try:
                appearance = (0.65 * old_appearance) + (0.35 * appearance)
            except Exception:
                pass

        track["bbox"] = bbox
        track["appearance"] = appearance if appearance is not None else old_appearance
        track["last_seen"] = now
        track["seen_count"] = track.get("seen_count", 0) + 1
        track["active"] = True
        track["state"] = state

    def find_best_memory_match(self, bbox, appearance, now, used_stable_ids):
        best_stable_id = None
        best_score = 0.0
        best_color_score = 0.0

        for stable_id, track in self.tracks.items():
            if stable_id in used_stable_ids:
                continue

            time_missing = now - track.get("last_seen", now)

            if time_missing > self.memory_seconds:
                continue

            previous_bbox = track.get("bbox")
            if previous_bbox is None:
                continue

            center_distance = self.get_center_distance(bbox, previous_bbox)

            if center_distance > self.max_center_distance:
                continue

            distance_score = 1.0 - min(1.0, center_distance / float(self.max_center_distance))
            iou_score = self.get_iou(bbox, previous_bbox)
            color_score = self.compare_appearance(appearance, track.get("appearance"))

            # If there is no overlap, require at least weak colour similarity.
            if iou_score < 0.05 and color_score < self.min_color_score:
                continue

            match_score = (
                (self.iou_weight * iou_score) +
                (self.distance_weight * distance_score) +
                (self.color_weight * color_score)
            )

            if match_score > best_score:
                best_score = match_score
                best_stable_id = stable_id
                best_color_score = color_score

        if best_stable_id is None:
            return None, 0.0, 0.0

        if best_score < self.min_match_score:
            return None, best_score, best_color_score

        return best_stable_id, best_score, best_color_score

    def assign(self, frame, detections):
        now = time.time()
        used_stable_ids = set()
        assigned_detections = []

        for detection in detections:
            raw_id = detection.get("raw_tracking_id")
            bbox = detection["bbox"]
            appearance = self.extract_appearance(frame, bbox)

            stable_id = None
            duplicate_state = "new"
            match_score = 0.0

            if raw_id is not None and raw_id in self.raw_to_stable:
                candidate_id = self.raw_to_stable[raw_id]

                if candidate_id not in used_stable_ids and candidate_id in self.tracks:
                    stable_id = candidate_id
                    duplicate_state = "tracked"
                    self.duplicates_prevented += 1

            if stable_id is None:
                matched_id, score, color_score = self.find_best_memory_match(
                    bbox,
                    appearance,
                    now,
                    used_stable_ids
                )

                if matched_id is not None:
                    stable_id = matched_id
                    match_score = score
                    duplicate_state = "reidentified"
                    self.total_reidentified += 1
                    self.duplicates_prevented += 1

            if stable_id is None:
                stable_id, duplicate_state = self.create_track(raw_id, bbox, appearance, now)
            else:
                self.update_track(stable_id, raw_id, bbox, appearance, now, duplicate_state)

            used_stable_ids.add(stable_id)

            detection["tracking_id"] = stable_id
            detection["stable_tracking_id"] = stable_id
            detection["duplicate_state"] = duplicate_state
            detection["reid_match_score"] = round(float(match_score), 3)
            detection["track_age_seconds"] = round(now - self.tracks[stable_id]["first_seen"], 1)

            assigned_detections.append(detection)

        for stable_id, track in self.tracks.items():
            track["active"] = stable_id in used_stable_ids

        self.last_frame_active_ids = used_stable_ids
        self.prune_old_tracks(now)

        return assigned_detections

    def prune_old_tracks(self, now):
        expired_ids = [
            stable_id
            for stable_id, track in self.tracks.items()
            if now - track.get("last_seen", now) > self.memory_seconds
        ]

        for stable_id in expired_ids:
            raw_ids = self.tracks[stable_id].get("raw_ids", set())

            for raw_id in raw_ids:
                if self.raw_to_stable.get(raw_id) == stable_id:
                    self.raw_to_stable.pop(raw_id, None)

            self.tracks.pop(stable_id, None)

    def get_summary(self):
        return {
            "session_unique_people": int(max(self.total_created, len(self.tracks))),
            "active_tracks": int(len(self.last_frame_active_ids)),
            "memory_tracks": int(len(self.tracks)),
            "reidentified_people": int(self.total_reidentified),
            "duplicates_prevented": int(self.duplicates_prevented),
            "privacy_mode": "Anonymous body tracking"
        }


class PeopleDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold,
        frame_width,
        tracker_type="bytetrack.yaml",
        enable_reid=True,
        reid_memory_seconds=18,
        reid_max_center_distance=260,
        reid_min_match_score=0.52,
        reid_min_color_score=0.18,
        reid_iou_weight=0.20,
        reid_distance_weight=0.35,
        reid_color_weight=0.45,
        reid_hist_bins=16
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_width = frame_width
        self.tracker_type = tracker_type
        self.enable_reid = enable_reid

        self.identity_manager = AnonymousIdentityManager(
            memory_seconds=reid_memory_seconds,
            max_center_distance=reid_max_center_distance,
            min_match_score=reid_min_match_score,
            min_color_score=reid_min_color_score,
            iou_weight=reid_iou_weight,
            distance_weight=reid_distance_weight,
            color_weight=reid_color_weight,
            hist_bins=reid_hist_bins
        )

    def reset_tracking(self):
        try:
            self.identity_manager.reset()
        except Exception:
            pass

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

                raw_tracking_id = None

                if box.id is not None:
                    raw_tracking_id = int(box.id[0])

                tracking_id = raw_tracking_id if raw_tracking_id is not None else "N/A"

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "tracking_id": tracking_id,
                    "raw_tracking_id": raw_tracking_id,
                    "stable_tracking_id": tracking_id,
                    "duplicate_state": "raw"
                })

        if self.enable_reid:
            detections = self.identity_manager.assign(frame, detections)

        return detections

    def get_tracking_summary(self):
        if not self.enable_reid:
            return {
                "session_unique_people": 0,
                "active_tracks": 0,
                "memory_tracks": 0,
                "reidentified_people": 0,
                "duplicates_prevented": 0,
                "privacy_mode": "YOLO tracking only"
            }

        return self.identity_manager.get_summary()