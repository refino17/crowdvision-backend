import cv2
import os
from datetime import datetime

ALERT_FOLDER = "data/alerts"


def save_alert_snapshot(frame, density_level, prefix=None):
    """
    Save alert/evidence frame atomically so the web dashboard never reads a half-written image.
    """
    os.makedirs(ALERT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_level = str(density_level or "alert").lower().replace(" ", "_")
    clean_prefix = str(prefix or clean_level).lower().replace(" ", "_")

    filename = os.path.join(
        ALERT_FOLDER,
        f"{clean_prefix}_{timestamp}.jpg"
    )
    temp_filename = filename.replace(".jpg", "_tmp.jpg")

    try:
        if frame is None or frame.size == 0:
            return None

        cv2.imwrite(temp_filename, frame)
        os.replace(temp_filename, filename)

        return filename
    except Exception as error:
        print(f"Evidence snapshot save error: {error}")

        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception:
            pass

        return None