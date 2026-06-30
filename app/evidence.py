import cv2
import os
from datetime import datetime

ALERT_FOLDER = "data/alerts"

def save_alert_snapshot(frame, density_level):
    os.makedirs(ALERT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{ALERT_FOLDER}/"
        f"{density_level.lower()}_{timestamp}.jpg"
    )

    cv2.imwrite(filename, frame)

    return filename