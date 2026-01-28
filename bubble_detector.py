#!/usr/bin/env python3
import cv2
import numpy as np
import os
from collections import deque
from datetime import datetime
from pathlib import Path
import csv

# ===================== CONFIG =====================
class Config:
    # Camera
    CAMERA_INDEX = 0          # 0 = default webcam
    PLAYBACK_DELAY = 1
    ROI_SCALE = 0.6           # % of min(frame_w, frame_h)
    CIRCLE_MARGIN = 45

    # Thresholds
    THRESH_LOW_START = 14.5
    THRESH_MED_START = 18.5
    THRESH_HIGH_START = 24.5
    HYSTERESIS_BUFFER = 0.5

    # Optical Flow
    OF_UP_THRESHOLD = 1.0
    MOTION_SCALE = 3.0
    OF_SCALE = 0.5
    OF_PYR_SCALE = 0.5
    OF_LEVELS = 3
    OF_WINSIZE = 15
    OF_ITER = 3
    OF_POLY_N = 5
    OF_POLY_SIGMA = 1.2

    # Image Processing
    MOG_HISTORY = 200
    MOG_VARTHRESH = 16
    MOG_LEARNING_RATE = 0.05
    ADAPTIVE_BLOCK = 31
    ADAPTIVE_C = 3
    MORPH_KERNEL = (3, 3)
    MORPH_OPEN_ITER = 1
    MORPH_CLOSE_ITER = 1
    CLAHE_CLIP = 2.2
    CLAHE_TILE = (8, 8)
    CANNY_LOW = 50
    CANNY_HIGH = 150

    # Fusion
    WEIGHT_AREA = 0.65
    WEIGHT_MOTION = 0.35

    # Texture
    USE_LBP_TEXTURE = True
    LBP_INTERVAL = 5
    LBP_DOWNSCALE = 0.5

    # System
    SMOOTH_WINDOW = 15
    LOG_DIR = "logs"

# ===================== HELPERS =====================
class CategoryTracker:
    def __init__(self):
        self.current = "NO BUBBLE"

    def update(self, val):
        if val >= Config.THRESH_HIGH_START:
            self.current = "HIGH"
        elif val >= Config.THRESH_MED_START and self.current in ["LOW", "NO BUBBLE"]:
            self.current = "MEDIUM"
        elif val >= Config.THRESH_LOW_START and self.current == "NO BUBBLE":
            self.current = "LOW"

        if self.current == "HIGH" and val < Config.THRESH_HIGH_START - Config.HYSTERESIS_BUFFER:
            self.current = "MEDIUM"
        elif self.current == "MEDIUM" and val < Config.THRESH_MED_START - Config.HYSTERESIS_BUFFER:
            self.current = "LOW"
        elif self.current == "LOW" and val < Config.THRESH_LOW_START - Config.HYSTERESIS_BUFFER:
            self.current = "NO BUBBLE"

        return self.current


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def lbp_uniformity_fast(gray):
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,0),(0,1),(1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    for dy, dx in neighbors:
        n = np.roll(gray, (dy, dx), axis=(0,1))
        lbp |= ((n >= gray).astype(np.uint8) << 1)
    hist, _ = np.histogram(lbp.ravel(), bins=256)
    hist = hist / (hist.sum() + 1e-9)
    return float(hist[:32].sum())


def calculate_confidence(area, motion, edge, lbp):
    intensity = min((area + motion) / 40.0, 1.0)
    edge_score = min(edge * 10.0, 1.0)
    tex = 1.0 if abs(lbp - 0.4) < 0.2 else 0.5
    return min(100.0, (0.4 * intensity + 0.4 * edge_score + 0.2 * tex) * 100)

# ===================== MAIN =====================
def main():
    ensure_dir(Config.LOG_DIR)

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("No frames from camera")
        return

    fh, fw = frame0.shape[:2]

    roi_size = int(min(fw, fh) * Config.ROI_SCALE)
    x = (fw - roi_size) // 2
    y = (fh - roi_size) // 2
    w = h = roi_size

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = max(10, min(w, h)//2 - Config.CIRCLE_MARGIN)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    circle_area = np.count_nonzero(mask)

    clahe = cv2.createCLAHE(Config.CLAHE_CLIP, Config.CLAHE_TILE)
    mog2 = cv2.createBackgroundSubtractorMOG2(
        Config.MOG_HISTORY, Config.MOG_VARTHRESH, False
    )

    prev_small = None
    cat = CategoryTracker()
    intensity_hist = deque(maxlen=Config.SMOOTH_WINDOW)
    motion_hist = deque(maxlen=Config.SMOOTH_WINDOW)

    log_file = open(
        os.path.join(Config.LOG_DIR, f"log_{datetime.now().strftime('%H%M%S')}.csv"),
        "w", newline=""
    )
    csvw = csv.writer(log_file)
    csvw.writerow(["frame", "intensity", "motion", "category", "confidence"])

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        fg = mog2.apply(gray, Config.MOG_LEARNING_RATE)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        adapt = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, Config.ADAPTIVE_BLOCK, Config.ADAPTIVE_C
        )

        comb = cv2.bitwise_or(fg, adapt)
        if np.mean(comb) > 127:
            comb = cv2.bitwise_not(comb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL)
        comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel)

        masked = cv2.bitwise_and(comb, comb, mask=mask)
        area_intensity = cv2.countNonZero(masked) / circle_area * 100.0

        edges = cv2.Canny(blur, Config.CANNY_LOW, Config.CANNY_HIGH)
        edge_density = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)) / circle_area

        small = cv2.resize(blur, None, fx=Config.OF_SCALE, fy=Config.OF_SCALE)
        motion = 0.0

        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None,
                Config.OF_PYR_SCALE, Config.OF_LEVELS,
                Config.OF_WINSIZE, Config.OF_ITER,
                Config.OF_POLY_N, Config.OF_POLY_SIGMA, 0
            )
            vy = flow[...,1]
            smask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            up = np.logical_and(vy < -Config.OF_UP_THRESHOLD, smask > 0)
            motion = min(np.sum(up) / np.count_nonzero(smask) * 100 * Config.MOTION_SCALE, 100)

        prev_small = small.copy()
        motion_hist.append(motion)
        motion_smooth = np.mean(motion_hist)

        intensity = Config.WEIGHT_AREA * area_intensity + Config.WEIGHT_MOTION * motion_smooth
        intensity_hist.append(intensity)
        intensity_smooth = np.mean(intensity_hist)

        lbp = 0.4
        if Config.USE_LBP_TEXTURE and frame_id % Config.LBP_INTERVAL == 0:
            lbp = lbp_uniformity_fast(
                cv2.resize(blur, None, fx=Config.LBP_DOWNSCALE, fy=Config.LBP_DOWNSCALE)
            )

        category = cat.update(intensity_smooth)
        confidence = calculate_confidence(area_intensity, motion_smooth, edge_density, lbp)

        csvw.writerow([frame_id, round(intensity_smooth,2), round(motion_smooth,2), category, round(confidence,1)])

        status = "BUBBLE DETECTED" if category != "NO BUBBLE" else "NO BUBBLE"
        color = (0,255,0) if category != "NO BUBBLE" else (0,0,255)

        cv2.circle(frame, (x+cx, y+cy), radius, (255,255,0), 2)
        cv2.rectangle(frame, (0,0), (frame.shape[1],60), (0,0,0), -1)
        cv2.putText(frame, status, (30,45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("IRC Bubble Detection (Live)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
