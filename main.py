#!/usr/bin/env python3
import cv2
import numpy as np
import os
from collections import deque
from datetime import datetime
from pathlib import Path
import csv

class Config:
    # Input Settings
    # give video_path = 0 for using camera feed
    VIDEO_PATH = r"C:\Users\" 
    PLAYBACK_DELAY = 30
    CIRCLE_MARGIN = 45 

    # Thresholds
    THRESH_LOW_START = 15.5
    THRESH_MED_START = 20.5   
    THRESH_HIGH_START = 25.5  
    HYSTERESIS_BUFFER = 0.5

    # Optical Flow & Motion
    OF_UP_THRESHOLD = 1.0   
    MOTION_SCALE = 3.0      
    OF_SCALE = 0.5
    OF_PYR_SCALE = 0.5
    OF_LEVELS = 3
    OF_WINSIZE = 15
    OF_ITER = 3
    OF_POLY_N = 5
    OF_POLY_SIGMA = 1.2

    # Filtering & weighting
    MOG_HISTORY = 200
    MOG_VARTHRESH = 16
    MOG_DETECT_SHADOWS = False
    MOG_LEARNING_RATE = 0.05 
    ADAPTIVE_BLOCK = 31
    ADAPTIVE_C = 3
    MORPH_KERNEL = (3,3)
    MORPH_OPEN_ITER = 1
    MORPH_CLOSE_ITER = 1
    CLAHE_CLIP = 2.2
    CLAHE_TILE = (8,8)
    CANNY_LOW = 50      
    CANNY_HIGH = 150
    
    WEIGHT_AREA = 0.65      
    WEIGHT_MOTION = 0.35    
    
    # Texture Analysis
    LBP_DOWNSCALE = 0.5
    LBP_INTERVAL = 5
    USE_LBP_TEXTURE = True 
    
    # System
    SMOOTH_WINDOW = 15 
    MAX_DISPLAY_W = 1200
    MAX_DISPLAY_H = 800
    LOG_DIR = "logs"
    LOG_INTERVAL_FRAMES = 1
    FLUSH_INTERVAL = 30 

class CategoryTracker:
    def __init__(self):
        self.current_cat = "NO BUBBLE"
    
    def update(self, intensity):
        # State upgrade logic
        if intensity >= Config.THRESH_HIGH_START:
            self.current_cat = "HIGH"
        elif intensity >= Config.THRESH_MED_START and self.current_cat in ["LOW", "NO BUBBLE"]:
            self.current_cat = "MEDIUM"
        elif intensity >= Config.THRESH_LOW_START and self.current_cat == "NO BUBBLE":
            self.current_cat = "LOW"
            
        # Hysteresis downgrade logic
        if self.current_cat == "HIGH":
            if intensity < (Config.THRESH_HIGH_START - Config.HYSTERESIS_BUFFER):
                self.current_cat = "MEDIUM"
        elif self.current_cat == "MEDIUM":
            if intensity < (Config.THRESH_MED_START - Config.HYSTERESIS_BUFFER):
                self.current_cat = "LOW"
        elif self.current_cat == "LOW":
            if intensity < (Config.THRESH_LOW_START - Config.HYSTERESIS_BUFFER):
                self.current_cat = "NO BUBBLE"
                
        return self.current_cat

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def resize_for_display(img, max_w=Config.MAX_DISPLAY_W, max_h=Config.MAX_DISPLAY_H):
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale
    return img, 1.0

def lbp_uniformity_fast(gray_small):
    lbp = np.zeros_like(gray_small, dtype=np.uint8)
    neighbors = [(-1,0),(0,1),(1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    for dy, dx in neighbors:
        neighbor = np.roll(gray_small, (dy, dx), axis=(0,1))
        lbp |= ((neighbor >= gray_small).astype(np.uint8) << 1)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    hist = hist / (hist.sum() + 1e-9)
    return float(hist[:32].sum())

def calculate_confidence(area_intensity, motion_intensity, edge_density, lbp_uniformity):
    avg_intensity = (area_intensity + motion_intensity) / 2.0
    intensity_score = min(avg_intensity / 20.0, 1.0) 
    edge_contribution = min(edge_density * 10.0, 1.0) 
    texture_contribution = 1.0
    if Config.USE_LBP_TEXTURE:
        dist = abs(lbp_uniformity - 0.4)
        if dist < 0.2: texture_contribution = 1.0
        else: texture_contribution = max(0.0, 1.0 - (dist - 0.2) * 2.0)
    confidence = (0.4 * intensity_score + 0.4 * edge_contribution + 0.2 * texture_contribution) * 100.0
    return min(100.0, max(0.0, confidence))

def main():
    ensure_dir(Config.LOG_DIR)
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    ret, frame0 = cap.read()
    if not ret: return

    # Initial ROI selection
    disp0, scale0 = resize_for_display(frame0)
    cv2.imshow("Select ROI", disp0)
    cv2.waitKey(1)
    roi_disp = cv2.selectROI("Select ROI", disp0, False, True)
    cv2.destroyWindow("Select ROI")
    
    x_d, y_d, w_d, h_d = roi_disp
    if w_d == 0: return
    x, y = int(x_d / scale0), int(y_d / scale0)
    w, h = int(w_d / scale0), int(h_d / scale0)
    fh, fw = frame0.shape[:2]
    x, y = max(0, min(x, fw-1)), max(0, min(y, fh-1))
    w, h = max(1, min(w, fw-x)), max(1, min(h, fh-y))

    # Mask setup
    c_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = max(10, min(w, h) // 2 - Config.CIRCLE_MARGIN)
    cv2.circle(c_mask, (cx, cy), radius, 255, -1)
    circle_area = np.count_nonzero(c_mask)
    print(f"Analysis Radius: {radius} px")

    # Algorithm initialization
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_TILE)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=Config.MOG_HISTORY, varThreshold=Config.MOG_VARTHRESH, detectShadows=False)

    intensity_hist = deque(maxlen=Config.SMOOTH_WINDOW)
    motion_hist = deque(maxlen=Config.SMOOTH_WINDOW)
    prev_small_for_of = None
    cat_tracker = CategoryTracker()
    
    logpath = os.path.join(Config.LOG_DIR, f"log_{datetime.now().strftime('%H%M%S')}.csv")
    log_file = open(logpath, "w", newline="")
    csvw = csv.writer(log_file)
    csvw.writerow(["frame","intensity","motion","category","confidence"])

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        roi_full = frame[y:y+h, x:x+w]
        if roi_full.size == 0: continue

        # Preprocessing
        gray = cv2.cvtColor(roi_full, cv2.COLOR_BGR2GRAY)
        gray_clahe = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_clahe, (5,5), 0)

        # Background subtraction & Adaptive Thresholding
        fg = mog2.apply(gray_clahe, learningRate=Config.MOG_LEARNING_RATE)
        _, fg_bin = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Config.ADAPTIVE_BLOCK, Config.ADAPTIVE_C)
        
        combined = cv2.bitwise_or(fg_bin, adapt)
        if np.mean(combined) > 127: combined = cv2.bitwise_not(combined)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=Config.MORPH_OPEN_ITER)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=Config.MORPH_CLOSE_ITER)
        
        masked_bubbles = cv2.bitwise_and(combined, combined, mask=c_mask)
        area_intensity = (cv2.countNonZero(masked_bubbles) / float(circle_area)) * 100.0

        # Edge detection
        edges = cv2.Canny(blur, Config.CANNY_LOW, Config.CANNY_HIGH)
        edge_density = float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=c_mask))) / float(circle_area)

        # Optical Flow for Motion
        small_blur = cv2.resize(blur, None, fx=Config.OF_SCALE, fy=Config.OF_SCALE)
        motion_up_frac = 0.0
        if prev_small_for_of is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small_for_of, small_blur, None, Config.OF_PYR_SCALE, Config.OF_LEVELS, Config.OF_WINSIZE, Config.OF_ITER, Config.OF_POLY_N, Config.OF_POLY_SIGMA, 0)
            vy = flow[..., 1]
            small_mask = cv2.resize(c_mask, (small_blur.shape[1], small_blur.shape[0]), interpolation=cv2.INTER_NEAREST)
            if small_mask.any():
                up_votes = np.logical_and(vy < -Config.OF_UP_THRESHOLD, small_mask > 0)
                motion_up_frac = float(np.sum(up_votes)) / float(np.count_nonzero(small_mask))
        prev_small_for_of = small_blur.copy()

        motion_intensity = min(motion_up_frac * 100 * Config.MOTION_SCALE, 100.0)
        motion_hist.append(motion_intensity)
        motion_smooth = float(np.mean(motion_hist)) if motion_hist else 0.0

        # Sensor Fusion
        intensity_raw = Config.WEIGHT_AREA * area_intensity + Config.WEIGHT_MOTION * motion_smooth
        intensity_hist.append(intensity_raw)
        intensity_smooth = float(np.mean(intensity_hist))

        # Texture check
        lbp_value = 0.4
        if Config.USE_LBP_TEXTURE and frame_id % Config.LBP_INTERVAL == 0:
            lbp_value = lbp_uniformity_fast(cv2.resize(blur, None, fx=Config.LBP_DOWNSCALE, fy=Config.LBP_DOWNSCALE))

        # Update Tracker
        category = cat_tracker.update(intensity_smooth)
        confidence = calculate_confidence(area_intensity, motion_smooth, edge_density, lbp_value)

        # Logging
        if frame_id % Config.LOG_INTERVAL_FRAMES == 0:
            csvw.writerow([frame_id, round(intensity_smooth, 2), round(motion_smooth, 2), category, round(confidence, 1)])
            if frame_id % Config.FLUSH_INTERVAL == 0: log_file.flush()

        # Display Logic
        annotated = frame.copy()
        cv2.circle(annotated, (x + cx, y + cy), radius, (0,255,0), 2)
        disp_img, _ = resize_for_display(annotated)
        dh, dw = disp_img.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Binary Output for Competition
        is_life_detected = (category == "LOW" or category == "MEDIUM" or category == "HIGH")
        
        status_color = (0, 255, 0) if is_life_detected else (0, 0, 255)
        status_text = "BUBBLE DETECTED" if is_life_detected else "NO BUBBLE DETECTED"
            
        # Draw Status Bar
        cv2.rectangle(disp_img, (0, 0), (dw, 60), (0,0,0), -1)
        text_size = cv2.getTextSize(status_text, font, 1.2, 2)[0]
        text_x = (dw - text_size[0]) // 2
        cv2.putText(disp_img, status_text, (text_x, 40), font, 1.2, status_color, 2, cv2.LINE_AA)

        # Debug Stats
        cv2.putText(disp_img, f"Intensity: {intensity_smooth:.1f}%", (20, 100), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp_img, f"Raw Cat: {category}", (20, 135), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
        
        cv2.imshow("Bubble Monitor", disp_img)

        if frame_id % 30 == 0:
            print(f"[Frame {frame_id}] {status_text} | Val={intensity_smooth:.1f} | Conf={confidence:.0f}%")

        if cv2.waitKey(Config.PLAYBACK_DELAY) & 0xFF == ord('q'): break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
