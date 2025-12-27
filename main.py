import cv2
import numpy as np
import os
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# --- Config & Tunables ---
class Config:
    # TODO: change this to arg parser later
    VIDEO_PATH = r"C:\Users\amaan\Downloads\ac1deb1a7451841508001ef957d0d4d6.jpg"
    
    # Thresholds (v9 calibration)
    THRESH_LOW = 24.0 
    THRESH_MED = 29.5 
    THRESH_HIGH = 34.5 
    
    # Hysteresis Buffer: prevents flickering when value hovers on line
    # Only downgrade if value drops this much below threshold
    BUFFER = 0.5

    # Visuals
    PLAYBACK_DELAY = 30 
    MAX_W = 1200
    MAX_H = 800
    CIRCLE_MARGIN = 45 # shrink analysis circle to avoid pipe walls

    # Processing
    MOG_HISTORY = 200
    MOG_THRESH = 16
    CLAHE_CLIP = 2.2
    
    # Motion
    OF_SCALE = 0.5 # downscale for speed
    OF_UP_THRESH = 1.0 # min upward velocity
    MOTION_WEIGHT = 3.0 # motion impact multiplier
    
    # Final Weights
    W_AREA = 0.65
    W_MOTION = 0.35
    
    LOG_DIR = "logs"

# Tracks state to avoid flip-flopping categories
class StateTracker:
    def __init__(self):
        self.state = "NO BUBBLE"
    
    def update(self, val):
        # 1. Upgrades (hard limits)
        if val >= Config.THRESH_HIGH:
            self.state = "HIGH"
        elif val >= Config.THRESH_MED and self.state in ["LOW", "NO BUBBLE"]:
            self.state = "MEDIUM"
        elif val >= Config.THRESH_LOW and self.state == "NO BUBBLE":
            self.state = "LOW"
            
        # 2. Downgrades (limit - buffer)
        # only drop if we are surely below the line
        if self.state == "HIGH":
            if val < (Config.THRESH_HIGH - Config.BUFFER):
                self.state = "MEDIUM"
        elif self.state == "MEDIUM":
            if val < (Config.THRESH_MED - Config.BUFFER):
                self.state = "LOW"
        elif self.state == "LOW":
            if val < (Config.THRESH_LOW - Config.BUFFER):
                self.state = "NO BUBBLE"
                
        return self.state

# resize for laptop screen
def smart_resize(img, max_w=Config.MAX_W, max_h=Config.MAX_H):
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale
    return img, 1.0

# texture check - bubbles are rougher than fluid
def get_texture_score(gray):
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,0),(0,1),(1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    for dy, dx in neighbors:
        neighbor = np.roll(gray, (dy, dx), axis=(0,1))
        lbp |= ((neighbor >= gray).astype(np.uint8) << 1)
    
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    hist = hist / (hist.sum() + 1e-9)
    return float(hist[:32].sum())

def get_confidence(area, motion, edges, texture):
    # heuristic confidence score
    avg_int = (area + motion) / 2.0
    s_int = min(avg_int / 20.0, 1.0)
    s_edge = min(edges * 10.0, 1.0)
    
    # bubbles usually have texture ~0.4
    s_tex = 1.0
    dist = abs(texture - 0.4)
    if dist >= 0.2:
        s_tex = max(0.0, 1.0 - (dist - 0.2) * 2.0)
        
    conf = (0.4*s_int + 0.4*s_edge + 0.2*s_tex) * 100.0
    return min(100.0, max(0.0, conf))

def main():
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print("Video load failed.")
        return

    ret, frame0 = cap.read()
    if not ret: return

    # ROI Selection
    disp0, s0 = smart_resize(frame0)
    cv2.imshow("Select Pipe (Enter)", disp0)
    x_d, y_d, w_d, h_d = cv2.selectROI("Select Pipe (Enter)", disp0, False, True)
    cv2.destroyAllWindows()
    
    if w_d == 0: return 
    
    # map back to full res
    x, y = int(x_d / s0), int(y_d / s0)
    w, h = int(w_d / s0), int(h_d / s0)
    
    # circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rad = max(10, min(w, h) // 2 - Config.CIRCLE_MARGIN)
    cv2.circle(mask, (cx, cy), rad, 255, -1)
    total_pixels = np.count_nonzero(mask)
    
    print(f"Radius: {rad}px")

    # init algos
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=(8,8))
    mog2 = cv2.createBackgroundSubtractorMOG2(history=Config.MOG_HISTORY, varThreshold=Config.MOG_THRESH, detectShadows=False)

    hist_int = deque(maxlen=15)
    hist_mot = deque(maxlen=15)
    tracker = StateTracker()
    prev_small = None
    
    # logs
    t_str = datetime.now().strftime('%H%M%S')
    f_log = open(f"{Config.LOG_DIR}/log_{t_str}.csv", "w", newline="")
    writer = csv.writer(f_log)
    writer.writerow(["frame", "intensity", "motion", "category", "conf"])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: continue

        # 1. Pre-process
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_c = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_c, (5,5), 0)

        # 2. Hybrid Detection
        fg = mog2.apply(gray_c, learningRate=0.05)
        _, fg_b = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        
        adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
        
        combined = cv2.bitwise_or(fg_b, adapt)
        # flip if mostly white
        if np.mean(combined) > 127: 
            combined = cv2.bitwise_not(combined)

        # noise clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        final = cv2.bitwise_and(combined, combined, mask=mask)
        
        # 3. Metrics
        area_pct = (cv2.countNonZero(final) / float(total_pixels)) * 100.0

        e = cv2.Canny(blur, 50, 150)
        e_dens = float(np.count_nonzero(cv2.bitwise_and(e, e, mask=mask))) / float(total_pixels)

        # 4. Optical Flow (small)
        curr_small = cv2.resize(blur, None, fx=Config.OF_SCALE, fy=Config.OF_SCALE)
        mot_up = 0.0
        
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vy = flow[..., 1]
            
            s_mask = cv2.resize(mask, (curr_small.shape[1], curr_small.shape[0]), interpolation=cv2.INTER_NEAREST)
            if s_mask.any():
                # negative vy = upward motion
                moving = np.logical_and(vy < -Config.OF_UP_THRESH, s_mask > 0)
                mot_up = float(np.sum(moving)) / float(np.count_nonzero(s_mask))
                
        prev_small = curr_small.copy()
        
        mot_pct = min(mot_up * 100 * Config.MOTION_WEIGHT, 100.0)
        hist_mot.append(mot_pct)
        mot_smooth = float(np.mean(hist_mot)) if hist_mot else 0.0

        # 5. Result
        raw = (Config.W_AREA * area_pct) + (Config.W_MOTION * mot_smooth)
        hist_int.append(raw)
        val_smooth = float(np.mean(hist_int))
        
        # texture check (every 5 frames to save cpu)
        tex_val = 0.4
        if frame_idx % 5 == 0:
            tex_val = get_texture_score(cv2.resize(blur, None, fx=0.5, fy=0.5))

        cat = tracker.update(val_smooth)
        conf = get_confidence(area_pct, mot_smooth, e_dens, tex_val)

        writer.writerow([frame_idx, round(val_smooth, 2), round(mot_smooth, 2), cat, round(conf, 1)])
        if frame_idx % 30 == 0: f_log.flush()

        # 6. Draw
        annotated = frame.copy()
        cv2.circle(annotated, (x + cx, y + cy), rad, (0, 255, 0), 2)
        
        disp, _ = smart_resize(annotated)
        
        c_state = (200, 200, 200)
        if cat == "HIGH": c_state = (0, 0, 255)
        elif cat == "MEDIUM": c_state = (0, 165, 255)
        elif cat == "LOW": c_state = (0, 255, 255)

        cv2.putText(disp, f"Int: {val_smooth:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(disp, f"St: {cat}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c_state, 2)
        cv2.putText(disp, f"Cf: {conf:.0f}%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Monitor", disp)
        cv2.imshow("Mask", smart_resize(final)[0])

        if frame_idx % 30 == 0:
            print(f"[{frame_idx}] {val_smooth:.1f}% | {cat}")

        if cv2.waitKey(Config.PLAYBACK_DELAY) & 0xFF == ord('q'):
            break

    f_log.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
