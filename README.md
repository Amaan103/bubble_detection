IRC 2026 ABEx Bubble Detector
Real-time bubble detection system for ABEx mission fluid analysis.

Features
4-Tier bubble classification: NO BUBBLE / LOW / MEDIUM / HIGH

Hysteresis tracking prevents false state changes

Multi-sensor fusion: Area + Motion + Edge density + LBP texture

Real-time logging to CSV for analysis



Installation
bash

# Clone repository
cd ~
git clone <your-repo-url>
cd irc-bubble-detector

# Install dependencies
pip3 install opencv-python==4.8.1.78 numpy

# Test camera
python3 bubble_detector.py --test-camera
Running the System
bash
# Live detection (USB webcam)
python3 bubble_detector.py

# Headless mode (no display)
python3 bubble_detector.py --headless

# Recorded video analysis
python3 bubble_detector.py --video test_vial.mp4

# Press 'q' to quit
Detection Categories
Category	Intensity Range	Mission Status
NO BUBBLE	<17.5%	Stable sample
LOW	17.5-22.5%	Minor activity
MEDIUM	22.5-27.5%	Active reaction
HIGH	>27.5%	Strong evolution
Adjustable Parameters
bash
# Edit config.py or use CLI flags
python3 bubble_detector.py \
    --roi-scale 0.6 \
    --thresh-low 17.5 \
    --thresh-med 22.5 \
    --thresh-high 27.5
Key Parameters:

THRESH_*: 15-35 (bubble intensity thresholds)

MOTION_SCALE: 2-5 (optical flow sensitivity)

Output Data
CSV Logs (logs/log_XXXXXX.csv):

text
frame,intensity,motion,category,confidence
1,15.2,2.1,NO BUBBLE,45.3
2,18.7,3.4,LOW,67.8
System Architecture
text
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   USB Cam    │───▶│   Processing    │───▶│   Logging    │
│  (640x480)   │    │  Pipeline      │    │  (CSV)       │
└──────────────┘    │                 │    └──────────────┘
                     │ • CLAHE         │
                     │ • MOG2 BG       │
                     │ • Adaptive Thresh│
                     │ • Optical Flow  │
                     │ • Edge Density  │
                     └─────────────────┘
Jetson Nano Optimization
bash
# Camera settings (already in code)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# Expected: 15-25 FPS @ 640x480 ROI
Troubleshooting
Issue	Solution
Low FPS	sudo jetson_clocks, ROI_SCALE=0.4
No camera	ls /dev/video*, check USB connection
False positives	Increase HYSTERESIS_BUFFER=1.0
No detection	Verify lighting, reduce CIRCLE_MARGIN=30
File Structure
text
irc-bubble-detector/
├── bubble_detector.py     # Main detection script
├── config.py             # All parameters
├── logs/                 # CSV output (gitignored)
└── README.md            # This file
Development Workflow
bash
# 1. Record test data
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('test.mp4', ...)
"

# 2. Analyze performance
python3 bubble_detector.py --video test.mp4 --log-only

# 3. Tune thresholds using CSV logs
Team Tasks
CV: Threshold tuning from lab tests

