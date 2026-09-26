# video_race_analysis

Computer-vision demo for extracting swimming-related movement timing from fixed-camera video.

## What this project demonstrates

- OpenCV-based video calibration
- MediaPipe pose estimation
- Conversion of normalized landmarks to pixel coordinates
- Time-accurate tracking using the video frame rate
- Peak detection for repeated wrist-movement cycles
- Visual quality control of detected cycles

## Important scope note

The horizontal wrist-displacement value is an exploratory trajectory metric. It should not be interpreted as a validated estimate of whole-body swimming stroke length without additional kinematic validation.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python video_rece_analysis.py race_50m_free.mp4 25

The optional second argument is the real-world distance, in metres, between the two calibration points clicked in the first video frame. It defaults to 25 m.

## Workflow

1. Click two image points with a known real-world separation.
2. Track the right wrist with MediaPipe in pixel coordinates.
3. Preserve actual video timestamps from the frame rate.
4. Detect repeated wrist-position cycles.
5. Save a diagnostic trajectory figure.

## Output

- stroke_analysis.png: right-wrist trajectory with detected cycle locations
- Console output: cycle rate, mean cycle duration, and an exploratory horizontal wrist-displacement metric
