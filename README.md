# video_race_analysis

Computer vision-based swimming race analysis tool for extracting stroke metrics from fixed-camera video recordings.

## Overview

This tool uses OpenCV and MediaPipe Pose Estimation to automatically track swimmer hand positions in race video, then calculates stroke rate, stroke length, and swimming velocity. It is designed for researchers analyzing technique from poolside camera footage.

## Features

- **Video Calibration**: Interactive 4-point calibration to establish pixel-to-meter scale for the pool.
- **Pose Tracking**: Automatic wrist position tracking using MediaPipe Pose.
- **Stroke Detection**: Signal processing (peak detection via SciPy) to identify stroke entry points.
- **Metric Calculation**: Computes stroke rate (strokes/min), stroke length (m), cycle time (s), and average velocity (m/s).

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- SciPy

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python video_rece_analysis.py <video_file_path>
# Example:
python video_rece_analysis.py race_50m_free.mp4
```

## Output

- `stroke_analysis.png`: Plot of hand Y-position over time with detected stroke entry points marked.
