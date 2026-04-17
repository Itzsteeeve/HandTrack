# Hand Tracking Camera App (Python)

Simple real-time hand tracking app using webcam + MediaPipe.

## Features
- Uses your webcam in a normal desktop window.
- Detects multiple hands at once.
- Draws hand landmarks and connecting lines.
- Shows a basic gesture label (`fist`, `open`, or finger count).
- On first run, downloads the official MediaPipe hand landmarker model automatically.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

Press `q` to quit.
