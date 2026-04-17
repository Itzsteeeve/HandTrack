# Hand Tracking Camera App (Python)

Simple real-time hand tracking app using webcam + MediaPipe.

## Features
- Uses your webcam in a normal desktop window.
- Detects multiple hands at once.
- Draws hand landmarks and connecting lines.
- Shows finger count and gesture labels in real time.
- Supports gestures: `thumbs_up`, `thumbs_down`, `rock`, `spock`, `heart` (two hands), `l_sign`, `ok`, `peace`, `point`, `fist`, `open`.
- Uses different colors for left and right hand overlays.
- Highlights recognized gestures with brighter colors.
- Uses background camera reading for lower latency and overlays current processing FPS.
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
