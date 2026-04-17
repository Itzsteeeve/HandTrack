from pathlib import Path
from collections import Counter, defaultdict, deque
from math import hypot
import urllib.request

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# MediaPipe hand graph topology (21 points, 20 edges).
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


# Landmark indices for fingertips and finger joints.
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
INDEX_MCP = 5

SMOOTHING_WINDOW = 7


def distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def is_finger_extended(tip, pip, wrist, ratio=1.12):
    """Finger is considered up when tip is farther from wrist than PIP joint."""
    return distance(tip, wrist) > distance(pip, wrist) * ratio


def detect_finger_states(hand_landmarks):
    """Return dict with boolean state for thumb/index/middle/ring/pinky."""
    lm = hand_landmarks
    wrist = lm[0]

    states = {
        "index": is_finger_extended(lm[INDEX_TIP], lm[INDEX_PIP], wrist),
        "middle": is_finger_extended(lm[MIDDLE_TIP], lm[MIDDLE_PIP], wrist),
        "ring": is_finger_extended(lm[RING_TIP], lm[RING_PIP], wrist),
        "pinky": is_finger_extended(lm[PINKY_TIP], lm[PINKY_PIP], wrist),
    }

    # Thumb is evaluated against index MCP + wrist, which is more stable than x-only checks.
    thumb_tip = lm[THUMB_TIP]
    thumb_ip = lm[THUMB_IP]
    index_mcp = lm[INDEX_MCP]
    thumb_far_from_index = distance(thumb_tip, index_mcp) > distance(thumb_ip, index_mcp) * 1.10
    thumb_far_from_wrist = distance(thumb_tip, wrist) > distance(thumb_ip, wrist) * 1.05
    states["thumb"] = thumb_far_from_index and thumb_far_from_wrist

    return states


def ensure_model_exists():
    """Download Hand Landmarker model on first run."""
    if MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading hand tracking model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download MediaPipe model. Check internet connection and try again."
        ) from exc


def draw_hand(frame, hand_landmarks):
    """Draw landmarks and connections with OpenCV."""
    h, w, _ = frame.shape
    points = []

    for lm in hand_landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        points.append((x_px, y_px))
        cv2.circle(frame, (x_px, y_px), 4, (40, 220, 255), -1, cv2.LINE_AA)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (80, 200, 80), 2, cv2.LINE_AA)


def classify_gesture(hand_landmarks, handedness_label):
    """Return (gesture_name, raised_fingers_count, states_dict)."""
    del handedness_label  # Not needed with orientation-agnostic logic.

    lm = hand_landmarks
    states = detect_finger_states(lm)
    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]
    raised_count = sum(states.values())

    thumb_to_index_tip = distance(lm[THUMB_TIP], lm[INDEX_TIP])
    palm_scale = max(distance(lm[0], lm[INDEX_MCP]), 1e-6)

    if raised_count == 0:
        return "fist", raised_count, states
    if thumb and not any([index, middle, ring, pinky]):
        if lm[THUMB_TIP].y < lm[0].y:
            return "thumbs_up", raised_count, states
        return "thumb", raised_count, states
    if index and middle and not any([thumb, ring, pinky]):
        return "peace", raised_count, states
    if index and not any([thumb, middle, ring, pinky]):
        return "point", raised_count, states
    if thumb and index and not any([middle, ring, pinky]) and thumb_to_index_tip < palm_scale * 0.45:
        return "ok", raised_count, states
    if raised_count == 5:
        return "open", raised_count, states

    return f"{raised_count}_fingers", raised_count, states


def majority_value(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def main():
    ensure_model_exists()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        frame_idx = 0
        finger_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
        gesture_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Could not read frame from camera.")
                break

            # Flip image for a mirror-like webcam experience.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_count = 0
            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)

                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    draw_hand(frame, hand_landmarks)

                    label = "Unknown"
                    if results.handedness and i < len(results.handedness) and results.handedness[i]:
                        label = results.handedness[i][0].category_name

                    gesture, finger_count, _states = classify_gesture(hand_landmarks, label)
                    finger_history[i].append(finger_count)
                    gesture_history[i].append(gesture)

                    smooth_fingers = majority_value(finger_history[i])
                    smooth_gesture = majority_value(gesture_history[i])

                    h, w, _ = frame.shape
                    wrist = hand_landmarks[0]
                    x_px = max(10, int(wrist.x * w))
                    y_px = max(20, int(wrist.y * h) - 20)
                    cv2.putText(
                        frame,
                        f"{label} | fingers: {smooth_fingers} | {smooth_gesture}",
                        (x_px, y_px),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"Hands: {hand_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (50, 220, 50),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
