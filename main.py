from pathlib import Path
from collections import Counter, defaultdict, deque
from math import acos, degrees, hypot
import threading
import time
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
THUMB_CMC = 1

SMOOTHING_WINDOW = 7
FPS_WINDOW = 24

HAND_THEMES = {
    "Left": {
        "point": (245, 170, 60),
        "line": (220, 130, 40),
        "text": (255, 225, 170),
    },
    "Right": {
        "point": (70, 230, 110),
        "line": (60, 185, 95),
        "text": (180, 255, 190),
    },
    "Unknown": {
        "point": (40, 220, 255),
        "line": (80, 200, 80),
        "text": (0, 255, 255),
    },
}

GESTURE_COLORS = {
    "thumbs_up": (0, 215, 255),
    "thumbs_down": (0, 120, 255),
    "rock": (235, 100, 255),
    "spock": (255, 210, 80),
    "heart": (120, 120, 255),
    "l_sign": (255, 180, 0),
    "ok": (255, 230, 60),
}


class AsyncCamera:
    """Background camera reader to keep only the newest frame and reduce latency."""

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.running = False
        self.latest_frame = None
        self.thread = None

    def is_opened(self):
        return self.cap.isOpened()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.2)
        self.cap.release()


def distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def vector(a, b):
    return (b.x - a.x, b.y - a.y)


def angle_between(v1, v2):
    norm = hypot(v1[0], v1[1]) * hypot(v2[0], v2[1])
    if norm <= 1e-9:
        return 0.0
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / norm
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return degrees(acos(cos_theta))


def get_palm_scale(lm):
    return max(distance(lm[0], lm[INDEX_MCP]), 1e-6)


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


def draw_hand(frame, hand_landmarks, handedness_label, gesture):
    """Draw landmarks and connections with OpenCV."""
    h, w, _ = frame.shape
    points = []

    theme = HAND_THEMES.get(handedness_label, HAND_THEMES["Unknown"])
    point_color = theme["point"]
    line_color = theme["line"]
    radius = 4
    thickness = 2

    if gesture in GESTURE_COLORS:
        point_color = GESTURE_COLORS[gesture]
        line_color = GESTURE_COLORS[gesture]
        radius = 6
        thickness = 3

    for lm in hand_landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        points.append((x_px, y_px))
        cv2.circle(frame, (x_px, y_px), radius, point_color, -1, cv2.LINE_AA)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], line_color, thickness, cv2.LINE_AA)


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
    palm_scale = get_palm_scale(lm)

    if raised_count == 0:
        return "fist", raised_count, states

    thumb_only = thumb and not any([index, middle, ring, pinky])
    if thumb_only:
        if lm[THUMB_TIP].y < lm[0].y - palm_scale * 0.15:
            return "thumbs_up", raised_count, states
        if lm[THUMB_TIP].y > lm[0].y + palm_scale * 0.15:
            return "thumbs_down", raised_count, states
        return "thumb", raised_count, states

    if index and pinky and not middle and not ring:
        return "rock", raised_count, states

    if thumb and index and not any([middle, ring, pinky]):
        index_vec = vector(lm[INDEX_MCP], lm[INDEX_TIP])
        thumb_vec = vector(lm[THUMB_IP], lm[THUMB_TIP])
        l_angle = angle_between(index_vec, thumb_vec)
        if 55 <= l_angle <= 125 and thumb_to_index_tip > palm_scale * 0.85:
            return "l_sign", raised_count, states
        if thumb_to_index_tip < palm_scale * 0.45:
            return "ok", raised_count, states

    if index and middle and not any([thumb, ring, pinky]):
        return "peace", raised_count, states

    if index and not any([thumb, middle, ring, pinky]):
        return "point", raised_count, states

    if index and middle and ring and pinky:
        gap_im = distance(lm[INDEX_TIP], lm[MIDDLE_TIP])
        gap_mr = distance(lm[MIDDLE_TIP], lm[RING_TIP])
        gap_rp = distance(lm[RING_TIP], lm[PINKY_TIP])
        if gap_mr > max(gap_im, gap_rp) * 1.35:
            return "spock", raised_count, states

    if raised_count == 5:
        return "open", raised_count, states

    return f"{raised_count}_fingers", raised_count, states


def detect_heart_two_hands(hand_a, hand_b, states_a, states_b):
    """Detect a basic two-hand heart gesture using thumb/index proximity across hands."""
    if not (states_a["thumb"] and states_a["index"] and states_b["thumb"] and states_b["index"]):
        return False

    scale = (get_palm_scale(hand_a) + get_palm_scale(hand_b)) / 2.0
    index_dist = distance(hand_a[INDEX_TIP], hand_b[INDEX_TIP])
    thumb_dist = distance(hand_a[THUMB_TIP], hand_b[THUMB_TIP])
    cross_dist_1 = distance(hand_a[INDEX_TIP], hand_b[THUMB_TIP])
    cross_dist_2 = distance(hand_b[INDEX_TIP], hand_a[THUMB_TIP])
    wrist_dist = distance(hand_a[0], hand_b[0])

    return (
        index_dist < scale * 1.10
        and thumb_dist < scale * 1.10
        and cross_dist_1 < scale * 1.40
        and cross_dist_2 < scale * 1.40
        and wrist_dist < scale * 4.0
    )


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

    cap = AsyncCamera(0)

    if not cap.is_opened():
        print("Error: Could not open camera.")
        return

    cap.start()

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        start_time = time.perf_counter()
        last_timestamp_ms = 0
        finger_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
        gesture_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
        frame_times = deque(maxlen=FPS_WINDOW)

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.002)
                continue

            # Flip image for a mirror-like webcam experience.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_times.append(time.perf_counter())

            proc_fps = 0.0
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 1e-6:
                    proc_fps = (len(frame_times) - 1) / elapsed

            hand_count = 0
            smoothed_data = {}
            labels = {}
            states_for_hand = {}
            heart_detected = False

            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)

                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    label = "Unknown"
                    if results.handedness and i < len(results.handedness) and results.handedness[i]:
                        label = results.handedness[i][0].category_name

                    gesture, finger_count, states = classify_gesture(hand_landmarks, label)
                    finger_history[i].append(finger_count)
                    gesture_history[i].append(gesture)

                    smooth_fingers = majority_value(finger_history[i])
                    smooth_gesture = majority_value(gesture_history[i])

                    smoothed_data[i] = (smooth_fingers, smooth_gesture)
                    labels[i] = label
                    states_for_hand[i] = states

                if hand_count >= 2:
                    for a in range(hand_count):
                        for b in range(a + 1, hand_count):
                            if detect_heart_two_hands(
                                results.hand_landmarks[a],
                                results.hand_landmarks[b],
                                states_for_hand[a],
                                states_for_hand[b],
                            ):
                                heart_detected = True
                                gesture_history[a].append("heart")
                                gesture_history[b].append("heart")
                                smooth_a = majority_value(gesture_history[a])
                                smooth_b = majority_value(gesture_history[b])
                                smoothed_data[a] = (smoothed_data[a][0], smooth_a)
                                smoothed_data[b] = (smoothed_data[b][0], smooth_b)

                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    smooth_fingers, smooth_gesture = smoothed_data.get(i, (0, "unknown"))
                    label = labels.get(i, "Unknown")

                    draw_hand(frame, hand_landmarks, label, smooth_gesture)

                    h, w, _ = frame.shape
                    wrist = hand_landmarks[0]
                    x_px = max(10, int(wrist.x * w))
                    y_px = max(20, int(wrist.y * h) - 20)
                    text_color = HAND_THEMES.get(label, HAND_THEMES["Unknown"])["text"]
                    if smooth_gesture in GESTURE_COLORS:
                        text_color = GESTURE_COLORS[smooth_gesture]

                    cv2.putText(
                        frame,
                        f"{label} | fingers: {smooth_fingers} | {smooth_gesture}",
                        (x_px, y_px),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        text_color,
                        2,
                        cv2.LINE_AA,
                    )

            if heart_detected:
                h, w, _ = frame.shape
                cv2.putText(
                    frame,
                    "HEART DETECTED",
                    (max(10, w // 2 - 120), max(40, h - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    GESTURE_COLORS["heart"],
                    3,
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

            cv2.putText(
                frame,
                f"FPS: {proc_fps:.1f}",
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (90, 230, 230),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
