import os
import cv2
import time
import threading
import numpy as np
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from classifier import classify_hand

# ----------------- Text-to-Speech Engine ----------------- #
class SpeechWorker:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 160)
        self._engine.setProperty('volume', 0.9)
        self._queue = []
        self._lock = threading.Lock()
        self._is_running = True
        self._thread = threading.Thread(target=self._process_speech, daemon=True)
        self._thread.start()

    def speak(self, text):
        with self._lock:
            # avoid queuing duplicates
            if not self._queue or self._queue[-1] != text:
                self._queue.append(text)

    def _process_speech(self):
        while self._is_running:
            phrase = None
            with self._lock:
                if self._queue:
                    phrase = self._queue.pop(0)
            if phrase:
                try:
                    self._engine.say(phrase)
                    self._engine.runAndWait()
                except Exception as e:
                    pass
            else:
                time.sleep(0.05)

    def stop(self):
        self._is_running = False

# ----------------- Visual Connections & Drawing ----------------- #
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def draw_hand_skeleton(frame, landmarks, color=(0, 255, 128)):
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    # Draw connection bones
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, coords[p1], coords[p2], (40, 40, 40), 4, cv2.LINE_AA)
        cv2.line(frame, coords[p1], coords[p2], color, 2, cv2.LINE_AA)

    # Draw joint points
    for i, pt in enumerate(coords):
        is_tip = i in [4, 8, 12, 16, 20]
        radius = 6 if is_tip else 4
        pt_color = (0, 120, 255) if is_tip else (255, 255, 255)
        cv2.circle(frame, pt, radius, pt_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

# ----------------- Main Real-Time Loop ----------------- #
def main():
    print("=" * 60)
    print("  AI Sign Language & Gesture Interpreter (Laptop Webcam)  ")
    print("=" * 60)
    print("  Controls:")
    print("  [SPACE]     : Add currently recognized letter/word to sentence")
    print("  [BACKSPACE] : Delete last word")
    print("  [S]         : Speak full sentence aloud")
    print("  [C]         : Clear sentence")
    print("  [Q]         : Quit")
    print("=" * 60)

    # Initialize Speech Worker
    speaker = SpeechWorker()

    # Load MediaPipe Hand Landmarker Task
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hand_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open laptop webcam. Please ensure camera access is enabled.")
        return

    # Frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State variables
    sentence_history = []
    active_sign = "..."
    confidence = 0.0
    stable_count = 0
    last_spoken_sign = ""
    last_frame_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Flip frame horizontally for natural selfie-mirror viewing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, current_time - last_frame_time))
        last_frame_time = current_time

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Hand Detection
        results = detector.detect(mp_image)

        detected_label = "..."
        detected_conf = 0.0

        if results.hand_landmarks:
            for hand_idx, landmarks in enumerate(results.hand_landmarks):
                # Pick sign color (Neon Cyan / Lime)
                hand_color = (255, 215, 0) if hand_idx == 0 else (0, 255, 128)
                draw_hand_skeleton(frame, landmarks, color=hand_color)

                # Classify single hand
                label, conf = classify_hand(landmarks)
                if label != "..." and conf > detected_conf:
                    detected_label = label
                    detected_conf = conf

                # Display bounding box label
                x_coords = [int(lm.x * w) for lm in landmarks]
                y_coords = [int(lm.y * h) for lm in landmarks]
                min_x, max_x = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                min_y, max_y = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

                # Draw subtle bounding box
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), hand_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{label}", (min_x, max(30, min_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"{label}", (min_x, max(30, min_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2, cv2.LINE_AA)

        # Gesture Stability Smoothing
        if detected_label == active_sign:
            stable_count += 1
        else:
            active_sign = detected_label
            confidence = detected_conf
            stable_count = 1

        # Auto-announce gesture if held steadily for ~15 frames (~0.5s)
        if stable_count == 15 and active_sign not in ["...", "UNKNOWN"]:
            if active_sign != last_spoken_sign:
                speaker.speak(active_sign.split(" ")[0])
                last_spoken_sign = active_sign

        # ----------------- UI / HUD Overlay ----------------- #
        # Top banner with translucent glassmorphic background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 25), -1)
        # Bottom banner for sentence output
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top Banner Text
        cv2.putText(frame, "SIGN:", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        sign_color = (0, 255, 128) if active_sign != "..." else (120, 120, 120)
        cv2.putText(frame, f"{active_sign}", (100, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, sign_color, 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Hold gesture to auto-speak", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

        # Key Controls Help (top right)
        cv2.putText(frame, "[SPACE] Add | [S] Speak | [C] Clear | [Q] Quit", (w - 480, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom Sentence Builder
        sentence_str = " ".join(sentence_history)
        if not sentence_str:
            sentence_str = "(Sentence will appear here... Press [SPACE] to commit gesture)"
            text_color = (130, 130, 130)
        else:
            text_color = (255, 255, 255)

        cv2.putText(frame, "SENTENCE:", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, sentence_str[:60], (150, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2, cv2.LINE_AA)

        # Window title constant
        win_name = "AI Sign Language Interpreter - Laptop Camera"
        cv2.imshow(win_name, frame)

        # Check if user clicked window [X] close button
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Keyboard Interactivity (handles both focused window and key codes)
        key = cv2.waitKey(1)
        if key != -1:
            key_code = key & 0xFF
            # Quit on 'q', 'Q', or ESC (27)
            if key_code in [ord('q'), ord('Q'), 27]:
                break
            elif key_code == 32:  # SPACE: commit current sign
                if active_sign not in ["...", "UNKNOWN"]:
                    clean_word = active_sign.split(" ")[0]
                    sentence_history.append(clean_word)
                    speaker.speak(clean_word)
            elif key_code == 8:  # BACKSPACE: remove last word
                if sentence_history:
                    sentence_history.pop()
            elif key_code in [ord('s'), ord('S')]:  # S: speak entire sentence
                if sentence_history:
                    full_text = " ".join(sentence_history)
                    speaker.speak(full_text)
            elif key_code in [ord('c'), ord('C')]:  # C: clear sentence
                sentence_history.clear()

    # Fast cleanup
    cap.release()
    cv2.destroyAllWindows()
    try:
        speaker.stop()
    except Exception:
        pass
    print("Interpreter closed safely.")
    os._exit(0)

if __name__ == "__main__":
    main()
