import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ========== CONFIGURATION ==========
# Modes: 0 = Cursor, 1 = Scroll, 2 = Volume, 3 = Multiselect
mode_names = ["Cursor", "Scroll", "Volume", "Multiselect"]
current_mode = 0
mode_baseline = None  # For vertical baseline in Scroll and Volume mode

# Timing thresholds (seconds)
MODE_TOGGLE_TIME = 1.5       # Hold both eyes closed to switch mode
CLICK_DEBOUNCE_DELAY = 0.5   # Debounce delay to prevent multiple clicks

# Blink detection threshold values (normalized vertical difference)
# (Left: diff = landmark145.y - landmark159.y; Right: diff = landmark374.y - landmark386.y)
# Adjust these if your measured values differ (try higher values if detection lags).
left_diff_threshold = 0.01
right_diff_threshold = 0.01

# Smoothing parameters
ALPHA = 0.3   # For blink detection smoothing.
SMOOTHING_FACTOR_CURSOR = 0.7  # For smoothing cursor movement

# ========== GLOBAL STATE ==========
last_left_click_time = 0
last_right_click_time = 0
both_eyes_closed_start = None

smoothed_left_diff = None
smoothed_right_diff = None

cursor_x, cursor_y = 0, 0
first_frame = True
prev_time = time.time()

# ========== SETUP MEDIAPIPE & WEBCAM ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                   static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.7,
                                   min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
screen_w, screen_h = pyautogui.size()

# Create full-screen window once.
cv2.namedWindow("Virtual Mouse", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame and convert to RGB.
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    # Display current mode text.
    cv2.putText(frame, f"Mode: {mode_names[current_mode]}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ----- Compute Raw Blink Differences -----
        # Left eye: landmark145 (bottom) and landmark159 (top)
        raw_left_diff = landmarks[145].y - landmarks[159].y
        # Right eye: landmark374 (bottom) and landmark386 (top)
        raw_right_diff = landmarks[374].y - landmarks[386].y

        # Apply exponential smoothing.
        if smoothed_left_diff is None:
            smoothed_left_diff = raw_left_diff
            smoothed_right_diff = raw_right_diff
        else:
            smoothed_left_diff = ALPHA * raw_left_diff + (1 - ALPHA) * smoothed_left_diff
            smoothed_right_diff = ALPHA * raw_right_diff + (1 - ALPHA) * smoothed_right_diff

        # Debug: Display blink differences.
        cv2.putText(frame, f"L_diff: {smoothed_left_diff:.4f}", (10, frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"R_diff: {smoothed_right_diff:.4f}", (10, frame_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Determine if eyes are closed.
        left_closed = smoothed_left_diff < left_diff_threshold
        right_closed = smoothed_right_diff < right_diff_threshold

        # ----- Mode Toggling -----
        if left_closed and right_closed:
            if both_eyes_closed_start is None:
                both_eyes_closed_start = current_time
            elif current_time - both_eyes_closed_start >= MODE_TOGGLE_TIME:
                current_mode = (current_mode + 1) % len(mode_names)
                mode_baseline = None
                both_eyes_closed_start = None
                last_left_click_time = 0
                last_right_click_time = 0
        else:
            both_eyes_closed_start = None

        # ----- Compute Iris Center for Cursor Movement -----
        left_iris = landmarks[468]   # Left iris center
        right_iris = landmarks[473]  # Right iris center
        iris_center_x = (left_iris.x + right_iris.x) / 2
        iris_center_y = (left_iris.y + right_iris.y) / 2

        # ----- Draw Dots around the Eyes (for visual feedback) -----
        # Left eye (blue)
        for idx in [33, 7, 163, 144, 145, 159, 160, 133]:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        # Right eye (red)
        for idx in [362, 382, 381, 380, 374, 385, 386, 263]:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        # Draw averaged iris center (green)
        cv2.circle(frame, (int(iris_center_x * frame_w), int(iris_center_y * frame_h)), 4, (0, 255, 0), -1)

        # ----- Mode-Specific Behavior -----
        if current_mode in [0, 3]:  # Cursor & Multiselect (Multiselect holds SHIFT)
            new_x = int(iris_center_x * screen_w)
            new_y = int(iris_center_y * screen_h)
            if first_frame:
                cursor_x, cursor_y = new_x, new_y
                first_frame = False
            else:
                cursor_x = int(SMOOTHING_FACTOR_CURSOR * cursor_x + (1 - SMOOTHING_FACTOR_CURSOR) * new_x)
                cursor_y = int(SMOOTHING_FACTOR_CURSOR * cursor_y + (1 - SMOOTHING_FACTOR_CURSOR) * new_y)
            pyautogui.moveTo(cursor_x, cursor_y)
            if current_mode == 3:
                pyautogui.keyDown("shift")
            else:
                pyautogui.keyUp("shift")

            # ----- Left Click: Trigger if Left Eye is closed and Right is open -----
            if left_closed and not right_closed and (current_time - last_left_click_time > CLICK_DEBOUNCE_DELAY):
                pyautogui.click(button="left")
                last_left_click_time = current_time
                cv2.putText(frame, "Left Click", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----- Right Click: Trigger if Right Eye is closed and Left is open -----
            if right_closed and not left_closed and (current_time - last_right_click_time > CLICK_DEBOUNCE_DELAY):
                pyautogui.click(button="right")
                last_right_click_time = current_time
                cv2.putText(frame, "Right Click", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif current_mode == 1:  # Scroll Mode
            if mode_baseline is None:
                mode_baseline = iris_center_y
            displacement = iris_center_y - mode_baseline
            scroll_val = int(displacement * 500)  # Scale as needed.
            if abs(scroll_val) > 5:
                pyautogui.scroll(-scroll_val)
            cv2.putText(frame, f"Scroll: {scroll_val}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        elif current_mode == 2:  # Volume Mode
            if mode_baseline is None:
                mode_baseline = iris_center_y
            displacement = iris_center_y - mode_baseline
            if abs(displacement) > 0.02:
                if displacement > 0:
                    pyautogui.press("volumedown")
                else:
                    pyautogui.press("volumeup")
            cv2.putText(frame, "Volume Control", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)

    # ----- FPS Calculation -----
    new_time = time.time()
    fps = int(1 / (new_time - prev_time)) if (new_time - prev_time) > 0 else 0
    prev_time = new_time
    cv2.putText(frame, f"FPS: {fps}", (frame_w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame in the pre-created full-screen window.
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
