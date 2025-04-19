import cv2
import mediapipe as mp
import time
import math
import pyttsx3
import numpy as np
import pyautogui

#############################################
# Setup text-to-speech engine (pyttsx3)
#############################################
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.8)

#############################################
# Morse Code Dictionary (Morse -> Letter)
#############################################
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9'
}


def decode_morse(morse_code):
    """
    Converts a Morse code string into plain text.

    • Extra spaces (for readability) are ignored.
    • A slash ("/") is interpreted as a word boundary.
      e.g.
         ".-   .--. .--. .-.. ." → "APPLE"
         ".- .--. .--. .-.. .  /  .. ... / .-. . -.." → "APPLE IS RED"
    """
    morse_code = morse_code.strip()
    word_tokens = morse_code.split("/")
    decoded_words = []
    for word in word_tokens:
        letter_tokens = word.split()  # drops extra whitespace
        decoded_word = ""
        for letter in letter_tokens:
            decoded_word += MORSE_CODE_DICT.get(letter, "")
        decoded_words.append(decoded_word)
    return " ".join(decoded_words)


#############################################
# Mediapipe Face Mesh Setup for Facial Landmarks
#############################################
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes (using Mediapipe face mesh)
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_EAR(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) using 6 eye landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    ear = (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2 * euclidean_distance(p1, p4))
    return ear


#############################################
# Timing and Threshold Parameters
#############################################
EAR_THRESHOLD = 0.25  # Eye considered closed if EAR is below this.
MIN_BLINK_DURATION = 0.5  # Left-eye blink must be at least 0.5 sec.
DOT_DASH_THRESHOLD = 1.5  # Left blink: <1.5 sec yields a dot; >=1.5 sec yields a dash.
MIN_RIGHT_BLINK_DURATION = 0.6  # Right blink must be held at least 0.6 sec to trigger deletion.
RIGHT_TTS_THRESHOLD = 1.0  # Right-eye hold 1.0-3.0 sec triggers TTS and typing.
RIGHT_MODE_SWITCH_THRESHOLD = 3.0  # Right-eye hold 3.0+ sec switches mode.
MIN_BOTH_BLINK_DURATION = 0.5  # Both-eye blink must be at least 0.5 sec.
LONG_BOTH_BLINK_THRESHOLD = 1.0  # Both-eye: 0.5-1.0 sec adds letter space; >=1.0 sec adds word boundary.

#############################################
# Global Variables
#############################################
morse_buffer = ""  # Accumulates dots, dashes, and spacing.
current_mode = "morse"  # Operating mode: "morse" (default) or "mouse".

# Blink state tracking variables.
left_blink_active = False
right_blink_active = False
both_blink_active = False
left_closed_start = 0
right_closed_start = 0
both_closed_start = 0


#############################################
# Function: Create the Morse Code Chart GUI image
#############################################
def get_chart_image(morse_buffer):
    chart_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(chart_img, "Morse Code Chart", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    letter_to_code = {}
    for code, letter in MORSE_CODE_DICT.items():
        if letter not in letter_to_code:
            letter_to_code[letter] = code
    sorted_keys = sorted(letter_to_code.keys(), key=lambda x: (not x.isdigit(), x))
    num_columns = 6
    num_rows = (len(sorted_keys) + num_columns - 1) // num_columns
    col_width = chart_img.shape[1] // num_columns
    row_height = 30
    for idx, letter in enumerate(sorted_keys):
        row = idx // num_columns
        col = idx % num_columns
        x = col * col_width + 10
        y = 50 + row * row_height
        text = f"{letter}: {letter_to_code[letter]}"
        cv2.putText(chart_img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    bottom_y = 50 + num_rows * row_height + 30
    cv2.putText(chart_img, "Live Morse Buffer:", (10, bottom_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(chart_img, morse_buffer, (10, bottom_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    live_translation = decode_morse(morse_buffer)
    cv2.putText(chart_img, "Live Translation:", (10, bottom_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
    cv2.putText(chart_img, live_translation, (10, bottom_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return chart_img


#############################################
# Main Loop
#############################################
cap = cv2.VideoCapture(0)
print("Starting the Eye-Morse system; press 'q' to exit.")
print("Modes: 'morse' for Morse input; 'mouse' for cursor control.")
print("Switch mode by holding the right eye for at least 3 seconds.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror frame and convert to RGB.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    # Defaults if no face is detected.
    left_EAR = 1.0
    right_EAR = 1.0
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        left_EAR = calculate_EAR(face_landmarks.landmark, left_eye_indices)
        right_EAR = calculate_EAR(face_landmarks.landmark, right_eye_indices)
        # Draw eye landmarks.
        for idx in left_eye_indices:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for idx in right_eye_indices:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Overlay EAR values.
    cv2.putText(frame, f"Left EAR: {left_EAR:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Right EAR: {right_EAR:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Mode: {current_mode.upper()}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Determine eye closure statuses.
    left_is_closed = (left_EAR < EAR_THRESHOLD)
    right_is_closed = (right_EAR < EAR_THRESHOLD)
    both_closed = left_is_closed and right_is_closed

    if current_mode == "morse":
        # Process both-eye blinks for spacing.
        if both_closed:
            if not both_blink_active:
                both_blink_active = True
                both_closed_start = current_time
        else:
            if both_blink_active:
                duration = current_time - both_closed_start
                if duration < MIN_BOTH_BLINK_DURATION:
                    print(f"Both-eye blink too short ({duration:.2f}s), ignored.")
                elif duration < LONG_BOTH_BLINK_THRESHOLD:
                    morse_buffer += " "
                    print("Letter space added. Buffer:", morse_buffer)
                else:
                    if not morse_buffer.endswith("/"):
                        morse_buffer += "/"
                    print("Word boundary (/) added. Buffer:", morse_buffer)
                both_blink_active = False

        # Process left-eye blinks for dot/dash.
        if left_is_closed:
            if not left_blink_active:
                left_blink_active = True
                left_closed_start = current_time
        else:
            if left_blink_active:
                duration = current_time - left_closed_start
                if duration >= MIN_BLINK_DURATION:
                    if duration < DOT_DASH_THRESHOLD:
                        morse_buffer += "."
                        print("Dot added. Buffer:", morse_buffer)
                    else:
                        morse_buffer += "-"
                        print("Dash added. Buffer:", morse_buffer)
                else:
                    print(f"Left blink too short ({duration:.2f}s), ignored.")
                left_blink_active = False

        # Process right-eye blinks ONLY if right eye is closed and left eye is open.
        if right_is_closed and not left_is_closed:
            if not right_blink_active:
                right_blink_active = True
                right_closed_start = current_time
        else:
            if right_blink_active:
                duration = current_time - right_closed_start
                if duration >= RIGHT_MODE_SWITCH_THRESHOLD:
                    current_mode = "mouse"
                    morse_buffer = ""  # clear buffer when switching modes
                    print("Switched to MOUSE mode.")
                elif duration >= RIGHT_TTS_THRESHOLD:
                    text_to_speak = decode_morse(morse_buffer)
                    print("TTS output triggered:", text_to_speak)
                    engine.say(text_to_speak)
                    engine.runAndWait()
                    time.sleep(1)  # Optional delay to allow target window focus.
                    pyautogui.write(text_to_speak + " ")
                elif duration >= MIN_RIGHT_BLINK_DURATION:
                    if len(morse_buffer) > 0:
                        morse_buffer = morse_buffer[:-1]
                        print("Deleted last symbol. Buffer:", morse_buffer)
                else:
                    print(f"Right blink too short ({duration:.2f}s), ignored.")
                right_blink_active = False

        live_translation = decode_morse(morse_buffer)
        cv2.putText(frame, f"Morse: {morse_buffer}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Translation: {live_translation}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif current_mode == "mouse":
        # In mouse mode, move the cursor using the average of left-eye landmarks.
        if results.multi_face_landmarks:
            pts = [face_landmarks.landmark[idx] for idx in left_eye_indices]
            avg_x = sum(pt.x for pt in pts) / len(pts)
            avg_y = sum(pt.y for pt in pts) / len(pts)
            screen_width, screen_height = pyautogui.size()
            cursor_x = int(avg_x * screen_width)
            cursor_y = int(avg_y * screen_height)
            pyautogui.moveTo(cursor_x, cursor_y)
        # In mouse mode, a left-eye blink simulates a left-click.
        if left_is_closed:
            if not left_blink_active:
                left_blink_active = True
                left_closed_start = current_time
        else:
            if left_blink_active:
                duration = current_time - left_closed_start
                if duration >= MIN_BLINK_DURATION:
                    print("Mouse left-click triggered.")
                    pyautogui.click()
                left_blink_active = False
        # In mouse mode, use right-eye blink (held ≥3 sec) to switch back to Morse mode.
        if right_is_closed:
            if not right_blink_active:
                right_blink_active = True
                right_closed_start = current_time
        else:
            if right_blink_active:
                duration = current_time - right_closed_start
                if duration >= RIGHT_MODE_SWITCH_THRESHOLD:
                    current_mode = "morse"
                    print("Switched to MORSE mode.")
                right_blink_active = False
        cv2.putText(frame, "Mouse mode: control cursor with your face; blink left to click.",
                    (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        live_translation = ""

    cv2.imshow("Eye-Morse System", frame)
    chart_img = get_chart_image(morse_buffer)
    cv2.imshow("Morse Code Chart", chart_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t") and current_mode == "morse":
        text_to_speak = decode_morse(morse_buffer)
        print("Keyboard TTS triggered:", text_to_speak)
        engine.say(text_to_speak)
        engine.runAndWait()
        pyautogui.write(text_to_speak + " ")

cap.release()
cv2.destroyAllWindows()
