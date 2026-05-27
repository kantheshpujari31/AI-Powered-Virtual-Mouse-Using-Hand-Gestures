import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Variables
click_time = 0
prev_action = None
click_count = 0

# Smoothing buffer
smooth_buffer = deque(maxlen=5)

def calc_distance(p1, p2):
    return math.dist(p1, p2)

def smooth_cursor(x, y):
    smooth_buffer.append((x, y))
    avg_x = sum([p[0] for p in smooth_buffer]) / len(smooth_buffer)
    avg_y = sum([p[1] for p in smooth_buffer]) / len(smooth_buffer)
    return int(avg_x), int(avg_y)

def fingers_up(landmarks):
    """ Returns [thumb, index, middle, ring, pinky] (1 = finger up) """
    finger_states = []

    # Thumb
    finger_states.append(1 if landmarks[4].x < landmarks[3].x else 0)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        finger_states.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

    return finger_states

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            # Key finger tips
            thumb = (landmarks[4].x, landmarks[4].y)
            index = (landmarks[8].x, landmarks[8].y)
            middle = (landmarks[12].x, landmarks[12].y)
            ring = (landmarks[16].x, landmarks[16].y)
            pinky = (landmarks[20].x, landmarks[20].y)

            finger_state = fingers_up(landmarks)

            # Cursor movement
            screen_x = int(index[0] * screen_width)
            screen_y = int(index[1] * screen_height)
            smooth_x, smooth_y = smooth_cursor(screen_x, screen_y)
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)

            # Distances
            dist_thumb_index = calc_distance(thumb, index)
            dist_thumb_middle = calc_distance(thumb, middle)
            dist_thumb_ring = calc_distance(thumb, ring)
            dist_thumb_pinky = calc_distance(thumb, pinky)
            dist_index_middle = calc_distance(index, middle)

            current_time = time.time()

            # --- Left / Double Click ---
            if dist_thumb_index < 0.04:
                if current_time - click_time < 0.4:
                    click_count += 1
                else:
                    click_count = 1

                click_time = current_time

                if click_count == 2:
                    pyautogui.doubleClick()
                    prev_action = "Double Click"
                    click_count = 0
                else:
                    pyautogui.click()
                    prev_action = "Left Click"

            # --- Right Click ---
            elif dist_thumb_middle < 0.04 and current_time - click_time > 0.5:
                pyautogui.rightClick()
                prev_action = "Right Click"
                click_time = current_time

            # --- Minimize ---
            elif dist_thumb_ring < 0.04 and current_time - click_time > 1.0:
                pyautogui.hotkey("win", "down")
                prev_action = "Minimize Window"
                click_time = current_time

            # --- Maximize (ONLY Pointer + Pinky Up) ---
            elif finger_state == [0, 1, 0, 0, 1]:
                pyautogui.hotkey("win", "up")
                prev_action = "Maximize Window"
                time.sleep(0.3)

            # --- Close Window ---
            elif dist_thumb_pinky < 0.04 and current_time - click_time > 1.0:
                pyautogui.hotkey("alt", "f4")
                prev_action = "Close Window"
                click_time = current_time

            # --- Scroll Gesture ---
            elif dist_index_middle < 0.04:
                if middle[1] < index[1] - 0.02:
                    pyautogui.scroll(100)
                    prev_action = "Scroll Up"
                elif middle[1] > index[1] + 0.02:
                    pyautogui.scroll(-100)
                    prev_action = "Scroll Down"

            # --- Zoom In (Palm facing camera → Left Hand) ---
            elif finger_state == [0, 1, 1, 1, 1] and handedness.classification[0].label == "Left":
                pyautogui.hotkey("ctrl", "+")
                prev_action = "Zoom In"
                time.sleep(0.3)

            # --- Zoom Out (Back of palm → All fingers up) ---
            elif finger_state == [1, 1, 1, 1, 1]:
                pyautogui.hotkey("ctrl", "-")
                prev_action = "Zoom Out"
                time.sleep(0.3)

            # --- Enter Gesture ---
            elif (dist_thumb_index > 0.08 and dist_thumb_middle > 0.08 and
                  dist_thumb_ring > 0.08 and dist_thumb_pinky > 0.08):
                if current_time - click_time > 1.0:
                    pyautogui.press("enter")
                    prev_action = "Enter"
                    click_time = current_time

    # Display Action
    if prev_action:
        cv2.putText(frame, f"Action: {prev_action}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Virtual Mouse (Smooth)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
