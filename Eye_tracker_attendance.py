import cv2
import dlib
import numpy as np
import os
import time
import random
from datetime import datetime
from scipy.spatial import distance as dist

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()

# Correct path to the shape predictor file
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Eye landmark indexes
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink detection threshold
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
blink_counter = 0
blink_detected = False

# Generate today's folder name
date_today = datetime.now().strftime("%Y-%m-%d")
folder_name = f"attendance_capture_{date_today}"
os.makedirs(folder_name, exist_ok=True)

# Get the next image number
image_count = len([f for f in os.listdir(folder_name) if f.endswith(".jpg")]) + 1

# Possible blink cases
blink_cases = ["Blink your left eye", "Blink your right eye", "Blink both eyes"]
current_case = random.choice(blink_cases)

# Start video capture
cap = cv2.VideoCapture(0)
fps_start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Determine which eye is blinking
        left_blink = left_EAR < EAR_THRESHOLD
        right_blink = right_EAR < EAR_THRESHOLD

        # If eyes are closed for consecutive frames, count as a blink
        if left_blink or right_blink:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0

        # Randomly display a case and capture when matched
        if blink_detected:
            if (current_case == "Blink your left eye" and left_blink and not right_blink) or \
               (current_case == "Blink your right eye" and right_blink and not left_blink) or \
               (current_case == "Blink both eyes" and left_blink and right_blink):

                image_name = f"{image_count}_{date_today}.jpg"
                image_path = os.path.join(folder_name, image_name)
                cv2.imwrite(image_path, frame)
                print(f"Captured: {image_path}")

                image_count += 1
                blink_detected = False
                current_case = random.choice(blink_cases)  # Change case after successful capture

        # Draw eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display information
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left EAR: {left_EAR:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right EAR: {right_EAR:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, current_case, (frame.shape[1]//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
