import cv2  # OpenCV for image processing
import dlib  # dlib for face detection and landmark prediction
import numpy as np  # NumPy for numerical operations
import os  # OS module for file handling
import time  # Time module for FPS calculation
import random  # Random module for selecting blink cases
import logging  # Logging module for better debugging
from datetime import datetime  # Datetime module for timestamping images
from scipy.spatial import distance as dist  # To calculate eye aspect ratio
from collections import deque  # For EAR visualization

# Configuration
PREDICTOR_PATH = "C:/Eye_blinking_demo_project/shape_predictor_68_face_landmarks_correct.dat"
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
CAPTURE_FOLDER_PREFIX = "attendance_capture_"
FPS_UPDATE_INTERVAL = 30  # Update FPS every 30 frames

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) to determine if an eye is open or closed.

    Args:
        eye (list): List of 6 (x, y) coordinates representing eye landmarks.

    Returns:
        float: EAR value.
    """
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance between eye landmarks
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance between another set of eye landmarks
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance between eye landmarks
    return (A + B) / (2.0 * C)  # Compute EAR

# Function to draw eye landmarks
def draw_eye_landmarks(frame, eye, color=(0, 255, 0)):
    for (x, y) in eye:
        cv2.circle(frame, (x, y), 2, color, -1)

# Function to display text on the frame
def display_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Graceful exit handler
def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released. Exiting program.")

# Main function
def main():
    # Load dlib's face detector and shape predictor
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Shape predictor model not found at {PREDICTOR_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Indices for left and right eye landmarks in dlib's 68-point facial landmark model
    LEFT_EYE = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))

    # Blink detection parameters
    blink_counter = 0
    blink_detected = False

    # Create a folder to store captured images, named by today's date
    date_today = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{CAPTURE_FOLDER_PREFIX}{date_today}"
    os.makedirs(folder_name, exist_ok=True)

    # Determine the next image number based on existing images in the folder
    image_count = len([f for f in os.listdir(folder_name) if f.endswith(".jpg")]) + 1

    # Define possible blink actions for verification
    blink_cases = ["Blink your left eye", "Blink your right eye", "Blink both eyes"]
    current_case = random.choice(blink_cases)

    # Start video capture from the default camera (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera. Please check your device.")

    fps_start_time = time.time()
    frame_count = 0
    ear_history = deque(maxlen=100)  # Store last 100 EAR values for visualization

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
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

                # Update EAR history for visualization
                ear_history.append(avg_EAR)

                # Determine which eye is blinking
                left_blink = left_EAR < EAR_THRESHOLD
                right_blink = right_EAR < EAR_THRESHOLD

                # Blink detection logic
                if left_blink or right_blink:
                    blink_counter += 1
                else:
                    if blink_counter >= CONSEC_FRAMES:
                        blink_detected = True
                    blink_counter = max(0, blink_counter - 1)

                # If the user performs the requested blink action, capture an image
                if blink_detected:
                    if (current_case == "Blink your left eye" and left_blink and not right_blink) or \
                       (current_case == "Blink your right eye" and right_blink and not left_blink) or \
                       (current_case == "Blink both eyes" and left_blink and right_blink):

                        image_name = f"{image_count}_{date_today}.jpg"
                        image_path = os.path.join(folder_name, image_name)
                        cv2.imwrite(image_path, frame)
                        logging.info(f"Captured: {image_path}")

                        image_count += 1
                        blink_detected = False
                        current_case = random.choice(blink_cases)

                # Draw eye landmarks and display EAR values
                draw_eye_landmarks(frame, left_eye)
                draw_eye_landmarks(frame, right_eye)
                display_text(frame, f"Left EAR: {left_EAR:.2f}", (10, 60), (0, 255, 255))
                display_text(frame, f"Right EAR: {right_EAR:.2f}", (10, 90), (0, 255, 255))
                display_text(frame, current_case, (frame.shape[1] // 2 - 100, 50), (255, 0, 0))

            # Update FPS periodically
            if frame_count % FPS_UPDATE_INTERVAL == 0:
                fps = frame_count / (time.time() - fps_start_time)
                display_text(frame, f"FPS: {fps:.2f}", (10, 30))

            cv2.imshow("Blink Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                break

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        cleanup(cap)

if __name__ == "__main__":
    main()