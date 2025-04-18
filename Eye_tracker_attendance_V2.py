import cv2  # OpenCV for image processing
import dlib  # dlib for face detection and landmark prediction
import numpy as np  # NumPy for numerical operations
import os  # OS module for file handling
import time  # Time module for FPS calculation
import random  # Random module for selecting blink cases
from datetime import datetime  # Datetime module for timestamping images
from scipy.spatial import distance as dist  # To calculate eye aspect ratio

# Load dlib's face detector

detector = dlib.get_frontal_face_detector()

# Path to the shape predictor model file
predictor_path = "C:/Eye_blinking_demo_project/shape_predictor_68_face_landmarks_correct.dat"
predictor = dlib.shape_predictor(predictor_path)  # Load the facial landmark predictor

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance between eye landmarks
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance between another set of eye landmarks
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance between eye landmarks
    return (A + B) / (2.0 * C)  # Compute EAR

# Indices for left and right eye landmarks in dlib's 68-point facial landmark model
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink detection parameters
EAR_THRESHOLD = 0.2  # Threshold to consider an eye blink
CONSEC_FRAMES = 3  # Number of consecutive frames an eye must be closed to count as a blink
blink_counter = 0  # Counter to track blink duration
blink_detected = False  # Flag to track successful blink detection

# Create a folder to store captured images, named by today's date
date_today = datetime.now().strftime("%Y-%m-%d")
folder_name = f"attendance_capture_{date_today}"
os.makedirs(folder_name, exist_ok=True)

# Determine the next image number based on existing images in the folder
image_count = len([f for f in os.listdir(folder_name) if f.endswith(".jpg")]) + 1

# Define possible blink actions for verification
blink_cases = ["Blink your left eye", "Blink your right eye", "Blink both eyes"]
current_case = random.choice(blink_cases)  # Randomly select an initial case

# Start video capture from the default camera (0)
cap = cv2.VideoCapture(0)
fps_start_time = time.time()  # Start time for FPS calculation
frame_count = 0  # Initialize frame counter

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit loop if the frame is not captured

    frame_count += 1
    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)  # Calculate FPS

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = detector(gray)  # Detect faces in the frame

    for face in faces:
        landmarks = predictor(gray, face)  # Detect facial landmarks
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])  # Convert to NumPy array

        left_eye = landmarks[LEFT_EYE]  # Get left eye landmarks
        right_eye = landmarks[RIGHT_EYE]  # Get right eye landmarks

        left_EAR = eye_aspect_ratio(left_eye)  # Compute EAR for left eye
        right_EAR = eye_aspect_ratio(right_eye)  # Compute EAR for right eye

        avg_EAR = (left_EAR + right_EAR) / 2.0  # Compute average EAR

        # Determine which eye is blinking
        left_blink = left_EAR < EAR_THRESHOLD
        right_blink = right_EAR < EAR_THRESHOLD

        # If eyes are closed for consecutive frames, count as a blink
        if left_blink or right_blink:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                blink_detected = True  # Blink is detected
            blink_counter = 0  # Reset counter if eyes are open

        # If the user performs the requested blink action, capture an image
        if blink_detected:
            if (current_case == "Blink your left eye" and left_blink and not right_blink) or \
               (current_case == "Blink your right eye" and right_blink and not left_blink) or \
               (current_case == "Blink both eyes" and left_blink and right_blink):

                image_name = f"{image_count}_{date_today}.jpg"  # Define image filename
                image_path = os.path.join(folder_name, image_name)  # Full path to save image
                cv2.imwrite(image_path, frame)  # Save captured frame as image
                print(f"Captured: {image_path}")

                image_count += 1  # Increment image count
                blink_detected = False  # Reset blink detection
                current_case = random.choice(blink_cases)  # Pick a new blink case

        # Draw circles around eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display FPS and EAR values on the screen
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left EAR: {left_EAR:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right EAR: {right_EAR:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, current_case, (frame.shape[1]//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Blink Detection", frame)  # Show the frame with visualizations

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
