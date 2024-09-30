import cv2
import numpy as np
import dlib
import argparse
import time
from math import hypot
import winsound

# Argument parser to choose between webcam or video file
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true', help='Use webcam for blink detection')
parser.add_argument('--video', type=str, help='Path to video file for blink detection')
args = parser.parse_args()

# Dlib setup for facial landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper function to calculate midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Helper function to calculate the blinking ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

# Function to handle blink detection logic
def blink_detection(cap):
    close_eye_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Calculate blink ratio for left and right eye
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # If the blinking ratio is high, consider it as blink
            if blinking_ratio > 4.0:
                close_eye_count += 1
                if close_eye_count == 10:  # Adjust this threshold as per requirement
                    print("Blink detected!")
                    winsound.Beep(3520, 100)  # Sound alert on blink detection
                    close_eye_count = 0

            else:
                close_eye_count = 0

        # Display the frame with blink detection data
        cv2.putText(frame, f"Blinks: {close_eye_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Blink Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution logic
if __name__ == "__main__":
    if args.webcam:
        print("Starting blink detection using webcam...")
        cap = cv2.VideoCapture(0)  # 0 means default webcam
    elif args.video:
        print(f"Starting blink detection using video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print("Error: No input source selected. Use --webcam or --video <path>")
        exit(1)

    # Start blink detection
    blink_detection(cap)
