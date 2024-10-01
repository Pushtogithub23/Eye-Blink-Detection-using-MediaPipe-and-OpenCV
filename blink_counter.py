import mediapipe as mp
import numpy as np
import cv2 as cv
from FaceMuskModule import FaceMeshGenerator
from utils import draw_text_with_bg

generator = FaceMeshGenerator()

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def eye_aspect_ratio(eye_landmarks, landmarks):
    # Calculating vertical distances
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    # Calculating horizontal distance
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    # Applying EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


EAR_THRESHOLD = 0.25  # Setting the threshold to determine eye closure
CONSEC_FRAMES = 1  # Defining the number of consecutive frames with closed eye for blink detection

blink_counter = 0
frame_counter = 0
stage = "open"
color = (0, 0, 255)

cap = cv.VideoCapture("VIDEOS/INPUTS/lady_blinking.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
filename = "VIDEOS/OUTPUTS/blink_counter.mp4"
out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    print("Error: couldn't open the video!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Generating face mesh and extracting landmarks
    frame, face_landmarks = generator.create_face_mesh(frame, draw=False)
    if len(face_landmarks) > 0:
        # Calculating EAR for both eyes
        right_ear = eye_aspect_ratio([33, 159, 158, 133, 153, 145], face_landmarks)
        left_ear = eye_aspect_ratio([362, 380, 374, 263, 386, 385], face_landmarks)

        ear = (right_ear + left_ear) / 2.0  # Averaging EAR for both eyes

        # Checking if EAR is below the blink threshold
        if ear < EAR_THRESHOLD:
            frame_counter += 1  # Counting frames with eye closed
            color = (0, 0, 255)
        else:
            # Registering a blink if eye was closed for sufficient consecutive frames
            if frame_counter >= CONSEC_FRAMES:
                blink_counter += 1
                color = (0, 255, 0)
            frame_counter = 0  # Resetting frame counter

        # Drawing landmarks for the eyes
        for loc in RIGHT_EYE:
            cv.circle(frame, (face_landmarks[loc]), 2, color, cv.FILLED)
        for loc in LEFT_EYE:
            cv.circle(frame, (face_landmarks[loc]), 2, color, cv.FILLED)

        # Displaying blink count on the frame
        draw_text_with_bg(frame, f"Blinks: {blink_counter}", (0, 45), font_scale=1, thickness=2,
                          bg_color=color, text_color=(0, 0, 0))

    # Writing the frame to the output video
    out.write(frame)
    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
