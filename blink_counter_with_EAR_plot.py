import mediapipe as mp
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from FaceMuskModule import FaceMeshGenerator
from utils import draw_text_with_bg

generator = FaceMeshGenerator()

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Setting EAR threshold and frame conditions
EAR_THRESHOLD = 0.275
CONSEC_FRAMES = 1
blink_counter = 0
frame_counter = 0

# Storing EAR values for live plotting
ear_values = []
frame_numbers = []
max_frames = 50

# Initializing the Matplotlib plot
plt.ioff()
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvas(fig)  # Creating canvas for the plot

x_vals = list(range(max_frames))
y_vals = [0] * max_frames
Y_vals = [EAR_THRESHOLD] * max_frames

# Plotting the EAR and the EAR threshold
line, = ax.plot(x_vals, y_vals, color='green', label="EAR")
line_1, = ax.plot(x_vals, Y_vals, color='red', label="EAR_THRESHOLD")
ax.set_ylim(0.15, 0.4)
ax.set_xlim(0, max_frames)
ax.set_xlabel("Frame Number")
ax.set_ylabel("EAR")
ax.set_title("Real-Time Eye Aspect Ratio (EAR)")
ax.grid(True)
ax.legend(fontsize=8)

cap = cv.VideoCapture("VIDEOS/INPUTS/lady_blinking.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

# Initializing variables for storing the new width and height
new_w, new_h = None, None

filename = "VIDEOS/OUTPUTS/blink_counter_and_ear_plot.mp4"
out = None  # Initializing the video writer as None for now

if not cap.isOpened():
    print("Error: couldn't open the video!")

frame_number = 0


# Defining helper functions
def eye_aspect_ratio(eye_landmarks, landmarks):
    # Calculating EAR using landmarks
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    ear = (A + B) / (2.0 * C)
    return ear


def plot_to_image(fig):
    """Converting the Matplotlib figure to an image that can be displayed with OpenCV."""
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    return img


def set_colors(ear):
    """Setting the color of the eye landmarks and plot line based on EAR value."""
    if ear < EAR_THRESHOLD:
        return (0, 0, 255), 'blue'  # Using red for blink (BGR for OpenCV, RGB for Matplotlib)
    else:
        return (0, 255, 0), 'green'  # Using green for normal (BGR for OpenCV, RGB for Matplotlib)


def draw_eye_landmarks(frame, landmarks, eye_landmarks, color):
    """Drawing eye landmarks on the frame."""
    for loc in eye_landmarks:
        cv.circle(frame, (landmarks[loc]), 2, color, cv.FILLED)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, face_landmarks = generator.create_face_mesh(frame, draw=False)

    if len(face_landmarks) > 0:
        # Calculating EAR for both eyes
        right_ear = eye_aspect_ratio([33, 159, 158, 133, 153, 145], face_landmarks)
        left_ear = eye_aspect_ratio([362, 380, 374, 263, 386, 385], face_landmarks)
        ear = (right_ear + left_ear) / 2.0

        # Appending EAR values for plotting
        ear_values.append(ear)
        frame_numbers.append(frame_number)

        # Setting color based on blink detection
        color, line_color = set_colors(ear)

        # Updating blink count
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        # Drawing landmarks for both eyes
        draw_eye_landmarks(frame, face_landmarks, RIGHT_EYE, color)
        draw_eye_landmarks(frame, face_landmarks, LEFT_EYE, color)

        # Displaying blink count on frame
        draw_text_with_bg(frame, f"Blinks: {blink_counter}", (0, 45), font_scale=1.5, thickness=2,
                          bg_color=color, text_color=(0, 0, 0))

        # Handling the plot updating
        if len(ear_values) > max_frames:
            ear_values.pop(0)
            frame_numbers.pop(0)

        # Updating the plot with EAR values
        line.set_xdata(frame_numbers)
        line.set_ydata(ear_values)
        line.set_color(line_color)
        line_1.set_xdata(frame_numbers)
        line_1.set_ydata([EAR_THRESHOLD] * len(frame_numbers))

        ax.set_xlim(min(frame_numbers), max(frame_numbers))
        ax.draw_artist(ax.patch)
        ax.draw_artist(line)
        ax.draw_artist(line_1)
        fig.canvas.flush_events()

        # Converting Matplotlib plot to image for OpenCV
        plot_img = plot_to_image(fig)

        # Resizing plot width to match the video frame width
        plot_img_resized = cv.resize(plot_img,
                                     (frame.shape[1], int(plot_img.shape[0] * frame.shape[1] / plot_img.shape[1])))

        # Stacking video and plot vertically
        stacked_frame = cv.vconcat([frame, plot_img_resized])

        # Resizing the final stacked frame to display
        resizing_factor = 0.45
        resized_shape = (int(resizing_factor * stacked_frame.shape[1]), int(resizing_factor * stacked_frame.shape[0]))
        stacked_frame_resized = cv.resize(stacked_frame, resized_shape)

        # Checking the dimensions of the stacked frame
        if new_w is None and new_h is None:
            # Setting width and height for the video writer
            new_w = stacked_frame.shape[1]
            new_h = stacked_frame.shape[0]

            # Initializing the video writer with the new dimensions
            out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))

        # Writing the stacked frame to the video
        out.write(stacked_frame)

        # Displaying the result
        cv.imshow("Video with EAR Plot", stacked_frame_resized)

    frame_number += 1

    if cv.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
