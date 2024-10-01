
# Eye Blink Detection with EAR and Real-Time Plotting

This project performs real-time eye blink detection using the Eye Aspect Ratio (EAR) and MediaPipe's facial landmark detection. The project draws eye landmarks, calculates the EAR, and plots the EAR in real-time along with blink detection results. The processed video frames are saved alongside the EAR plot.

![blink_counter_and_ear_plot](https://github.com/user-attachments/assets/c367824f-017f-43c5-966f-67ef771b0f5e)


## Table of Contents
1. [Introduction](#introduction)
2. [EAR Formula Explanation](#ear-formula)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Results](#results)
6. [References](#references)

---

## Introduction

This project uses **MediaPipe's Face Mesh** to detect facial landmarks and calculates the **Eye Aspect Ratio (EAR)** to track blinks in a video. EAR is a reliable metric to identify eye closures based on geometric relationships of key landmarks around the eyes. This project processes a video file and displays both the blink count and a real-time plot of the EAR value.

![image](https://github.com/user-attachments/assets/91b03db7-b81a-4e13-bd5d-a0ae502524c4)

- Eye blink detection is performed by checking the EAR of both eyes.
- A blink is registered when the EAR falls below a defined threshold for a set number of consecutive frames.
- The result is overlaid on the video alongside the EAR plot, and the output is saved as a video.

---

## EAR Formula

The **Eye Aspect Ratio (EAR)** is computed using the vertical and horizontal distances between six specific facial landmarks around each eye.

### EAR Calculation:

For the right eye, the landmarks used are:
`[33, 159, 158, 133, 153, 145]`.

For the left eye, the landmarks are:
`[362, 380, 374, 263, 386, 385]`.

The EAR formula for each eye is:

$$ EAR = \frac{|P_2 - P_6| + |P_3 - P_5|}{2 \times |P_1 - P_4|} $$

Where:
- $P_1, P_2, P_3, P_4, P_5, P_6$ are the corresponding landmark coordinates for each eye.
  - $P_2$ and $P_6$ are the vertical points,
  - $P_3$ and $P_5$ are also vertical points,
  - $P_1$ and $P_4$ are the horizontal endpoints of the eye.

### Explanation:

- **Numerator**: Sum of the distances between the two sets of vertical eye landmarks $|P_2 - P_6|$ and $|P_3 - P_5|$.
- **Denominator**: Twice the distance between the horizontal eye landmarks $|P_1 - P_4|$.
- The EAR value decreases when the eye closes and increases when the eye opens, making it a useful metric for detecting blinks.

---

## Project Structure

```
├── VIDEOS
│   ├── DOWNLOADED VIDEOS
│   │   └── lady_blinking.mp4      # Input video file for blink detection
│   ├── MASKED VIDEOS
│       └── blink_counter_and_ear_plot.mp4  # Output video with EAR plot and blink count
├── Face_muskModule.py              # Module to detect facial landmarks using MediaPipe Face Mesh
├── utils.py                        # Utility functions for drawing text and bounding boxes
├── requirements.txt                # List of Python dependencies
└── main.py                         # Main script for blink detection and EAR plotting
```

---

## How to Run

### Step 1: Install dependencies

All the required dependencies are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```

### Step 2: Run the blink detection

Once the dependencies are installed, you can run the main script:

```bash
python main.py
```

- This will process the input video (`lady_blinking.mp4`) and display the result with EAR plots in real time. The output video will be saved to the `MASKED VIDEOS` folder as `blink_counter_and_ear_plot.mp4`.

### Key Parameters:

- **EAR_THRESHOLD**: Threshold value to detect eye closure (set to 0.275 by default).
- **CONSEC_FRAMES**: Minimum consecutive frames where EAR must be below the threshold to register a blink.

---

## Results

The output video displays the following:
- **Blink Count**: The number of blinks detected is displayed on the frame.
- **Real-Time EAR Plot**: The live EAR value is plotted alongside the EAR threshold to visualize eye closure and blinks.

The final video is saved with both the blink counter and the EAR plot stacked together for easy visualization.

---

## References

1. This project utilizes **MediaPipe** for facial landmark detection. The official MediaPipe documentation can be found [here](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
2. [Video by Karolina Kaboompics from Pexels](https://www.pexels.com/video/woman-looking-at-camera-7195601/)


--- 
