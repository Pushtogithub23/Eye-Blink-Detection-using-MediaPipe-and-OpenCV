
# Eye Blink Detection with MediaPipe and OpenCV and Real-Time EAR Plotting

This project performs real-time eye blink detection using the Eye Aspect Ratio (EAR) and MediaPipe's facial landmark detection. The project draws eye landmarks, calculates the EAR, and plots the EAR in real-time, along with blink detection results. 

![lady_blinking_output](https://github.com/user-attachments/assets/3365ad9f-ef82-4a95-918e-686b1106beec)


## Introduction

This project uses **MediaPipe's Face Mesh** to detect facial landmarks and calculates the **Eye Aspect Ratio (EAR)** to track blinks in a video. EAR is a reliable metric to identify eye closures based on geometric relationships of key landmarks around the eyes. This project processes a video file and displays the blink count and a real-time plot of the EAR value.

- Eye blink detection is performed by checking the EAR of both eyes.
- A blink is registered when the EAR falls below a defined threshold for a set number of consecutive frames.
- The result is overlaid on the video alongside the EAR plot, and the output is saved as a video.

---

## 🌟 Features

- **Real-time Blink Detection**: Accurately tracks and counts eye blinks
- **Customizable Parameters**: 
  - Adjustable Eye Aspect Ratio (EAR) threshold
  - Configurable consecutive frame detection
- **Visualization Tools**:
  - Real-time EAR plot
  - Blink counter overlay
  - Eye landmark tracking
- **Flexible Video Processing**:
  - Support for video files and webcam input
  - Optional video output saving

## 🧮 Eye Aspect Ratio (EAR) Theory

### Mathematical Formula
The Eye Aspect Ratio (EAR) is a metric to measure eye openness. The formula is:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where:
- `p1`, `p2`, `p3`, `p4`, `p5`, `p6` are specific eye landmark points
- `||x||` represents the Euclidean distance between points

### Conceptual Breakdown
1. **Vertical Eye Distances**: 
   - `||p2 - p6||` and `||p3 - p5||` measure vertical eye distances
   - These points track the upper and lower eyelid positions

2. **Horizontal Eye Distance**:
   - `||p1 - p4||` represents the horizontal eye width
   - Acts as a normalization factor

3. **Interpretation**:
   - When eyes are open: Higher EAR value
   - When eyes are closed: Lower EAR value

### Code Implementation
Here's the key implementation from the `eye_aspect_ratio` method:

```python
def eye_aspect_ratio(self, eye_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - 
                       np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - 
                       np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - 
                       np.array(landmarks[eye_landmarks[3]]))
    return (A + B) / (2.0 * C)
```

### Blink Detection Logic
The blink detection uses two critical parameters:
1. `ear_threshold`: Minimum EAR value to consider an eye "closed"
2. `consec_frames`: Number of consecutive frames below the threshold to confirm a blink

```python
def update_blink_count(self, ear):
    blink_detected = False
    
    if ear < self.ear_threshold:
        self.frame_counter += 1
    else:
        if self.frame_counter >= self.consec_frames:
            self.blink_counter += 1
            blink_detected = True
        self.frame_counter = 0
        
    return blink_detected
```

### 📊 Typical EAR Values
- Open Eyes: ~0.3 - 0.4
- Closed Eyes: ~0.2 - 0.3

### 🔍 Landmark Selection
The code uses specific MediaPipe Face Mesh landmark indices:
- Right Eye: `[33, 159, 158, 133, 153, 145]`
- Left Eye: `[362, 380, 374, 263, 386, 385]`

These indices precisely capture the eye's geometric structure for accurate EAR calculation.

### Practical Considerations
1. Average both eyes' EAR for more robust detection
2. Calibrate `ear_threshold` for different subjects
3. Adjust `consec_frames` to reduce false positives


## 🛠 Prerequisites

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV.git
cd Eye-Blink-Detection-using-MediaPipe-and-OpenCV
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Blink Counter

```python
from blink_counter import BlinkCounter

# Initialize blink counter
blink_counter = BlinkCounter(
    video_path='path/to/video.mp4',
    ear_threshold=0.3,
    consec_frames=4,
    save_video=True
)
blink_counter.process_video()
```
![blink_counter_1](https://github.com/user-attachments/assets/a024d6d4-6538-49c8-8b76-a9cd127d62ac)

### Blink Counter with EAR Plot

```python
from blink_counter_and_EAR_plot import BlinkCounterandEARPlot

# Initialize blink counter with real-time EAR visualization
blink_counter = BlinkCounterandEARPlot(
    video_path='path/to/video.mp4',
    threshold=0.294,
    consec_frames=3,
    save_video=True
)
blink_counter.process_video()
```

![blinking_1_output](https://github.com/user-attachments/assets/9374bb45-83b9-44d1-83bf-b85e9b5c45b5)


## 📊 Key Components

1. **FaceMeshModule.py**
   - Utilizes MediaPipe Face Mesh for facial landmark detection
   - Supports both image and video processing
   - Configurable detection parameters

2. **utils.py**
   - Provides utility drawing functions
   - Enhanced error handling for OpenCV operations
   - Text and overlay drawing utilities

3. **Jupyter Notebook**
   - In-depth analysis of eye landmarks
   - EAR threshold and consecutive frame optimization
   - Detailed visualization of blink detection parameters

## 🔍 How It Works

The blink detection system uses the Eye Aspect Ratio (EAR) algorithm:
- Calculates the ratio of eye landmark distances
- Detects blinks when EAR falls below a specified threshold
- Requires consecutive frames below the threshold to confirm a blink

## 🎓 Customization

You can fine-tune blink detection by adjusting the following:
- `ear_threshold`: Sensitivity of blink detection
- `consec_frames`: Number of frames required to confirm a blink

## 📝 Jupyter Notebook Analysis

The included Jupyter notebook provides:
- Landmark visualization
- EAR calculation methodology
- Parameter optimization techniques
- Comparative analysis of different threshold settings


---

## References

1. This project utilizes **MediaPipe** for facial landmark detection. The official MediaPipe documentation can be found [here](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
2. Plotly [documentation](https://plotly.com/python/subplots/) for interactive plotting.
3. [Eye Aspect Ratio (EAR)](https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706)
4. [Dewi, C., Chen, R., Chang, C., Wu, S., Jiang, X., & Yu, H. (2021). Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety. Electronics, 11(19), 3183](https://doi.org/10.3390/electronics11193183).
5. [Soukupová, E., & Čech, M. (2016). Real-Time Eye Blink Detection using Facial Landmarks. *Computer Vision and Applications*, 2016(1), 1-10](https://doi.org/10.1007/s00521-016-0934-z).

--- 
