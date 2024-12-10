import numpy as np
import cv2 as cv
from FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils
import os

class BlinkCounter:
    """
    A class to detect and count eye blinks in a video using facial landmarks.
    
    This class processes video input to detect eye blinks by calculating the Eye Aspect Ratio (EAR)
    of both eyes. It can either process a video file or save the processed output to a new video file.
    
    Attributes:
        ear_threshold (float): Threshold below which eyes are considered closed
        consec_frames (int): Number of consecutive frames eyes must be closed to count as blink
    """
    
    def __init__(self, video_path, ear_threshold, consec_frames, save_video=False, output_filename=None):
        """
        Initialize the BlinkCounter with video source and processing parameters.
        
        Args:
            video_path (str): Path to the input video file
            ear_threshold (float): Threshold for eye aspect ratio to detect blink (default: 0.3)
            consec_frames (int): Number of consecutive frames needed to confirm blink (default: 4)
            save_video (bool): Whether to save the processed video
            output_filename (str, optional): Name for the output video file
        """
        # Initialize face mesh detector
        self.generator = FaceMeshGenerator() 
        self.video_path = video_path
        self.save_video = save_video
        self.output_filename = output_filename
        
        # Define facial landmarks for eye detection
        # Each list contains indices corresponding to points around the eyes
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Specific landmarks for EAR calculation
        # These specific points are used to calculate the eye aspect ratio
        self.RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
        self.LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]
        
        # Blink detection parameters
        self.ear_threshold = ear_threshold  # Eye aspect ratio threshold for blink detection
        self.consec_frames = consec_frames  # Minimum consecutive frames for a valid blink
        self.blink_counter = 0    # Counter for total blinks detected
        self.frame_counter = 0    # Counter for consecutive frames below threshold
        
        # Define colors for visualization (in BGR format)
        self.GREEN_COLOR = (86, 241, 13)  # Used when eyes are open
        self.RED_COLOR = (30, 46, 209)    # Used when eyes are closed
        
        # Set up output video directory and path if saving is enabled
        if self.save_video and self.output_filename:
            save_dir = "DATA/VIDEOS/OUTPUTS" # Replace with your desired directory
            os.makedirs(save_dir, exist_ok=True)
            self.output_filename = os.path.join(save_dir, self.output_filename)

    def update_blink_count(self, ear):
        """
        Update blink counter based on current eye aspect ratio.
        
        This method implements the blink detection logic:
        - If EAR is below threshold, increment frame counter
        - If EAR returns above threshold and enough consecutive frames were counted,
          increment blink counter
        
        Args:
            ear (float): Current eye aspect ratio
            
        Returns:
            bool: True if a new blink was detected, False otherwise
        """
        blink_detected = False
        
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                blink_detected = True
            self.frame_counter = 0
            
        return blink_detected

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        """
        Calculate the eye aspect ratio (EAR) for given eye landmarks.
        
        The EAR is calculated using the formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        where p1-p6 are specific points around the eye.
        
        Args:
            eye_landmarks (list): Indices of landmarks for one eye
            landmarks (list): List of all facial landmarks
        
        Returns:
            float: Calculated eye aspect ratio
        """
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def set_colors(self, ear):
        """
        Determine visualization color based on eye aspect ratio.
        
        Args:
            ear (float): Current eye aspect ratio
        
        Returns:
            tuple: BGR color values
        """
        return self.RED_COLOR if ear < self.ear_threshold else self.GREEN_COLOR

    def draw_eye_landmarks(self, frame, landmarks, eye_landmarks, color):
        """
        Draw landmarks around the eyes on the frame.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on
            landmarks (list): List of facial landmarks
            eye_landmarks (list): Indices of landmarks for one eye
            color (tuple): BGR color values for drawing
        """
        for loc in eye_landmarks:
            cv.circle(frame, (landmarks[loc]), 4, color, cv.FILLED)

    def process_video(self):
        """
        Main method to process the video and detect blinks.
        
        This method:
        1. Opens the video file
        2. Processes each frame to detect faces and calculate EAR
        3. Counts blinks based on EAR values
        4. Displays and optionally saves the processed video
        
        Raises:
            IOError: If video file cannot be opened
            Exception: For other processing errors
        """
        try:
            # Open video capture
            cap = cv.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {self.video_path}")
                raise IOError("Error: couldn't open the video!")

            # Get video properties
            w, h, fps = (int(cap.get(x)) for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS
            ))

            # Initialize video writer if saving is enabled
            if self.save_video:
                self.out = cv.VideoWriter(
                    self.output_filename,
                    cv.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h)
                )

            # Main processing loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect facial landmarks
                frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)

                if len(face_landmarks) > 0:
                    # Calculate eye aspect ratio
                    right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
                    left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
                    ear = (right_ear + left_ear) / 2.0

                    # Update blink detection
                    self.update_blink_count(ear)

                    # Determine visualization color based on EAR
                    color = self.set_colors(ear)

                    # Draw visualizations
                    self.draw_eye_landmarks(frame, face_landmarks, self.RIGHT_EYE, color)
                    self.draw_eye_landmarks(frame, face_landmarks, self.LEFT_EYE, color)
                    DrawingUtils.draw_text_with_bg(frame, f"Blinks: {self.blink_counter}", (0, 60),
                                    font_scale=2, thickness=3,
                                    bg_color=color, text_color=(0, 0, 0))

                    # Save frame if enabled
                    if self.save_video:
                        self.out.write(frame)

                    # Display the frame
                    resized_frame = cv.resize(frame, (1280, 720))
                    cv.imshow("Blink Counter", resized_frame)

                # Break loop if 'p' is pressed
                if cv.waitKey(int(1000/fps)) & 0xFF == ord('p'):
                    break

            # Cleanup
            cap.release()
            if self.save_video:
                self.out.release()
            cv.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    input_video_path = "DATA/VIDEOS/INPUTS/blinking_4.mp4"
    
    # Create blink counter with custom parameters
    blink_counter = BlinkCounter(
        video_path=input_video_path,
        ear_threshold=0.3,  
        consec_frames=4,    
        save_video=True,
        output_filename="blink_counter_4.mp4"
    )
    blink_counter.process_video()
