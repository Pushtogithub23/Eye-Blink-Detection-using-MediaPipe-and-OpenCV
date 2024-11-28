import cv2 as cv
import mediapipe as mp
import os


class FaceMeshGenerator:
    def __init__(self, mode=False, num_faces=2, min_detection_con=0.5, min_track_con=0.5):
        """
        Initialize FaceMesh detector with specified parameters
        """
        try:
            self.results = None
            self.mode = mode
            self.num_faces = num_faces
            self.min_detection_con = min_detection_con
            self.min_track_con = min_track_con

            self.mp_faceDetector = mp.solutions.face_mesh
            self.face_mesh = self.mp_faceDetector.FaceMesh(
                static_image_mode=self.mode,
                max_num_faces=self.num_faces,
                min_detection_confidence=self.min_detection_con,
                min_tracking_confidence=self.min_track_con
            )

            self.mp_Draw = mp.solutions.drawing_utils
            self.drawSpecs = self.mp_Draw.DrawingSpec(thickness=1, circle_radius=2)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceMeshGenerator: {str(e)}")

    def create_face_mesh(self, frame, draw=True):
        """
        Create face mesh landmarks for the given frame
        Returns: processed frame and dictionary of landmarks
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")

        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.face_mesh.process(frame_rgb)
            landmarks_dict = {}

            if self.results.multi_face_landmarks:
                for face_lms in self.results.multi_face_landmarks:
                    if draw:
                        self.mp_Draw.draw_landmarks(
                            frame,
                            face_lms,
                            self.mp_faceDetector.FACEMESH_CONTOURS,
                            self.drawSpecs,
                            self.drawSpecs
                        )

                    # Convert normalized landmarks to pixel coordinates
                    ih, iw, _ = frame.shape
                    for ID, lm in enumerate(face_lms.landmark):
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        landmarks_dict[ID] = (x, y)

            return frame, landmarks_dict
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {str(e)}")


def generate_face_mesh(video_path, resizing_factor, save_video=False, filename=None):
    """
    Process video stream and generate face mesh
    Args:
        video_path: Path to video file or 0 for webcam
        resizing_factor: Factor to resize output display
        save_video: Boolean to enable video saving
        filename: Output video filename
    """
    try:
        # Initialize video capture
        cap = cv.VideoCapture(0 if video_path == 0 else video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        # Get video properties
        f_w, f_h, fps = (int(cap.get(x)) for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS
            ))

        # Initialize video writer if saving is enabled
        out = None
        if save_video:
            if not filename:
                raise ValueError("Filename must be provided when save_video is True")
            
            video_dir = r"D:\PyCharm\PyCharm_files\MEDIAPIPE\FACE_MESH\VIDEOS"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                
            save_path = os.path.join(video_dir, filename)
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(save_path, fourcc, fps, (f_w, f_h))

        mesh_generator = FaceMeshGenerator()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, landmarks_dict = mesh_generator.create_face_mesh(frame)

            if save_video and out is not None:
                out.write(frame)

            if video_path == 0:
                frame = cv.flip(frame, 1)

            # Ensure resizing factor is valid
            if resizing_factor <= 0:
                raise ValueError("Resizing factor must be positive")

            resized_frame = cv.resize(frame, (int(f_w * resizing_factor), int(f_h * resizing_factor)))
            cv.imshow('Video', resized_frame)
            
            # Break loop if 'p' is pressed
            if cv.waitKey(1) & 0xff == ord('p'):
                break

    except Exception as e:
        print(f"Error during video processing: {str(e)}")
    
    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    video_path = 0
    resizing_factor = 1 if video_path == 0 else 0.5
    generate_face_mesh(video_path, resizing_factor)

