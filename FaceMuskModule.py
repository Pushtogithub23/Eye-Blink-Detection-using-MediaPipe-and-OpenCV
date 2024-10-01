import cv2 as cv
import mediapipe as mp
import os


class FaceMeshGenerator:
    def __init__(self, mode=False, num_faces=2, min_detection_con=0.5, min_track_con=0.5):
        self.results = None
        self.mode = mode
        self.num_faces = num_faces
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con

        self.mp_faceDetector = mp.solutions.face_mesh
        self.face_mesh = self.mp_faceDetector.FaceMesh(static_image_mode=self.mode, max_num_faces=self.num_faces,
                                                       min_detection_confidence=self.min_detection_con,
                                                       min_tracking_confidence=self.min_track_con)
        self.mp_Draw = mp.solutions.drawing_utils
        self.drawSpecs = self.mp_Draw.DrawingSpec(thickness=1, circle_radius=2)

    def create_face_mesh(self, frame, draw = True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # generating face mask
        self.results = self.face_mesh.process(frame_rgb)
        landmarks_dict = {}
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_Draw.draw_landmarks(frame, face_lms, self.mp_faceDetector.FACEMESH_CONTOURS, self.drawSpecs,
                                            self.drawSpecs)

                for ID, lm in enumerate(face_lms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks_dict[ID] = (x, y)
        return frame, landmarks_dict


def generate_face_mesh(video_path, resizing_factor, save_vido=False, filename=None):
    cap = cv.VideoCapture(0 if video_path == 0 else video_path)
    if not cap.isOpened():
        print("Couldn't open the video!")
        return
        # getting video properties
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    resized_frame_shape = (int(resizing_factor * frame_width), int(resizing_factor * frame_height))

    # To save video
    if save_vido:
        if filename:
            video_dir = r"D:\PyCharm\PyCharm_files\MEDIAPIPE\FACE_MESH\VIDEOS"
            save_vido_path = os.path.join(video_dir, filename)
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(save_vido_path, fourcc, fps, (frame_width, frame_height))
        else:
            return

    mesh_generator = FaceMeshGenerator()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks_dict = mesh_generator.create_face_mesh(frame)

        if save_vido:
            out.write(frame)

        if video_path == 0:
            frame = cv.flip(frame, 1)
        resized_frame = cv.resize(frame, resized_frame_shape)
        cv.imshow('Video', resized_frame)
        if cv.waitKey(1) & 0xff == ord('p'):
            break

    cap.release()
    if save_vido:
        out.release()
    cv.destroyAllWindows()


# generate_face_mesh(0, resizing_factor=1)

