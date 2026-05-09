"""
MediaPipe face mesh + emotion detection for NPC perception.
Tracks user's face, detects gaze direction, and infers emotional state.
SDKs: MediaPipe, OpenCV
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not available. Install: pip install mediapipe")


@dataclass
class FaceState:
    detected: bool
    landmarks: Optional[np.ndarray]    # (468, 3) normalized x,y,z
    gaze_direction: str                 # "looking_at_camera", "looking_away", "eyes_closed"
    emotion: str                        # "neutral", "happy", "surprised", "thinking"
    head_pose: Dict[str, float]         # {yaw, pitch, roll} in degrees
    mouth_open: bool
    eye_blink: Dict[str, bool]          # {left, right}
    confidence: float


class MediaPipeFaceTracker:
    """
    Real-time face tracking using MediaPipe FaceMesh.
    Provides 468-landmark face mesh, gaze, emotion, and head pose.
    """

    # Key landmark indices
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    NOSE_TIP = 4
    LEFT_EYE_CENTER = 468
    RIGHT_EYE_CENTER = 473

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not MP_AVAILABLE:
            raise ImportError("mediapipe required. Install: pip install mediapipe")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._prev_state: Optional[FaceState] = None
        print(f"[MediaPipe] FaceTracker ready | max_faces={max_faces}")

    def process_frame(self, frame: np.ndarray) -> FaceState:
        """
        Process a single BGR frame and return face state.
        frame: (H, W, 3) BGR numpy array
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceState(
                detected=False, landmarks=None,
                gaze_direction="unknown", emotion="neutral",
                head_pose={"yaw": 0, "pitch": 0, "roll": 0},
                mouth_open=False, eye_blink={"left": False, "right": False},
                confidence=0.0,
            )

        lm = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark])

        gaze = self._compute_gaze(landmarks, w, h)
        emotion = self._estimate_emotion(landmarks)
        head_pose = self._compute_head_pose(landmarks, w, h)
        mouth_open = self._is_mouth_open(landmarks)
        eye_blink = self._detect_blinks(landmarks)

        state = FaceState(
            detected=True,
            landmarks=landmarks,
            gaze_direction=gaze,
            emotion=emotion,
            head_pose=head_pose,
            mouth_open=mouth_open,
            eye_blink=eye_blink,
            confidence=0.9,
        )
        self._prev_state = state
        return state

    def _compute_gaze(self, landmarks: np.ndarray, w: int, h: int) -> str:
        """Determine if user is looking at camera."""
        nose = landmarks[self.NOSE_TIP]
        nose_x_norm = nose[0] / w
        nose_y_norm = nose[1] / h
        if 0.35 < nose_x_norm < 0.65 and 0.3 < nose_y_norm < 0.7:
            return "looking_at_camera"
        return "looking_away"

    def _estimate_emotion(self, landmarks: np.ndarray) -> str:
        """Coarse emotion estimation from facial geometry."""
        mouth_pts = landmarks[self.MOUTH_INDICES]
        mouth_width = np.linalg.norm(mouth_pts[0] - mouth_pts[6])
        mouth_height = np.linalg.norm(mouth_pts[3] - mouth_pts[9])
        smile_ratio = mouth_width / (mouth_height + 1e-6)
        if smile_ratio > 4.0:
            return "happy"
        if mouth_height > 15:
            return "surprised"
        return "neutral"

    def _compute_head_pose(self, landmarks: np.ndarray, w: int, h: int) -> Dict[str, float]:
        """Estimate head pose from 6-point correspondences."""
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ], dtype=np.float64)
        image_points = np.array([
            landmarks[1][:2],
            landmarks[152][:2],
            landmarks[263][:2],
            landmarks[33][:2],
            landmarks[287][:2],
            landmarks[57][:2],
        ], dtype=np.float64)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                return {"yaw": float(angles[1]), "pitch": float(angles[0]), "roll": float(angles[2])}
        except Exception:
            pass
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def _is_mouth_open(self, landmarks: np.ndarray, threshold: float = 12.0) -> bool:
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        return float(np.linalg.norm(upper_lip - lower_lip)) > threshold

    def _detect_blinks(self, landmarks: np.ndarray) -> Dict[str, bool]:
        def ear(eye_pts):
            v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
            v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
            h = np.linalg.norm(eye_pts[0] - eye_pts[3])
            return (v1 + v2) / (2.0 * h + 1e-6)
        l_pts = landmarks[[33, 160, 158, 133, 153, 144]]
        r_pts = landmarks[[362, 385, 387, 263, 373, 380]]
        return {"left": ear(l_pts) < 0.2, "right": ear(r_pts) < 0.2}

    def run_camera(self, callback=None, camera_id: int = 0):
        """Run face tracking on live camera feed."""
        cap = cv2.VideoCapture(camera_id)
        print(f"[MediaPipe] Camera {camera_id} opened. Press Q to quit.")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                state = self.process_frame(frame)
                if callback:
                    callback(state, frame)
                cv2.putText(frame, f"Gaze: {state.gaze_direction} | Emotion: {state.emotion}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def close(self):
        self.face_mesh.close()
