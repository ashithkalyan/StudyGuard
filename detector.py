# === detector.py ===
"""
StudyGuard AI - Core Computer Vision Detector
Framework-agnostic class. No Flask imports here on purpose,
so this module can be reused (CLI tool, different backend, tests) later.
"""

import math
import os
import time

import cv2
import numpy as np
import mediapipe as mp

YOLO_AVAILABLE = False
if os.environ.get("DISABLE_YOLO", "false").lower() != "true":
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except Exception:
        YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe FaceMesh - 468/478 point model)
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
TOP_LIP = 13
BOTTOM_LIP = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# 6-point head pose model landmarks
HP_NOSE_TIP = 1
HP_CHIN = 152
HP_LEFT_EYE = 263
HP_RIGHT_EYE = 33
HP_LEFT_MOUTH = 287
HP_RIGHT_MOUTH = 57

# Pose landmark indices (MediaPipe Pose - 33 point model)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# 3D reference model for solvePnP (generic head model, millimeters)
MODEL_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],          # nose tip
    [0.0, -63.6, -12.5],      # chin
    [-43.3, 32.7, -26.0],     # left eye corner
    [43.3, 32.7, -26.0],      # right eye corner
    [-28.9, -28.9, -24.1],    # left mouth corner
    [28.9, -28.9, -24.1],     # right mouth corner
], dtype=np.float64)


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class StudyDetector:
    """
    Encapsulates all computer vision logic for one monitoring session.
    Call process_frame() once per captured frame.
    Call record_baseline() once at the start of a session, before
    process_frame() is used for real monitoring.
    """

    # ---- temporal thresholds (assuming ~10 processed fps) -----------------
    WARNING_FRAMES = 30      # ~3s of bad state -> WARNING
    ALERT_FRAMES = 80        # ~8s of bad state -> ALERT
    RECOVER_FRAMES = 20      # ~2s of good state -> start recovering
    FOCUS_RESTORE_FRAMES = 40  # ~4s of good state -> back to FOCUSED

    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh_module = mp_face_mesh
        self.mp_pose_module = mp_pose

        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[StudyDetector] YOLO failed to load: {e}")
                self.yolo_model = None

        # ---- state -----------------------------------------------------
        self.baseline = None
        self.state = "IDLE"  # IDLE | FOCUSED | WARNING | ALERT | RECOVERING
        self.fatigue_index = 0.0
        self.attention_score = 100.0

        self.ear_buffer = []
        self.yaw_buffer = []
        self.mar_consecutive = 0

        self.consecutive_bad_frames = 0
        self.consecutive_good_frames = 0

        self.phone_detected = False
        self.current_alert = None

        self._yolo_frame_skip = 0  # run YOLO every N frames to save CPU

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def compute_ear(self, landmarks_px, eye_indices):
        p = [landmarks_px[i] for i in eye_indices]
        vertical_1 = euclidean(p[1], p[5])
        vertical_2 = euclidean(p[2], p[4])
        horizontal = euclidean(p[0], p[3])
        if horizontal == 0:
            return 0.3
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def compute_ear_avg(self, landmarks_px):
        left = self.compute_ear(landmarks_px, LEFT_EYE)
        right = self.compute_ear(landmarks_px, RIGHT_EYE)
        return (left + right) / 2.0

    def compute_mar(self, landmarks_px):
        vertical = euclidean(landmarks_px[TOP_LIP], landmarks_px[BOTTOM_LIP])
        horizontal = euclidean(landmarks_px[MOUTH_LEFT], landmarks_px[MOUTH_RIGHT])
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def estimate_head_pose(self, landmarks_px, frame_shape):
        h, w = frame_shape[:2]
        image_points = np.array([
            landmarks_px[HP_NOSE_TIP],
            landmarks_px[HP_CHIN],
            landmarks_px[HP_LEFT_EYE],
            landmarks_px[HP_RIGHT_EYE],
            landmarks_px[HP_LEFT_MOUTH],
            landmarks_px[HP_RIGHT_MOUTH],
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    @staticmethod
    def _landmarks_to_px(face_landmarks, w, h):
        return [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

    # ------------------------------------------------------------------
    # Baseline calibration
    # ------------------------------------------------------------------
    def record_baseline(self, frames):
        """
        frames: list of BGR numpy arrays (~30 frames captured over ~5s).
        Returns the baseline dict on success, or None if calibration failed
        (fewer than 5 frames had a detectable face).
        """
        face_y_list, slope_list, pitch_list, n2s_list, ear_list = [], [], [], [], []

        for frame in frames:
            try:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(rgb)
                pose_results = self.pose.process(rgb)

                if not face_results.multi_face_landmarks:
                    continue

                face_landmarks = face_results.multi_face_landmarks[0]
                lm_px = self._landmarks_to_px(face_landmarks, w, h)

                nose = lm_px[NOSE_TIP]
                chin = lm_px[CHIN]
                forehead = lm_px[FOREHEAD]

                face_y_norm = nose[1] / h
                pitch_deg = math.degrees(math.atan2(forehead[1] - chin[1], forehead[0] - chin[0]))
                ear_avg = self.compute_ear_avg(lm_px)

                shoulder_slope = None
                nose_to_shoulder = None
                if pose_results.pose_landmarks:
                    plm = pose_results.pose_landmarks.landmark
                    l_sh = (plm[LEFT_SHOULDER].x * w, plm[LEFT_SHOULDER].y * h)
                    r_sh = (plm[RIGHT_SHOULDER].x * w, plm[RIGHT_SHOULDER].y * h)
                    shoulder_slope = abs(l_sh[1] - r_sh[1]) / w
                    mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
                    nose_to_shoulder = euclidean(nose, mid) / h

                face_y_list.append(face_y_norm)
                pitch_list.append(pitch_deg)
                ear_list.append(ear_avg)
                if shoulder_slope is not None:
                    slope_list.append(shoulder_slope)
                if nose_to_shoulder is not None:
                    n2s_list.append(nose_to_shoulder)
            except Exception:
                continue

        if len(face_y_list) < 5:
            return None

        baseline = {
            "face_y_norm": float(np.mean(face_y_list)),
            "shoulder_slope": float(np.mean(slope_list)) if slope_list else 0.02,
            "head_pitch": float(np.mean(pitch_list)),
            "nose_to_shoulder": float(np.mean(n2s_list)) if n2s_list else 0.40,
            "ear_avg": float(np.mean(ear_list)),
        }
        self.baseline = baseline
        self.state = "FOCUSED"
        return baseline

    # ------------------------------------------------------------------
    # Phone detection (YOLO)
    # ------------------------------------------------------------------
    def _detect_phone(self, frame):
        """Returns (phone_detected: bool, boxes: list of (x1,y1,x2,y2,conf))."""
        if self.yolo_model is None:
            return False, []

        # Run YOLO every 3rd frame to keep things fast - phones don't
        # appear/disappear within 300ms, so this is imperceptible.
        self._yolo_frame_skip = (self._yolo_frame_skip + 1) % 3
        if self._yolo_frame_skip != 0 and hasattr(self, "_last_phone_state"):
            return self._last_phone_state

        h, w = frame.shape[:2]
        results = self.yolo_model(frame, verbose=False)
        phone_detected = False
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 67 and conf > 0.5:  # COCO class 67 = cell phone
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_y = (y1 + y2) / 2.0
                    boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
                    # Treat as "in use" if phone center sits in the upper
                    # 2/3 of the frame (i.e. near face/hands, not flat on desk)
                    if center_y < h * (2.0 / 3.0):
                        phone_detected = True

        self._last_phone_state = (phone_detected, boxes)
        return phone_detected, boxes

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _update_state_machine(self, bad_state):
        if bad_state:
            self.consecutive_bad_frames += 1
            self.consecutive_good_frames = 0
        else:
            self.consecutive_good_frames += 1
            self.consecutive_bad_frames = 0

        if self.state in ("FOCUSED", "IDLE"):
            if self.consecutive_bad_frames >= self.WARNING_FRAMES:
                self.state = "WARNING"
        elif self.state == "WARNING":
            if self.consecutive_bad_frames >= self.ALERT_FRAMES:
                self.state = "ALERT"
            elif self.consecutive_good_frames >= self.RECOVER_FRAMES:
                self.state = "RECOVERING"
        elif self.state == "ALERT":
            if self.consecutive_good_frames >= self.RECOVER_FRAMES:
                self.state = "RECOVERING"
        elif self.state == "RECOVERING":
            if self.consecutive_good_frames >= self.FOCUS_RESTORE_FRAMES:
                self.state = "FOCUSED"
            elif self.consecutive_bad_frames >= self.WARNING_FRAMES:
                self.state = "WARNING"

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """
        frame: BGR numpy array (mutated in-place with overlays drawn).
        Returns a result dict describing the current state.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_results = self.face_mesh.process(rgb)
        except Exception:
            face_results = None
        try:
            pose_results = self.pose.process(rgb)
        except Exception:
            pose_results = None

        try:
            phone_detected, phone_boxes = self._detect_phone(frame)
        except Exception:
            phone_detected, phone_boxes = False, []
        self.phone_detected = phone_detected

        # Draw pose skeleton if available (drawn regardless of face presence)
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks,
                self.mp_pose_module.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw phone bounding boxes
        for (x1, y1, x2, y2, conf) in phone_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if not face_results or not face_results.multi_face_landmarks:
            self._draw_status_bar(frame, "NO FACE", (60, 60, 60))
            self._update_state_machine(bad_state=True)
            self.current_alert = None
            return {
                "attention": "No Face",
                "posture": "Unknown",
                "fatigue_index": round(self.fatigue_index, 1),
                "phone_detected": self.phone_detected,
                "alert": None,
                "score": round(self.attention_score, 1),
                "state": self.state,
            }

        face_landmarks = face_results.multi_face_landmarks[0]
        lm_px = self._landmarks_to_px(face_landmarks, w, h)

        self.mp_drawing.draw_landmarks(
            frame, face_landmarks, self.mp_face_mesh_module.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        nose = lm_px[NOSE_TIP]
        face_y_norm = nose[1] / h
        ear_avg = self.compute_ear_avg(lm_px)
        mar = self.compute_mar(lm_px)

        try:
            pitch, yaw, roll = self.estimate_head_pose(lm_px, frame.shape)
        except Exception:
            pitch, yaw, roll = 0.0, 0.0, 0.0

        shoulder_slope = self.baseline["shoulder_slope"] if self.baseline else 0.02
        nose_to_shoulder = self.baseline["nose_to_shoulder"] if self.baseline else 0.40
        if pose_results and pose_results.pose_landmarks:
            plm = pose_results.pose_landmarks.landmark
            l_sh = (plm[LEFT_SHOULDER].x * w, plm[LEFT_SHOULDER].y * h)
            r_sh = (plm[RIGHT_SHOULDER].x * w, plm[RIGHT_SHOULDER].y * h)
            shoulder_slope = abs(l_sh[1] - r_sh[1]) / w
            mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
            nose_to_shoulder = euclidean(nose, mid) / h

        # Rolling buffers (smoothing reduces flicker / false alerts)
        self.ear_buffer.append(ear_avg)
        if len(self.ear_buffer) > 30:
            self.ear_buffer.pop(0)
        ear_rolling = sum(self.ear_buffer) / len(self.ear_buffer)

        self.yaw_buffer.append(yaw)
        if len(self.yaw_buffer) > 30:
            self.yaw_buffer.pop(0)
        yaw_rolling = sum(self.yaw_buffer) / len(self.yaw_buffer)

        # ---- Attention ----------------------------------------------
        if self.baseline is None:
            attention = "Calibrating"
        elif abs(yaw_rolling) > 25:
            attention = "Looking Away"
        elif abs(yaw_rolling) > 15:
            attention = "Distracted"
        else:
            attention = "Focused"

        # ---- Posture ---------------------------------------------------
        if self.baseline is None:
            posture = "Calibrating"
        else:
            head_dropped = (face_y_norm - self.baseline["face_y_norm"]) > 0.10
            slouching = (shoulder_slope - self.baseline["shoulder_slope"]) > 0.06
            hunching = nose_to_shoulder < (self.baseline["nose_to_shoulder"] * 0.82)

            if head_dropped and slouching:
                posture = "Head Dropped"
            elif slouching:
                posture = "Slouching"
            elif hunching:
                posture = "Hunching"
            else:
                posture = "Good"

        # ---- Fatigue index ----------------------------------------------
        fatigue_signals = 0
        if self.baseline and ear_rolling < (self.baseline["ear_avg"] * 0.75):
            fatigue_signals += 1
        if pitch > 15:
            fatigue_signals += 1
        if mar > 0.6:
            self.mar_consecutive += 1
            if self.mar_consecutive >= 2:
                fatigue_signals += 1
        else:
            self.mar_consecutive = 0

        if fatigue_signals >= 2:
            self.fatigue_index = min(100.0, self.fatigue_index + 0.5)
        else:
            self.fatigue_index = max(0.0, self.fatigue_index - 0.3)

        # ---- Attention score ---------------------------------------------
        if attention == "Focused" and posture == "Good":
            self.attention_score = min(100.0, self.attention_score + 0.2)
        elif attention == "Looking Away" or posture == "Head Dropped":
            self.attention_score = max(0.0, self.attention_score - 1.0)
        else:
            self.attention_score = max(0.0, self.attention_score - 0.5)

        # ---- State machine -------------------------------------------
        bad_state = (attention not in ("Focused",)) or (posture not in ("Good",)) or (self.fatigue_index > 60)
        self._update_state_machine(bad_state)

        if self.state == "ALERT" and posture != "Good":
            self.current_alert = "Posture Alert"
        elif self.state == "ALERT" and self.fatigue_index > 60:
            self.current_alert = "Drowsiness Alert"
        elif self.phone_detected:
            self.current_alert = "Phone Detected"
        else:
            self.current_alert = None

        # ---- Draw overlays ---------------------------------------------
        bar_color = {
            "FOCUSED": (0, 200, 0),
            "WARNING": (0, 165, 255),
            "ALERT": (0, 0, 255),
            "RECOVERING": (0, 200, 200),
        }.get(self.state, (50, 50, 50))
        self._draw_status_bar(frame, self.state, bar_color)

        cv2.putText(frame, f"{attention} | {posture}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return {
            "attention": attention,
            "posture": posture,
            "fatigue_index": round(self.fatigue_index, 1),
            "phone_detected": self.phone_detected,
            "alert": self.current_alert,
            "score": round(self.attention_score, 1),
            "state": self.state,
        }

    @staticmethod
    def _draw_status_bar(frame, label, color):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), color, -1)
        cv2.putText(frame, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass
