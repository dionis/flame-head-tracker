from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import landmark_pb2


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class FaceMetrics:
    bbox: Tuple[int, int, int, int]
    score: float
    interocular_distance_px: Optional[float]


@dataclass
class FaceAnalysisResult:
    annotated_image_rgb: np.ndarray
    faces: List[FaceMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_faces": len(self.faces),
            "faces": [
                {
                    "bbox": {
                        "x": f.bbox[0],
                        "y": f.bbox[1],
                        "w": f.bbox[2],
                        "h": f.bbox[3],
                    },
                    "score": round(float(f.score), 4),
                    "interocular_distance_px": None if f.interocular_distance_px is None else round(float(f.interocular_distance_px), 2),
                }
                for f in self.faces
            ],
        }


class FaceAnalyzer:
    def __init__(self) -> None:
        self._face_mesh_spec = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _compute_interocular_distance(self, landmarks: List[landmark_pb2.NormalizedLandmark], width: int, height: int) -> Optional[float]:
        try:
            left_idx, right_idx = 33, 263
            left = landmarks[left_idx]
            right = landmarks[right_idx]
            lx, ly = int(left.x * width), int(left.y * height)
            rx, ry = int(right.x * width), int(right.y * height)
            return float(np.hypot(lx - rx, ly - ry))
        except Exception:
            return None

    def analyze_image(
        self,
        image: np.ndarray,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = False,
    ) -> FaceAnalysisResult:
        rgb = image if image.ndim == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Face detection for bounding boxes
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence) as detector:
            det = detector.process(rgb)

        # Face mesh for detailed landmarks
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as mesh:
            mesh_res = mesh.process(rgb)

        annotated = rgb.copy()
        faces: List[FaceMetrics] = []

        # Draw detections
        if det and det.detections:
            for detection in det.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)
                x, y = max(0, x), max(0, y)
                w_box = max(0, min(w_box, w - x))
                h_box = max(0, min(h_box, h - y))
                cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                score = float(detection.score[0]) if detection.score else 0.0
                faces.append(FaceMetrics(bbox=(x, y, w_box, h_box), score=score, interocular_distance_px=None))

        # Draw mesh and compute metrics
        if mesh_res and mesh_res.multi_face_landmarks:
            for idx, face_landmarks in enumerate(mesh_res.multi_face_landmarks):
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                if refine_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

                interocular = self._compute_interocular_distance(face_landmarks.landmark, w, h)
                if idx < len(faces):
                    faces[idx].interocular_distance_px = interocular
                else:
                    faces.append(FaceMetrics(bbox=(0, 0, 0, 0), score=0.0, interocular_distance_px=interocular))

        return FaceAnalysisResult(annotated_image_rgb=annotated, faces=faces)

    def process_video_file(
        self,
        input_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = False,
        maximum_frame_dimension: int = 1280,
    ) -> Tuple[str, Dict[str, Any]]:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = 1.0
        if max(width, height) > maximum_frame_dimension:
            scale = maximum_frame_dimension / float(max(width, height))
        out_w, out_h = int(width * scale), int(height * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = input_path + ".annotated.mp4"
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_faces = 0
        max_faces_in_frame = 0

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence) as detector:
            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ) as mesh:
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if scale != 1.0:
                        frame_rgb = cv2.resize(frame_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

                    det = detector.process(frame_rgb)
                    mesh_res = mesh.process(frame_rgb)

                    annotated = frame_rgb.copy()
                    faces_in_frame = 0

                    if det and det.detections:
                        for detection in det.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            x = int(bboxC.xmin * out_w)
                            y = int(bboxC.ymin * out_h)
                            w_box = int(bboxC.width * out_w)
                            h_box = int(bboxC.height * out_h)
                            x, y = max(0, x), max(0, y)
                            w_box = max(0, min(w_box, out_w - x))
                            h_box = max(0, min(h_box, out_h - y))
                            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                            faces_in_frame += 1

                    if mesh_res and mesh_res.multi_face_landmarks:
                        for face_landmarks in mesh_res.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=annotated,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                            )
                            mp_drawing.draw_landmarks(
                                image=annotated,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                            )
                            if refine_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=annotated,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_IRISES,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                                )

                    writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    total_faces += faces_in_frame
                    max_faces_in_frame = max(max_faces_in_frame, faces_in_frame)

        cap.release()
        writer.release()

        aggregate = {
            "frames": total_frames,
            "avg_faces_per_frame": round((total_faces / total_frames) if total_frames else 0.0, 3),
            "max_faces_in_frame": int(max_faces_in_frame),
            "output_path": out_path,
        }
        return out_path, aggregate


