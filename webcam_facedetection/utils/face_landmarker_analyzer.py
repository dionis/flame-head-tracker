
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Optional[Tuple[int, int]]:
    # Checks if the normalized coordinates are valid.
    if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
        return None
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class FaceLandmarkerResult(NamedTuple):
    annotated_image_rgb: np.ndarray
    face_landmarks: List[Any]
    blendshapes: List[Any]
    expressions: Dict[str, float]
    pose_transform: Optional[np.ndarray]
    is_neutral_face: bool


class FaceLandmarkerAnalyzer:
    def __init__(self, model_path: str = f"models{os.sep}face_landmarker_v2_with_blendshapes.task"):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        self.detector = self._load_detector()
        
        #A neutral face usually appears when most expression blendshape scores are close to zero, 
        # i.e., no significant activation of smile, frown, eyebrow raise, surprise, etc.
        
        self.threshold = 0.5
        #float(1/5)
        



    def _load_detector(self):
        
        #Bibliofrafy about solutions
        # Mediapipe runtime error whilst following the Hand_landmarker guide. Unable to initialize runfiles
        # https://stackoverflow.com/questions/75985134/mediapipe-runtime-error-whilst-following-the-hand-landmarker-guide-unable-to-in
        
        
        
        model_file = open(self.model_path, "rb")
        model_data = model_file.read()
        model_file.close()
        
        base_options = python.BaseOptions(model_asset_buffer = model_data)
        #base_options = python.BaseOptions(model_asset_path=self.model_path)
 
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def analyze_image(
        self,
        image: np.ndarray,
    ) -> FaceLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)

        annotated_image = np.copy(image)
        face_landmarks_list = detection_result.face_landmarks
        blendshapes = detection_result.face_blendshapes
        pose_transform = detection_result.facial_transformation_matrixes

        # Draw the face landmarks.
        for face_landmarks in face_landmarks_list:
            # Draw the facial landmarks with connections.
            for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                start_node = connection[0]
                end_node = connection[1]
                start_point = _normalized_to_pixel_coordinates(
                    face_landmarks[start_node].x,
                    face_landmarks[start_node].y,
                    image.shape[1],
                    image.shape[0],
                )
                end_point = _normalized_to_pixel_coordinates(
                    face_landmarks[end_node].x,
                    face_landmarks[end_node].y,
                    image.shape[1],
                    image.shape[0],
                )
                if start_point and end_point:
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 1)

        expressions = {}
        is_neutral_face = False
        
        if blendshapes and blendshapes[0]:           
            for category in blendshapes[0]:
                expressions[category.category_name] = category.score
                
            #We exclude the eye_open category because it is not a facial expression
            exclude_categories = ["cheekSquintLeft", "cheekSquintRight", "_neutral", "noseSneerLeft", "noseSneerRight"]
            is_neutral_face = self.is_neutral( [ element for element in blendshapes[0] if element.category_name not in exclude_categories])  
        else:
            is_neutral_face = False

        return FaceLandmarkerResult(
            annotated_image_rgb = annotated_image,
            face_landmarks = face_landmarks_list,
            blendshapes = blendshapes,
            expressions = expressions,
            pose_transform=pose_transform[0] if pose_transform else None,
            is_neutral_face = is_neutral_face,
        )

    def process_video_file(
        self,
        input_path: str,
        maximum_frame_dimension: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        output_path = "output_video.mp4"  # You might want to make this dynamic or temp
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        aggregate_results = {
            "total_frames": 0,
            "faces_detected_per_frame": [],
            "expressions_per_frame": [],
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            aggregate_results["total_frames"] += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.analyze_image(frame_rgb)
            
            aggregate_results["faces_detected_per_frame"].append(len(result.face_landmarks))
            aggregate_results["expressions_per_frame"].append(result.expressions)

            out.write(cv2.cvtColor(result.annotated_image_rgb, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        return output_path, aggregate_results


    def is_neutral(self, blendshapes: List[Any]) -> bool:
        return all(abs(bs.score) < self.threshold for bs in blendshapes)