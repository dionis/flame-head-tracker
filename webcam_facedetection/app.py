import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from utils.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from utils.face_landmarker_analyzer import FaceLandmarkerAnalyzer, FaceLandmarkerResult


analyzer = FaceAnalyzer()
landmarker_analyzer = FaceLandmarkerAnalyzer()


def process_image(
    image: np.ndarray,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    refine_landmarks: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if image is None:
        return None, {"error": "No se proporcionó imagen"}

    result = analyzer.analyze_image(
        image=image,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
    )
    return result.annotated_image_rgb, result.to_dict()


def process_video(
    video_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    refine_landmarks: bool,
    max_dimension: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not video_path:
        return None, {"error": "No se proporcionó video"}

    output_path, aggregate = analyzer.process_video_file(
        input_path=video_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
        maximum_frame_dimension=max_dimension,
    )
    return output_path, aggregate


def process_stream(
    frame: np.ndarray,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    refine_landmarks: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if frame is None:
        return None, {"error": "Sin fotograma"}

    result = analyzer.analyze_image(
        image=frame,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
    )
    return result.annotated_image_rgb, result.to_dict()


def process_landmarker_image(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if image is None:
        return None, {"error": "No se proporcionó imagen"}

    result = landmarker_analyzer.analyze_image(image=image)
    return result.annotated_image_rgb, {
        "face_landmarks": len(result.face_landmarks),
        "expressions": result.expressions,
    }


def process_landmarker_video(
    video_path: str,
    max_dimension: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not video_path:
        return None, {"error": "No se proporcionó video"}

    output_path, aggregate = landmarker_analyzer.process_video_file(
        input_path=video_path, maximum_frame_dimension=max_dimension
    )
    return output_path, aggregate


def process_landmarker_stream(
    frame: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    if frame is None:
        return None, {"error": "Sin fotograma"}, False

    result = landmarker_analyzer.analyze_image(image=frame)
    return result.annotated_image_rgb, {
        "face_landmarks": len(result.face_landmarks),
        "expressions": result.expressions,
        "is_neutral_face": result.is_neutral_face,
    }, gr.Label(value="Yes" if result.is_neutral_face else "No", elem_classes=["neutral-face-true"] if result.is_neutral_face else ["neutral-face-false"]), gr.Label(value="Yes" if len(result.face_landmarks) == 1 else "No", elem_classes=["single-face-true"] if len(result.face_landmarks) == 1 else ["single-face-false"])

def clear_components():
    return None, {}, None, None
with gr.Blocks(title="Detección de Rostros con MediaPipe", theme=gr.themes.Soft(), css=".neutral-face-true { background-color: red !important; } .neutral-face-false { background-color: blue !important; } .single-face-true { background-color: green !important; } .single-face-false { background-color: yellow !important; }") as demo:
    gr.Markdown(
        """
        ### Detección de Rostros con MediaPipe + Gradio
        - Sube una imagen, un video o usa tu webcam para detectar rostros y visualizar características faciales.
        - Se superponen cajas y mallas faciales (Face Mesh) y se reportan métricas básicas por rostro.
        """
    )

    with gr.Row():
        min_det = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confianza mínima de detección")
        min_track = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confianza mínima de seguimiento")
        refine = gr.Checkbox(value=False, label="Refinar landmarks (iris)")

    with gr.Tab("Imagen"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Imagen", sources=["upload", "clipboard"], image_mode="RGB")
            img_out = gr.Image(type="numpy", label="Resultado", interactive=False)
        img_json = gr.JSON(label="Métricas")

        img_btn = gr.Button("Procesar imagen")
        img_btn.click(
            fn=process_image,
            inputs=[img_in, min_det, min_track, refine],
            outputs=[img_out, img_json],
        )

    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label="Video (mp4, mov, webm)")
            vid_out = gr.Video(label="Video anotado")
        vid_json = gr.JSON(label="Resumen del video")
        max_dim = gr.Slider(480, 1920, value=1280, step=80, label="Máx. dimensión de fotograma (px)")

        vid_btn = gr.Button("Procesar video")
        vid_btn.click(
            fn=process_video,
            inputs=[vid_in, min_det, min_track, refine, max_dim],
            outputs=[vid_out, vid_json],
        )

    with gr.Tab("Webcam"):
        clear_manual_btn = gr.Button("Clear Manually")
        with gr.Row():
            cam_in = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Webcam",
                image_mode="RGB",
            )
            cam_out = gr.Image(type="numpy", label="Salida", interactive=False)
        cam_json = gr.JSON(label="Métricas (en vivo)")

        cam_in.stream(
            fn=process_stream,
            inputs=[cam_in, min_det, min_track, refine],
            outputs=[cam_out, cam_json],
        )
        # cam_in.release(
        #     fn=clear_components,
        #     outputs=[cam_out, cam_json],
        # )
        clear_manual_btn.click(clear_components, inputs=[], outputs=[cam_out, cam_json])



    with gr.Tab("Face Landmarker"):
        gr.Markdown(
            """
            ### Detección de Landmarks Faciales con MediaPipe Face Landmarker
            - Sube una imagen, un video o usa tu webcam para detectar landmarks faciales y estimar expresiones.
            - Se superponen los landmarks faciales y se reportan las expresiones estimadas.
            """
        )
        with gr.Tab("Imagen Landmarker"):
            with gr.Row():
                land_img_in = gr.Image(type="numpy", label="Imagen", sources=["upload", "clipboard"], image_mode="RGB")
                land_img_out = gr.Image(type="numpy", label="Resultado", interactive=False)
            land_img_json = gr.JSON(label="Métricas")

            land_img_btn = gr.Button("Procesar imagen con Landmarker")
            land_img_btn.click(
                fn=process_landmarker_image,
                inputs=[land_img_in],
                outputs=[land_img_out, land_img_json],
            )

        with gr.Tab("Video Landmarker"):
            with gr.Row():
                land_vid_in = gr.Video(label="Video (mp4, mov, webm)")
                land_vid_out = gr.Video(label="Video anotado")
            land_vid_json = gr.JSON(label="Resumen del video")
            land_max_dim = gr.Slider(480, 1920, value=1280, step=80, label="Máx. dimensión de fotograma (px)")

            land_vid_btn = gr.Button("Procesar video con Landmarker")
            land_vid_btn.click(
                fn=process_landmarker_video,
                inputs=[land_vid_in, land_max_dim],
                outputs=[land_vid_out, land_vid_json],
            )
        
        with gr.Tab("Webcam Landmarker"):
            
            with gr.Row():
                land_cam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                    image_mode="RGB",
                )
                land_cam_out = gr.Image(type="numpy", label="Salida", interactive=False)
            land_cam_json = gr.JSON(label="Métricas (en vivo)")
            #land_neutral_label = gr.Label(label="Neutral Face Detected", value="No", show_label=True)
            land_single_face_label = gr.Label(label="Single Face Detected", value="No", show_label=True)
            
            land_cam_in.stream(
                fn=process_landmarker_stream,
                inputs=[land_cam_in],
                 outputs=[land_cam_out, land_cam_json,  land_single_face_label],
                #outputs=[land_cam_out, land_cam_json, land_neutral_label, land_single_face_label],
            )
            # land_cam_in.release(
            #     fn=clear_components,
            #   ,
            # )
            clear_manual_btn.click(clear_components, inputs=[],  outputs=[land_cam_out, land_cam_json, land_single_face_label])



    with gr.Tab("3D viewer"):
        gr.Markdown(
            """
            ###3D Model Viewer
            - Upload a 3D file (OBJ, GLTF/GLB, STL) to view in the browser.
            - Supports models with textures.
            """
        )
        with gr.Row():
             # Replace with your converted file
             #oficial_path 
             gbl_oficial_path = "/teamspace/studios/this_studio/flame-head-tracker/out_arkit_flame/neutral.obj"
             
             model_in =  gr.Model3D( value = gbl_oficial_path, # Replace with your converted file
                                     label="My 3D Model",
                                     clear_color=(0, 0, 0, 0) # Example: Transparent background
                     ) if os.path.exists(gbl_oficial_path)  else gr.Model3D(label="Modelo 3D", interactive=True)
            # No explicit output needed for gr.Model3D as it's a viewer


if __name__ == "__main__":
    demo.launch(share=True)


