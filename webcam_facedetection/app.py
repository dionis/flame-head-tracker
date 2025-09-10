import os, sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import subprocess
import json
from google import genai
from google.api_core  import exceptions
from PIL import Image
from io import BytesIO
import uuid
import shutil
from utils.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from utils.face_landmarker_analyzer import FaceLandmarkerAnalyzer, FaceLandmarkerResult
import random
sys.path.append("/teamspace/studios/this_studio/DECA")
from decalib.datasets import datasets

# --- NUEVO: Leer variables de entorno desde archivo .env si existe ---
from dotenv import load_dotenv
load_dotenv()  # Esto cargará las variables de entorno desde un archivo .env si está presente

# --- NUEVO: Leer la API KEY de Gemini desde variable de entorno ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# if GOOGLE_API_KEY:
#     genai.configure(api_key=GOOGLE_API_KEY)
# else:
#     print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini API calls will fail.")

analyzer = FaceAnalyzer()
landmarker_analyzer = FaceLandmarkerAnalyzer()

AVATAR_SCRIPT_PATH = "create_avatar.py"
AVATAR_SCRIPT_PATH = "/teamspace/studios/this_studio/flame-head-tracker/generate_arkit_flame_meshes.py"
AVATAR_SCRIPT_PATH = "flame-head-tracker/generate_arkit_flame_meshes.py"

NEUTRAL_IMAGES_ADDRESS = "/teamspace/studios/this_studio/_neutral_images"
NEUTRAL_IMAGES_ADDRESS = "/teamspace/studios/this_studio/neutral_images"
DEFAULT_3D_MODEL_PATH_ADDRESS = "/teamspace/studios/this_studio/flame-head-tracker/out_arkit_flame/"
DEFAULT_3D_MODEL_PATH = "/teamspace/studios/this_studio/flame-head-tracker/out_arkit_flame/neutral.obj"
DEFAULT_PROCESSIG_IMAGE = "/teamspace/studios/this_studio/_neutral_images/neutral_face__jg6lzhz0cnk_e4752063-ea8f-4deb-ae59-c72ccd49e05d.jpg"

GEMINI_MODEL_NAME = "gemini-2.5-flash-image-preview"

AVATAR_OUTPUT_DIR = "generated_avatars"
os.makedirs(AVATAR_OUTPUT_DIR, exist_ok=True)

MESSAGE_EXCEPTION_NEUTRAL_IMAGES_NOT_EXIST = 'Not exist imagen info in address: '
MESSAGE_ERROR_IN_PROCESS_TRANSFORMED = 'Error in process AI image transformed.'
MESSAGE_NOT_IMAGES_AVATAR = 'Error not image for avatar creations'
MESSAGE_CUOTA_EXCEEDED = 'Gemini API quota exceeded.'


###################################################################
#
# Call Gemini Nano Banana Bibliografy
#    https://gist.github.com/patrickloeber/c4492974c6d625a6a57413810a605b12
#    https://gist.github.com/patrickloeber
#
####################################################################
def transform_image_with_gemini(image_array: np.ndarray, prompt: str) -> np.ndarray:
#Tuple[np.ndarray, Dict[str, Any]]:
    if GOOGLE_API_KEY is None:
        return None
        #, {"error": "Gemini API key not configured. Please set GOOGLE_API_KEY environment variable."}
    if image_array is None and not prompt:
        print("Validate if error")

        return None,
        # {"error": "No image or prompt provided for transformation."}
   

    try:
        # if image_array.dtype != np.uint8:
        #   if image_array.max() <= 1.0:
        #     image_array = (image_array * 255).astype(np.uint8)
        #   else:
        #     image_array = image_array.astype(np.uint8)
    
        # image = Image.fromarray(image_array, mode='RGBA')
        print("Call Gemini API for image transformation")
        client = genai.Client(api_key = GOOGLE_API_KEY)
       # Call the API to generate content        
        if image_array.size == 0:
            response = client.models.generate_content(
                                         model = GEMINI_MODEL_NAME,
                                         contents = prompt,
                                    )
        else:           
            pil_image = Image.fromarray(image_array)           
            # Pass both the text prompt and the image in the 'contents' list
            print("Call Gemini API for image transformation")
            response = client.models.generate_content(
                model = GEMINI_MODEL_NAME,
                contents=[prompt, pil_image],
            )

        generated_image_numpy = None
        # The response can contain both text and image data.
        # Iterate through the parts to find and save the image.
        for part in response.candidates[0].content.parts:
            if part.text is not None:
               text_result = part.text
            elif part.inline_data is not None:
                generated_image_numpy = np.asarray(Image.open(BytesIO(part.inline_data.data)))  


        return generated_image_numpy
        #, {"status": "success", "message": getattr(response, "text", str(response))}
    except exceptions.ResourceExhausted as e:
        raise gr.Error(MESSAGE_CUOTA_EXCEEDED)
    except genai.errors.ClientError as e:
        print(e)
        raise gr.Error( e.message)
    except Exception as e:
        return e
        #, {"error": f"Error transforming image with Gemini: {e}"}

def unavailable_botton():
    # Devuelve el botón con 'interactive=False' inmediatamente
    return gr.Button(interactive=False)
def available_botton():
    # Devuelve el botón con 'interactive=True'
    return gr.Button(interactive=True)

def create_avatar_from_transformed_image(image: np.ndarray, session_id: Optional[str], req: gr.Request) -> Dict[str, Any]:
    if image is None or image.size == 0:
        raise gr.Error( "No transformed image to create avatar from")   
     # Remove existing files in the output directory
    if os.path.exists(NEUTRAL_IMAGES_ADDRESS):       
         
         ##Validate if image has a face and neutral face position
         result = landmarker_analyzer.analyze_image(image=image)
         
         if len(result.face_landmarks) == 0:
            raise gr.Error( "No face found in the image")
         elif len(result.face_landmarks) > 1:
            raise gr.Error( "There are more than one face found in the image")
         elif result.is_neutral_face == False:
            raise gr.Error( "The face is not in the neutral position")

        ##Save images in component in neutral_images
         pil_image = Image.fromarray(image)    
         filename = f"neutral_face_trasformed_{session_id}.jpg"      
         temp_img_path = os.path.join(NEUTRAL_IMAGES_ADDRESS, filename)         

         pil_image.save(temp_img_path)
    
         result = run_avatar_script(temp_img_path, "transformed_image")
    else:
       raise gr.Error(MESSAGE_ERROR_IN_PROCESS_TRANSFORMED)
    #os.remove(temp_img_path)
    return result


def run_avatar_script(input_path: str, input_type: str) -> Dict[str, Any]:
    try:
        
        if not os.path.exists(AVATAR_OUTPUT_DIR):
          os.makedirs(AVATAR_OUTPUT_DIR, exist_ok=True)

        print(f"Input path in file => {input_path}")

        if not os.path.exists(input_path):
          return {"error": f"{MESSAGE_EXCEPTION_NEUTRAL_IMAGES_NOT_EXIST}{AVATAR_OUTPUT_DIR}"} 


        command = ["cd","flame-head-tracker/"]
        print(f"the script is executing in: {os.getcwd()}")
        #result = subprocess.run(command, capture_output=True, text=True, check=True)
        sys.path.append("DECA")
        print("Add DECA libraries in context")
        #sys.path.append("/teamspace/studios/this_studio/flame-head-tracker/submodules/")
        print("Add to libraries space")
        command = ["ls","-l"]
        #result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"<=== Execute task files in directory ====>\n\n")


        
        command = [
            "uv", "run", 
             AVATAR_SCRIPT_PATH,
            "--input_path", input_path,
            "--input_type", input_type,
            "--output_dir", AVATAR_OUTPUT_DIR,
        ]
        #flame-head-tracker

        print("<=== Execute task files ====>")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Execution python script result:\n\n {result.stdout}")
        #output = json.loads(result.stdout)
        return { 'output': 'Call python scripts'}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error running avatar script: {e.stderr}"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON output from script: {result.stdout}"}
    except FileNotFoundError:
        return {"error": f"Avatar script not found at {AVATAR_SCRIPT_PATH}"}


def create_avatar_image(image: np.ndarray, session_id: Optional[str], req: gr.Request ) -> Dict[str, Any]:
    if image is None:
        gr.Error ("error : No frame provided for avatar creation")
    
    # Save the image to a temporary file
    list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
    
    if len(list_of_files) == 0:
        raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    else: #Only a face imafes for get information  
        output_dir = ""
        for file_name in list_of_files:
            if f"neutral_face_{session_id}" in file_name:
                output_dir = file_name
                break
            
        if output_dir == '':
            raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)   
        temp_img_path = os.path.join(NEUTRAL_IMAGES_ADDRESS, output_dir)
        print(f"Avatar create with file images {temp_img_path}")
    
    result = run_avatar_script(temp_img_path, "image")
    #os.remove(temp_img_path)  # Clean up temporary file
    return result


def create_avatar_video(video_path: str, max_dimension: int, session_id: Optional[str], req: gr.Request ) -> Dict[str, Any]:
    list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
    
    if len(list_of_files) == 0:
        raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    else: #Only a face imafes for get information
        output_dir = ""
        for file_name in list_of_files:
             if f"neutral_face_{session_id}" in file_name:
                output_dir = file_name
                break
        
        if output_dir == '':
            raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)   
        temp_frame_path = os.path.join(NEUTRAL_IMAGES_ADDRESS, output_dir)
        print(f"Avatar create with file images {temp_frame_path}")
    
        # For video, the path is already a file path, no need to save temporarily
        result = run_avatar_script(video_path, "video")
        return result


def create_avatar_webcam(frame: np.ndarray, session_id: Optional[str],  req: gr.Request ) -> Dict[str, Any]:
    if frame is None:
        print ("error : No frame provided for avatar creation")
    
    temp_frame_path = ''
    
    # Save the webcam frame to a temporary file
    if os.path.exists(NEUTRAL_IMAGES_ADDRESS):
         list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
         if len(list_of_files) >= 1: #Only a face imafes for get information
            output_dir = ""
            for file_name in list_of_files:
              if f"neutral_face_{session_id}" in file_name:
                output_dir = file_name
                break
           
            if output_dir == '':
              raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)   
            temp_frame_path = os.path.join(NEUTRAL_IMAGES_ADDRESS, output_dir)
            print(f"Avatar create with file images {temp_frame_path}")
    
    if temp_frame_path == None or temp_frame_path == '':
        return {"error": "No frame provided for avatar creation"}
    else:
       result = run_avatar_script(temp_frame_path, "webcam_frame")
    #os.remove(temp_frame_path)  # Clean up temporary file
    return result
SHOWING_FACE = 1

ERROR_MESSAGE_MORE_ONE_FACE = "There are more than one face or none \
    in your picture, please fixed image or video"


def process_image(
    image: np.ndarray,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    refine_landmarks: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if image is None:
        return None, {"error": "No image provided"}

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
        return None, {"error": "No video provided"}

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
        return None, {"error": "No frame"}

    result = analyzer.analyze_image(
        image=frame,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
    )
    return result.annotated_image_rgb, result.to_dict()


def process_landmarker_image(
    image: np.ndarray,
    session_id: Optional[str], 
    req: gr.Request
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if image is None:
        return None, {"error": "No image provided"}
    
    landmarker_analyzer.user_image_filename = f"{session_id}"
    result = landmarker_analyzer.analyze_image(image=image)
    
    SHOWING_FACE = len(result.face_landmarks)
    
    if SHOWING_FACE == 0 or SHOWING_FACE > 1:
       raise gr.Error(ERROR_MESSAGE_MORE_ONE_FACE) 
    
    return result.annotated_image_rgb, {
        "face_landmarks": SHOWING_FACE,
        "expressions": result.expressions,
    }


def process_landmarker_video(
    video_path: str,
    max_dimension: int,
    session_id: Optional[str], 
    req: gr.Request
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not video_path:
        return None, {"error": "No video provided"}

    landmarker_analyzer.user_image_filename = f"{session_id}"
    output_path, aggregate = landmarker_analyzer.process_video_file(
        input_path=video_path, maximum_frame_dimension=max_dimension
    )
    
    frame_sum_by_face = sum(aggregate["faces_detected_per_frame"])
    len_of_frame = len(aggregate["faces_detected_per_frame"])
    
    SHOWING_FACE =  (frame_sum_by_face- len_of_frame)
    
    print(f"Size of faces detected {frame_sum_by_face} !!!")
    print(f"Size of frame {len_of_frame} !!!")
    print(f" Size of face detected computed {SHOWING_FACE} !!!")
    
    if SHOWING_FACE == 0 or SHOWING_FACE > 1:
        raise gr.Error(ERROR_MESSAGE_MORE_ONE_FACE) 
    
    return output_path, aggregate


def process_landmarker_stream(
    frame: np.ndarray,
    session_id: Optional[str],
    req: gr.Request
) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    if frame is None:
        return None, {"error": "No frame"}, False
    
    landmarker_analyzer.user_image_filename = f"{session_id}"

    result = landmarker_analyzer.analyze_image(image=frame)
    
    SHOWING_FACE = len(result.face_landmarks)
    
    if SHOWING_FACE == 0 or SHOWING_FACE > 1:
        gr.Error(ERROR_MESSAGE_MORE_ONE_FACE) 
    
    return result.annotated_image_rgb, {
        "face_landmarks": SHOWING_FACE,
        "expressions": result.expressions,
        "is_neutral_face": result.is_neutral_face,
    }, gr.Label(value="Yes" if result.is_neutral_face else "No", elem_classes=["neutral-face-true"] if result.is_neutral_face else ["neutral-face-false"]), gr.Label(value="Yes" if len(result.face_landmarks) == 1 else "No", elem_classes=["single-face-true"] if len(result.face_landmarks) == 1 else ["single-face-false"])

def clear_components():
    return None, {}, None, None

def delete_directory(req: gr.Request):    
    
    if not req.session_hash:
        return
    
    for f in os.listdir(NEUTRAL_IMAGES_ADDRESS):
        if req.session_hash in f:
            print("Image create in session will be deleted")
            file_to_delete = os.path.join(NEUTRAL_IMAGES_ADDRESS, f)
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
            
    for f in os.listdir(DEFAULT_3D_MODEL_PATH_ADDRESS):
        if req.session_hash in f:
            print("3D model file create in session will be deleted")
            
            file_to_delete = os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS, f)
            if os.path.isfile(file_to_delete):
              os.remove(file_to_delete)
    
    #user_dir: Path = current_dir / str(req.session_hash)
    #shutil.rmtree(str(user_dir))
    #os.remove()
    
    
def start_session(session_id, request: gr.Request):
    if session_id is None:
        session_id =  f"_{request.session_hash}_" + str(uuid.uuid4())  # Crear un ID único
    return f"Session ID: {session_id}", session_id

def check_neutral_image_exist(session_id: str, validate:bool = True) -> np.ndarray | None:
    if not session_id:
       if validate:
        raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
       return None    
    
    if not os.path.exists(NEUTRAL_IMAGES_ADDRESS):
        if validate:
           raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
        return None
    
    list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
    
    if len(list_of_files) == 0:
        if validate:
          raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
        return None
    else: #Only a face imafes for get information
        for file_name in list_of_files:
             if f"neutral_face_{session_id}" in file_name:
                #print("Find images to show") 
                return  np.asarray(Image.open(os.path.join(NEUTRAL_IMAGES_ADDRESS, file_name)))
                #return os.path.join(NEUTRAL_IMAGES_ADDRESS, file_name)
    print("Validate images to show") 
    if validate:
      raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    return None


def check_neutral_image_exist_aux(session_id: str, validate:bool = True) -> np.ndarray | None:
    return np.asarray(Image.open(DEFAULT_PROCESSIG_IMAGE))
    # if not session_id:
    #    if validate:
    #     raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    #    return None    
    
    # if not os.path.exists(NEUTRAL_IMAGES_ADDRESS):
    #     if validate:
    #        raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    #     return None
    
    # list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
    
    # if len(list_of_files) == 0:
    #     if validate:
    #       raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    #     return None
    # else: #Only a face imafes for get information
    #     for file_name in list_of_files:
    #          if f"neutral_face_{session_id}" in file_name:
    #             print("Find images to show") 
    #             return  np.asarray(Image.open(os.path.join(NEUTRAL_IMAGES_ADDRESS, file_name)))
    #             #return os.path.join(NEUTRAL_IMAGES_ADDRESS, file_name)
    # print("Validate images to show") 
    # if validate:
    #   raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    # return None

def check_neutral_3d_image_exist(session_id: str, validate:bool = True) -> str:
    if not session_id:
      if validate:
        raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
      return None
    
    if not os.path.exists(DEFAULT_3D_MODEL_PATH_ADDRESS):
       if validate:
           raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
       return None
    
    list_of_files =  os.listdir(DEFAULT_3D_MODEL_PATH_ADDRESS)
    
    if len(list_of_files) == 0:
        if validate:
          raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
        return None
    else: #Only a face imafes for get information
        for file_name in list_of_files:
             if f"neutral_{session_id}" in file_name:
                return os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS, file_name)
    
    if validate: 
         raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    return None

with gr.Blocks(title="Face Detection with MediaPipe", theme=gr.themes.Soft(), css=".neutral-face-true { background-color: red !important; } .neutral-face-false { background-color: blue !important; } .single-face-true { background-color: green !important; } .single-face-false { background-color: yellow !important; }") as demo:
   
    session_id = gr.State()
    output = gr.Textbox(label="Session ID")
    request = None
    demo.load(start_session, inputs=[session_id], outputs=[output, session_id])
    # gr.Markdown(
    #     """
    #     ### Face Detection with MediaPipe + Gradio
    #     - Upload an image, video or use your webcam to detect faces and visualize facial features.
    #     - Boxes and facial meshes (Face Mesh) are superimposed and basic metrics are reported per face.
    #     """
    # )

    # with gr.Row():
    #     min_det = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confianza mínima de detección")
    #     min_track = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confianza mínima de seguimiento")
    #     refine = gr.Checkbox(value=False, label="Refinar landmarks (iris)")

    # with gr.Tab("Imagen"):
          
    #     with gr.Row():
    #         img_in = gr.Image(type="numpy", label="Imagen", sources=["upload", "clipboard"], image_mode="RGB")
    #         img_out = gr.Image(type="numpy", label="Resultado", interactive=False)
    #     img_json = gr.JSON(label="Métricas")

    #     img_btn = gr.Button("Procesar imagen")
    #     img_btn.click(
    #         fn=process_image,
    #         inputs=[img_in, min_det, min_track, refine],
    #         outputs=[img_out, img_json],
    #     )

    # with gr.Tab("Video"):
    #     with gr.Row():
    #         vid_in = gr.Video(label="Video (mp4, mov, webm)")
    #         vid_out = gr.Video(label="Video anotado")
    #     vid_json = gr.JSON(label="Resumen del video")
    #     max_dim = gr.Slider(480, 1920, value=1280, step=80, label="Máx. dimensión de fotograma (px)")

    #     vid_btn = gr.Button("Procesar video")
    #     vid_btn.click(
    #         fn=process_video,
    #         inputs=[vid_in, min_det, min_track, refine, max_dim],
    #         outputs=[vid_out, vid_json],
    #     )

    # with gr.Tab("Webcam"):
    #     clear_manual_btn = gr.Button("Clear Manually")
        
    #     if not SHOWING_FACE == 0 or SHOWING_FACE > 1:
    #          gr.Error(ERROR_MESSAGE_MORE_ONE_FACE) 
             
    #     with gr.Row():
          
    #         cam_in = gr.Image(
    #             sources=["webcam"],
    #             streaming=True,
    #             type="numpy",
    #             label="Webcam",
    #             image_mode="RGB",
    #         )
    #         cam_out = gr.Image(type="numpy", label="Salida", interactive=False)
    #     cam_json = gr.JSON(label="Métricas (en vivo)")

    #     cam_in.stream(
    #         fn=process_stream,
    #         inputs=[cam_in, min_det, min_track, refine],
    #         outputs=[cam_out, cam_json],
    #     )

    #     clear_manual_btn.click(clear_components, inputs=[], outputs=[cam_out, cam_json])



    with gr.Tab("Face Landmarker") as faceLandmarker_tab:           
        #faceLandmarker_tab.select(start_session, inputs=[session_id], outputs=[output, session_id])  
             
        gr.Markdown(
            """
            ### Face Landmark Detection with MediaPipe Face Landmarker
            - Upload an image, video or use your webcam to detect facial landmarks and estimate expressions.
            - Facial landmarks are superimposed and estimated expressions are reported.
            """
        )
        with gr.Tab("Imagen Landmarker"):
            with gr.Row():
                land_img_in = gr.Image(type="numpy", label="Imagen", sources=["upload", "clipboard"], image_mode="RGB")
                land_img_out = gr.Image(type="numpy", label="Resultado", interactive=False)
            land_img_json = gr.JSON(label="Métricas")

            land_img_btn = gr.Button("Process image with Landmarker")
            land_img_btn.click(
                fn=process_landmarker_image,
                inputs=[land_img_in, session_id],
                outputs=[land_img_out, land_img_json],
            )
            
            create_avatar_img_btn = gr.Button("Create avatar")
            create_avatar_img_btn.click(
                fn=create_avatar_image,
                inputs=[land_img_in, session_id],
                outputs=[land_img_json],
            )
        with gr.Tab("Video Landmarker"):
            with gr.Row():
                land_vid_in = gr.Video(label="Video (mp4, mov, webm)")
                land_vid_out = gr.Video(label="Annotated video")
            land_vid_json = gr.JSON(label="Video summary")
            land_max_dim = gr.Slider(480, 1920, value=1280, step=80, label="Max. frame dimension (px)")

            land_vid_btn = gr.Button("Process video with Landmarker")
            land_vid_btn.click(
                fn=process_landmarker_video,
                inputs=[land_vid_in, land_max_dim, session_id],
                outputs=[land_vid_out, land_vid_json],
            )
            create_avatar_vid_btn = gr.Button("Create avatar")
            create_avatar_vid_btn.click(
                fn=create_avatar_video,
                inputs=[land_vid_in, land_max_dim, session_id],
                outputs=[land_vid_json],
            )
        
        with gr.Tab("Webcam Landmarker"):
            clear_manual_btn = gr.Button("Clear Manually")
            with gr.Row():
          
                land_cam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                    image_mode="RGB",
                )
                land_cam_out = gr.Image(type="numpy", label="Salida", interactive=False)
                
            land_single_face_label = gr.Label(label="Single Face Detected", value="No", show_label=True)
          
            land_cam_json = gr.JSON(label="Métricas (en vivo)")
            #land_neutral_label = gr.Label(label="Neutral Face Detected", value="No", show_label=True)
            
            land_cam_in.stream(
                fn=process_landmarker_stream,
                inputs=[land_cam_in, session_id],
                 outputs=[land_cam_out, land_cam_json,  land_single_face_label],
                #outputs=[land_cam_out, land_cam_json, land_neutral_label, land_single_face_label],
            )
            # land_cam_in.release(
            #     fn=clear_components,
            #   ,
            # )
            clear_manual_btn.click(clear_components, inputs=[],  outputs=[land_cam_out, land_cam_json, land_single_face_label])

            create_avatar_webcam_btn = gr.Button("Create avatar")
            create_avatar_webcam_btn.click(
                fn=create_avatar_webcam,
                inputs=[land_cam_in, session_id],
                outputs=[land_cam_json],
            )

    with gr.Tab("Visualizador 3D") as threeDVisualizer_tab: 
        #threeDVisualizer_tab.select(check_neutral_3d_image_exist, inputs=[session_id], outputs=[])  
     
        gr.Markdown(
            """
            ### 3D Models Visualizer
            - Upload a 3D file (OBJ, GLTF/GLB, STL) to view it in the browser.
            - Supports models with textures.
            """
        )
        with gr.Row():
            if os.path.exists(DEFAULT_3D_MODEL_PATH):
              print("3D model exist")

            model_in = gr.Model3D(
                label="3D model",
                interactive=True,
                value = check_neutral_3d_image_exist(session_id, False),
            )
            # Add a file upload component for users to upload their own 3D models
            #file_upload = gr.File(label="Upload your own 3D model (OBJ, GLTF/GLB, STL)")

            #file_upload.upload(lambda x: x, inputs=file_upload, outputs=model_in)

    with gr.Tab("Image Transformer") as imageTransformer_tab:
      
     
        with gr.Row():
            with gr.Column():
               neutral_image_path = NEUTRAL_IMAGES_ADDRESS + os.sep + f"neutral_face_{session_id}.jpg"
               print ("Neutral image path for transform => ", neutral_image_path)
              
            #    img_transform_in = gr.Image(
            #                             type ="numpy", 
            #                             label ="Input Image", 
            #                             sources =["upload", "clipboard"], 
            #                             image_mode="RGB",                                    
            #                         )
                    
               img_transform_in = gr.Image(
                                        type ="numpy", 
                                        label ="Input Image", 
                                        sources =[], 
                                        image_mode="RGB",                                    
                                    )
               img_transform_prompt = gr.Textbox(label="Prompt", placeholder="Describe the transformation...")
               
               imageTransformer_tab.select(check_neutral_image_exist_aux, inputs=[session_id], outputs=[img_transform_in])
            #    if not os.path.exists(neutral_image_path):
            #       raise gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
              
            with gr.Column():        
                img_transform_out = gr.Image(type="numpy", label="Transformed Image", interactive=False)

        with gr.Row():
            generate_image_btn = gr.Button("Generate Image")
            create_avatar_transformed_btn = gr.Button("Create Avatar")

        generate_image_btn.click(
            fn=transform_image_with_gemini,
            inputs=[img_transform_in, img_transform_prompt],
            outputs=[img_transform_out],
        )

        create_avatar_transformed_btn.click(
                fn = unavailable_botton,
                inputs = None,
                outputs = generate_image_btn,
                # 'queue=False' asegura que esta acción se ejecute de inmediato, no en la cola
                queue=False
        ).then(  
                fn = create_avatar_from_transformed_image,
                inputs=[img_transform_out,session_id],
                outputs=[],
                queue=True
        ).then(
            fn= available_botton,
            inputs = None,
            outputs = generate_image_btn,
            queue =False
        )  

    ## Free and delete user directory when the user close the application
    #
    # Bibliografy:
    #       https://www.gradio.app/guides/resource-cleanup
    #
    #      Request object: https://www.gradio.app/docs/gradio/request
    demo.unload(delete_directory)
    
if __name__ == "__main__":
    demo.launch(share=True)


