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
from utils_tool.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from utils_tool.face_landmarker_analyzer import FaceLandmarkerAnalyzer, FaceLandmarkerResult
import random
sys.path.append("/teamspace/studios/this_studio/DECA")
from decalib.datasets import datasets
import time
sys.path.append("/teamspace/studios/this_studio/flame-head-tracker")
#sys.path.append("/teamspace/studios/this_studio/flame-head-tracker/utils")

from generate_arkit_flame_meshes import Tracker3DImage

import requests # Import requests
import base64 # Import base64 for decoding
import io # Import io for BytesIO
import soundfile as sf # Import soundfile for audio processing
from pydub import AudioSegment # Import pydub for audio manipulation

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


### 2d files and 3D object to transforms

converter = Tracker3DImage()

AVATAR_SCRIPT_PATH = "create_avatar.py"
AVATAR_SCRIPT_PATH = "/teamspace/studios/this_studio/flame-head-tracker/generate_arkit_flame_meshes.py"
AVATAR_SCRIPT_PATH = "flame-head-tracker/generate_arkit_flame_meshes.py"

NEUTRAL_IMAGES_ADDRESS = "/teamspace/studios/this_studio/flame-head-tracker/_neutral_images"
NEUTRAL_IMAGES_ADDRESS = "/teamspace/studios/this_studio/flame-head-tracker/neutral_images"
DEFAULT_3D_MODEL_PATH_ADDRESS = "/teamspace/studios/this_studio/flame-head-tracker/out_arkit_flame/"
DEFAULT_3D_MODEL_PATH = "/teamspace/studios/this_studio/flame-head-tracker/out_arkit_flame/neutral.obj"
DEFAULT_PROCESSIG_IMAGE = "/teamspace/studios/this_studio/_neutral_images/neutral_face__jg6lzhz0cnk_e4752063-ea8f-4deb-ae59-c72ccd49e05d.jpg"

DEFAULT_2D_to_3D_MODEL_PATHS = "E:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/webcam_facedetection/webcam_facedetection/neutral_images/test_charcter.jpg"
DEFAULT_3D_MODEL_PATH_ADDRESS_AI_TOOLS = ""

GEMINI_MODEL_NAME = "gemini-2.5-flash-image-preview"

AVATAR_OUTPUT_DIR = "generated_avatars"
os.makedirs(AVATAR_OUTPUT_DIR, exist_ok=True)

MESSAGE_EXCEPTION_NEUTRAL_IMAGES_NOT_EXIST = 'Not exist imagen info in address: '
MESSAGE_ERROR_IN_PROCESS_TRANSFORMED = 'Error in process AI image transformed.'
MESSAGE_NOT_IMAGES_AVATAR = 'Error not image for avatar creations'
MESSAGE_CUOTA_EXCEEDED = 'Gemini API quota exceeded.'
SHOWING_FACE = 1

ERROR_MESSAGE_MORE_ONE_FACE = "There are more than one face or none \
    in your picture, please fixed image or video"

ERROR_MESSAGE_IMAGE_NOT_NEUTRAL = "Not image in neutral face position"

streaming_active = True

# --- NUEVO: Leer la API KEY de Gemini desde variable de entorno ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MESHY_API_KEY = os.environ.get("MESHY_API_KEY") # Get Meshy API key
STABILITYAI_API_KEY = os.environ.get("STABILITYAI_API_KEY") # Get Stability AI API key

CUBE_BY_CSM_API_KEY = os.environ.get("CUBE_BY_CSM_API_KEY") # Get Stability AI API key

###################################################################
#
# Call Gemini Nano Banana Bibliografy
#    https://gist.github.com/patrickloeber/c4492974c6d625a6a57413810a605b12
#    https://gist.github.com/patrickloeber
#
####################################################################
def transform_image_with_gemini(image_array: np.ndarray, prompt: str) -> np.ndarray:
#Tuple[np.ndarray, Dict[str, Any]]:
    print(f"The Google API KEY {GOOGLE_API_KEY}")
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
        #  elif result.is_neutral_face == False:
        #     raise gr.Error( "The face is not in the neutral position")

        ##Save images in component in neutral_images
         pil_image = Image.fromarray(image)    
         filename = f"neutral_face_{session_id}.jpg"      
         temp_img_path = os.path.join(NEUTRAL_IMAGES_ADDRESS, filename)  

         if os.path.exists(temp_img_path):
            os.remove(temp_img_path)     

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


        # command = ["cd","flame-head-tracker/"]
        # print(f"the script is executing in: {os.getcwd()}")
        # #result = subprocess.run(command, capture_output=True, text=True, check=True)
        # sys.path.append("DECA")
        # print("Add DECA libraries in context")
        # #sys.path.append("/teamspace/studios/this_studio/flame-head-tracker/submodules/")
        # print("Add to libraries space")
        # command = ["ls","-l"]
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
        #result = subprocess.run(command, capture_output=True, text=True, check=True)
        start_time = time.time() 

        result = converter.process_image_to_3d_object(
            input_path = input_path,
            input_type = input_type,
            output_dir = AVATAR_OUTPUT_DIR
        ) 
      
        end_time = time.time() # record the end time

        elapsed_time = end_time - start_time # calculate the elapsed time
        print(f"time taken to run load_image_and_run: {elapsed_time:.2f} seconds\n\n") # print the elapsed time

        #print(f"Execution python script result:\n\n {result.stdout}")
        print(f"Execution python script result:\n\n {result}")
        #output = json.loads(result.stdout)
        return { 'output': 'Call python scripts', 'process_time': f"{elapsed_time:.2f} seconds", 'fbx_file_address': result}
    except subprocess.CalledProcessError as e:
        return {"error": f"Error running avatar script: {e.stderr}"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON output from script: {result.stdout}"}
    except FileNotFoundError:
        return {"error": f"Avatar script not found at {AVATAR_SCRIPT_PATH}"}


def create_avatar_image(image: np.ndarray, session_id: Optional[str], req: gr.Request ) -> tuple[Dict[str, Any], Any]:
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
    return result, result['fbx_file_address']


def create_avatar_video(video_path: str, max_dimension: int, session_id: Optional[str], req: gr.Request ) -> tuple[Dict[str, Any], Any]:
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
        return result, result['fbx_file_address']


def create_avatar_webcam(frame: np.ndarray, session_id: Optional[str],  req: gr.Request ) -> tuple[Dict[str, Any], Any]:
    if frame is None:
        print ("error : No frame provided for avatar creation")
    
    temp_frame_path = ''
    global streaming_active
    streaming_active = False
    
    # Save the webcam frame to a temporary file
    if os.path.exists(NEUTRAL_IMAGES_ADDRESS):
         list_of_files =  os.listdir(NEUTRAL_IMAGES_ADDRESS)
         if len(list_of_files) >= 1: #Only a face for get information
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
    return result, result['fbx_file_address']

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

    if not result.is_neutral_face:
        raise gr.Error(ERROR_MESSAGE_IMAGE_NOT_NEUTRAL) 
    
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

    if not streaming_active:
        return None, {}, None
    else:
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
    global streaming_active
    streaming_active = False
    
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
            if os.path.isdir(file_to_delete):
                shutil.rmtree(file_to_delete)
    
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
                print(f"find 3D objfile {file_name}")
                return os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS, file_name)
             else:
                print(f"File with not patern {file_name}")
    
    if validate: 
         raise  gr.Error(MESSAGE_NOT_IMAGES_AVATAR)
    return None

####
#   Meshy Image to 3D API
#      https://docs.meshy.ai/en/api/image-to-3d
#    
#    Image to 3D API is a feature that allows you to integrate Meshy's Image to 3D 
#    capabilities into your own application. In this section, you'll find all the 
#    information you need to get started with this API.
#
###

def transform_image_with_meshy(image: np.ndarray, prompt: str) -> str:
    if MESHY_API_KEY is None:
        raise gr.Error("Meshy API key not configured. Please set MESHY_API_KEY environment variable.")
    
    if image is None:
        raise gr.Error("No image provided for Meshy transformation.")

    # Save the input image to a temporary file
    temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(temp_img_file.name)
    temp_img_file.close()

    base_url = "https://api.meshy.ai"
    endpoint = "/v1/image-to-3d"

    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
    }

    
    payload = {
        # Using data URI example
        # image_url: f'data:image/png;base64,{YOUR_BASE64_ENCODED_IMAGE_DATA}',
        "image_url": get_base64_image_from_file(temp_img_file.name),
        "enable_pbr": True,
        "should_remesh": True,
        "should_texture": True,
        "texture_image_url": get_base64_image_from_file(temp_img_file.name)
    }
    
    # payload = {
    #     "mode": "preview",
    #     "prompt": "a monster mask",
    #     "art_style": "realistic",
    #     "should_remesh": True
    # }
    
    
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}"
    }

    #######
    #
    # SLL generation files
    #  [How to generate your own / self-signed SSL certificates for use with an On-Premise deployments](https://kb.teramind.co/en/articles/8791235-how-to-generate-your-own-self-signed-ssl-certificates-for-use-with-an-on-premise-deployments)
    #
    #  [Generate an Azure Application Gateway self-signed certificate with a custom root CA](https://learn.microsoft.com/en-us/azure/application-gateway/self-signed-certificates)
    #
    #  Source Code: [Python Requests - How to use system ca-certificates (debian/ubuntu)?](https://stackoverflow.com/questions/42982143/python-requests-how-to-use-system-ca-certificates-debian-ubuntu)
    ######
    try:
        import certifi 
        meshy_url = "https://api.meshy.ai/openapi/v1/image-to-3d"
       # meshy_url = "https://api.meshy.ai/openapi/v2/text-to-3d"
        # response = requests.post(f"{base_url}{endpoint}", headers=headers, files=files, data=data)
        # response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Call for transform 3D in Meshy cloud whit {MESHY_API_KEY} with information {payload}")
        response = requests.post(
            meshy_url,
            headers=headers,
            json=payload,
            #verify=False
            allow_redirects=True,
            verify=False
            #verify="E:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/webcam_facedetection/webcam_facedetection/ssl_certificate/Amazon-RSA-2048-M02.pem"
        )
     
        response.raise_for_status()
        print("Response from Meshy cloud: ", response.json())
        result = response.json()
        print(f"Call for transform 3D in Meshy cloud {response.json()}")
        
        if result and "result" in result:
            # For now, just return a dummy path. In a real scenario, you would download the model.
            #print(f"Meshy API call successful. Model URL: {result['model_url']}")
            task_id = result['result']
            print("Task ID: ", task_id)
            result = {}
            
            while True:
                response = requests.get(
                    f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
                    headers=headers,
                )
            
                response.raise_for_status()
                result = response.json()
                #print(f"Get the obj file in transform 3D from Meshy cloud {response.json()}")
                print(f"Meshy API call successful. Model URL: {result['progress']}")
                
                progress = result['progress']
                
                if progress == 100:
                    break
                else:
                    time.sleep(60*2)
            
            if result and "model_urls" in result:
               # print(f"Meshy API call successful. Model URL: {result['progress']}")
                print(f"Meshy API call successful. Model URL: {result['model_urls']['obj']}")
                
                obj_url = result['model_urls']['obj']
                response = requests.get(obj_url, headers=headers)
                response.raise_for_status()
                obj_data = response.content
                
                path_meshy_obj_file = os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS_AI_TOOLS, "neutral_meshy.obj")
                
                if os.path.exists( path_meshy_obj_file):
                    os.remove(path_meshy_obj_file)                    
                     
                with open(path_meshy_obj_file, "wb") as f:
                    f.write(obj_data)
                print(f"Write the obj file in transform 3D from Meshy cloud in path: {path_meshy_obj_file}")
                
                obj_url = result['model_urls']['glb']
                response = requests.get(obj_url, headers=headers)
                response.raise_for_status()
                obj_data = response.content
                
                path_meshy_obj_file = os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS_AI_TOOLS, "neutral_meshy.glb")
                
                if os.path.exists( path_meshy_obj_file):
                    os.remove(path_meshy_obj_file)                    
                     
                with open(path_meshy_obj_file, "wb") as f:
                    f.write(obj_data)
                print(f"Write the obj file in transform 3D from Meshy cloud in path: {path_meshy_obj_file}")
                return path_meshy_obj_file
            else:
                raise gr.Error(f"Meshy API did not return a model URL: {result}")
        
           
            print(response.json())
            #return DEFAULT_3D_MODEL_PATH # Replace with actual downloaded model path
        else:
            raise gr.Error(f"Meshy API did not return a model URL: {result}")
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Error calling Meshy API: {e}")
    #finally:
    #    os.unlink(temp_img_file.name) # Clean up the temporary image file



####
#  Cube by CSM.  Image to 3D API
#      https://docs.csm.ai/
#      https://docs.csm.ai/sessions/image-to-3d?language=python
#    Image to 3D API is a feature that allows you to integrate Meshy's Image to 3D 
#    capabilities into your own application. In this section, you'll find all the 
#    information you need to get started with this API.
#
###

def transform_image_with_cubeby_csm(image: np.ndarray, prompt: str) -> str:
    if CUBE_BY_CSM_API_KEY is None:
        raise gr.Error("Cube By CSM API key not configured. Please set CUBE_BY_CSM_API_KEY environment variable.")
    
    if image is None:
        raise gr.Error("No image provided for Meshy transformation.")

    # Save the input image to a temporary file
    temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(temp_img_file.name)
    temp_img_file.close()

    base_url = url = "https://api.csm.ai/v3/sessions/"
    

    payload = {
        # Using data URI example
        # image_url: f'data:image/png;base64,{YOUR_BASE64_ENCODED_IMAGE_DATA}',

        "type": "image_to_3d",
	    "input": { 
         "settings": {
            "texture_model": "pbr"           
         },
         "image":get_base64_image_from_file(temp_img_file.name)}
    }
    headers = {
       "x-api-key": CUBE_BY_CSM_API_KEY,
	   "Content-Type": "application/json"
    }

    #######
    #
    # SLL generation files
    #  [How to generate your own / self-signed SSL certificates for use with an On-Premise deployments](https://kb.teramind.co/en/articles/8791235-how-to-generate-your-own-self-signed-ssl-certificates-for-use-with-an-on-premise-deployments)
    #
    #  [Generate an Azure Application Gateway self-signed certificate with a custom root CA](https://learn.microsoft.com/en-us/azure/application-gateway/self-signed-certificates)
    #
    #  Source Code: [Python Requests - How to use system ca-certificates (debian/ubuntu)?](https://stackoverflow.com/questions/42982143/python-requests-how-to-use-system-ca-certificates-debian-ubuntu)
    ######
    try:
        import certifi 
       
        # response = requests.post(f"{base_url}{endpoint}", headers=headers, files=files, data=data)
        # response.raise_for_status()  # Raise an exception for HTTP errors
        #print(f"Call for transform 3D in Meshy cloud whit {CUBE_BY_CSM_API_KEY} with information {payload}")
        response = requests.post(
            base_url,
            headers=headers,
            json=payload,
            #verify=False
            #verify="E:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/webcam_facedetection/webcam_facedetection/ssl_certificate/Amazon-RSA-2048-M02.pem"
        )
     
        response.raise_for_status()
        #print("Response from Cube by CSM cloud: ", response.json())
        result = response.json()
        #print(f"Call for transform 3D in Cube by CSM cloud {response.json()}")
        
        if result and "_id" in result:
            # For now, just return a dummy path. In a real scenario, you would download the model.
            #print(f"Meshy API call successful. Model URL: {result['model_url']}")
            task_id = result['_id']
            print("Task ID: ", task_id)
            
            url = f"https://api.csm.ai/v3/sessions/{task_id}"


            while True:
                response = requests.get(
                    url,
                    headers=headers,
                )
           
                response.raise_for_status()
                result = response.json()
                print(f" Status in Result ==> {result['status']}")
                status = result['status']
               
                if status == 'failed':
                   raise gr.Error(f"Some error appear in transformation procession")
               
                if status == 'complete':
                   break
                
                time.sleep(60*2)
           
            print(f"Get the obj file in transform 3D from Cube by CSM cloud {response.json()}")
            
            
            
            if result and "output" in result:
                print(f"n\nMeshy API call successful. Model URL: {result['output']['meshes'][0]}")
                
                obj_url = result['output']['meshes'][0]['data']['obj_url']
                
                if obj_url == "":
                    raise gr.Error(f"Cube by CSM API did not return a obj_url URL")
                response = requests.get(obj_url, headers=headers)
                response.raise_for_status()
                obj_data = response.content
                
                path_meshy_obj_file = os.path.join(DEFAULT_3D_MODEL_PATH_ADDRESS_AI_TOOLS, "neutral_cubebycsm.obj")
                
                if os.path.exists( path_meshy_obj_file):
                    shutil.rmtree(path_meshy_obj_file)                    
                     
                with open(path_meshy_obj_file, "wb") as f:
                    f.write(obj_data)
                
                return path_meshy_obj_file
            else:
                raise gr.Error(f"Meshy API did not return a model URL: {result}")
        
           
            print(response.json())
            #return DEFAULT_3D_MODEL_PATH # Replace with actual downloaded model path
        else:
            raise gr.Error(f"Meshy API did not return a model URL: {result}")
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Error calling Meshy API: {e}")
    #finally:
    #    os.unlink(temp_img_file.name) # Clean up the temporary image file
    
    

def transform_image_with_stability_ai(image: np.ndarray, prompt: str) -> str:
    if STABILITY_API_KEY is None:
        raise gr.Error("Stability AI API key not configured. Please set STABILITY_API_KEY environment variable.")
    
    if image is None:
        raise gr.Error("No image provided for Stability AI transformation.")

    # Convert the numpy array image to JPEG format and save to a temporary file
    temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    Image.fromarray(image).save(temp_img_file.name, format="JPEG")
    temp_img_file.close()
    
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    url = f"{api_host}/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}"
    }

    # Prepare the files for the request
    with open(temp_img_file.name, "rb") as img_file:
        files = {
            "init_image": img_file,
        }

        data = {
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1,
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.35,
            "cfg_scale": 7,
            "clip_guidance_preset": "NONE",
            "sampler": "K_EULER",
            "samples": 1,
            "steps": 30,
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            if "artifacts" not in response_data or len(response_data["artifacts"]) == 0:
                raise gr.Error("No artifacts received from Stability AI API.")
            
            # Decode the base64 image and save it temporarily
            output_image_data = response_data["artifacts"][0]["base64"]
            output_image = Image.open(io.BytesIO(base64.b64decode(output_image_data)))
            output_image_path = tempfile.mktemp(suffix=".png")
            output_image.save(output_image_path)

            # NOTE: This API generates a 2D image. Converting this 2D image to a 3D model
            # would require additional processing or a different API, which is not covered here.
            print(f"Stability AI API call successful. Generated 2D image saved to: {output_image_path}")
            return DEFAULT_3D_MODEL_PATH # Return a dummy 3D model path for now

        except requests.exceptions.RequestException as e:
            raise gr.Error(f"Error calling Stability AI API: {e}")
        finally:
            os.unlink(temp_img_file.name) # Clean up the temporary input image file

def transform_image_with_trellis(image: np.ndarray, prompt: str) -> str:
    # Placeholder for TRELLIS API call
    print(f"TRELLIS: Image received, prompt: {prompt}")
    return DEFAULT_3D_MODEL_PATH # Return a dummy 3D model path

def transform_image_with_hunyuan3d_2_1(image: np.ndarray, prompt: str) -> str:
    #Bibliography: from 
    #     https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
    #     https://www.perplexity.ai/search/busca-ejemplos-donde-se-lleve-a6lrCC4ISn2V.m2KIgKx_w
    
    
    import sys
    sys.path.insert(0, './hy3dshape')
    sys.path.insert(0, './hy3dpaint')
    from textureGenPipeline import Hunyuan3DPaintPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    # let's generate a mesh first
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
    mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

    paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
    mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')


# Audio functions
AUDIO_OUTPUT_DIR = "user_audio_records"
AUDIO_OUTPUT_FILE_ADDRESS = ""
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

def record_audio(audio_path, session_id):
    if audio_path is None:
        raise gr.Error("No audio recorded.")
    
    # Save the recorded audio to a temporary file
    # Gradio's Audio component already saves to a temp file, so we just need to move/rename it
    global AUDIO_OUTPUT_FILE_ADDRESS
    
    filename = f"recorded_audio_{session_id}.wav"
    
    output_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    shutil.copy(audio_path, output_path)
    
    AUDIO_OUTPUT_FILE_ADDRESS = output_path
    return output_path, f"Audio recorded and saved to {output_path}"

def save_audio(audio_file, session_id):
    if audio_file is None:
        raise gr.Error("No audio file to save.")
 
    filename = f"user_audio_{session_id}.wav"
    output_path = AUDIO_OUTPUT_FILE_ADDRESS if os.path.exist(AUDIO_OUTPUT_FILE_ADDRESS)  else os.path.join(AUDIO_OUTPUT_DIR, filename)
   
    shutil.copy(audio_file, output_path)
    return output_path, f"Audio saved to {output_path}"

def play_audio_from_prompt(prompt: str, session_id: str) -> Tuple[str, str]:
    if not prompt:
        raise gr.Error("Please provide a prompt to generate audio.")
    
    # Placeholder for actual text-to-speech logic
    # In a real application, you would integrate a TTS API here
    dummy_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"generated_audio_{session_id}_{random.randint(0, 1000)}.wav")
    # Create a dummy silent audio file
    AudioSegment.silent(duration=1000).export(dummy_audio_path, format="wav")
    
    return dummy_audio_path, f"Playing audio for prompt: '{prompt}'"

def play_audio_from_prompt_xtts_v2(prompt: str, session_id: str) -> Tuple[str, str]:
    if not prompt or prompt == "":
        raise gr.Error("Please provide a prompt to generate audio.")
    elif not os.path.exists(AUDIO_OUTPUT_FILE_ADDRESS):
        raise gr.Error("No audio file to play.")
    
    # Placeholder for actual text-to-speech logic
    # In a real application, you would integrate a TTS API here
    dummy_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"generated_audio_{session_id}_{random.randint(0, 1000)}.wav")
    # Create a dummy silent audio file
    AudioSegment.silent(duration=1000).export(dummy_audio_path, format="wav")
    
    return dummy_audio_path, f"Playing audio for prompt: '{prompt}'"

def play_audio_from_prompt_dia(prompt: str, session_id: str) -> Tuple[str, str]:
    if not prompt or prompt == "":
        raise gr.Error("Please provide a prompt to generate audio.")
    elif not os.path.exists(AUDIO_OUTPUT_FILE_ADDRESS):
        raise gr.Error("No audio file to play.")
    
    # Placeholder for actual text-to-speech logic
    # In a real application, you would integrate a TTS API here
    dummy_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"generated_audio_{session_id}_{random.randint(0, 1000)}.wav")
    # Create a dummy silent audio file
    AudioSegment.silent(duration=1000).export(dummy_audio_path, format="wav")
    
    return dummy_audio_path, f"Playing audio for prompt: '{prompt}'"

def play_audio_from_prompt_openvoice_v2(prompt: str, session_id: str) -> Tuple[str, str]:
    if not prompt or prompt == "":
        raise gr.Error("Please provide a prompt to generate audio.")
    elif not os.path.exists(AUDIO_OUTPUT_FILE_ADDRESS):
        raise gr.Error("No audio file to play.")
    # Placeholder for actual text-to-speech logic
    # In a real application, you would integrate a TTS API here
    dummy_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"generated_audio_{session_id}_{random.randint(0, 1000)}.wav")
    # Create a dummy silent audio file
    AudioSegment.silent(duration=1000).export(dummy_audio_path, format="wav")
    
    return dummy_audio_path, f"Playing audio for prompt: '{prompt}'" 
###########################################
#
#  Example functions
#
###########################################

# streaming_active = True

# def stream_handler(img):
#     if streaming_active:
#         # Process img frame
#         return img
#     else:
#         # Ignore frames on stop
#         return None

# def stop_stream():
#     global streaming_active
#     streaming_active = False
#     print("Stream stopped by user")

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
    #     cam_json = gr.JSON(label="Metrics (live)")

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
                fn = process_landmarker_image,
                inputs=[land_img_in, session_id],
                outputs=[land_img_out, land_img_json],
            )
            
            create_avatar_img_btn = gr.Button("Create avatar")
            
            avatar_creation_json = gr.JSON(label="Metrics (live)")

            download_output = gr.File(
                label ="⬇️ Download FBX file",
                visible = True,
                interactive = False
            )
            create_avatar_img_btn.click(
                fn=create_avatar_image,
                inputs=[land_img_in, session_id],
                outputs=[land_img_json, download_output],
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
            avatar_creation_json = gr.JSON(label="Metrics (live)")

            download_output = gr.File(
                label ="⬇️ Download FBX file",
                visible = True,
                interactive = False
            )
            create_avatar_vid_btn.click(
                fn=create_avatar_video,
                inputs=[land_vid_in, land_max_dim, session_id],
                outputs=[avatar_creation_json, download_output],
            )
        
        with gr.Tab("Webcam Landmarker"):
            clear_manual_btn = gr.Button("Clear Manually")
            with gr.Row():
                with gr.Column():
                    land_cam_in = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Webcam",
                        image_mode="RGB",
                    )
                     
                    land_single_face_label = gr.Label(label="Single Face Detected", value="No", show_label=True)
                with gr.Column():
                   land_cam_out = gr.Image(type="numpy", label="Salida", interactive=False)               
                   land_cam_json = gr.JSON(label="Metrics (live)")
          
            
            land_cam_in.stream(
                fn=process_landmarker_stream,
                inputs=[land_cam_in, session_id],
                 outputs=[land_cam_out, land_cam_json,  land_single_face_label],
            )


            clear_manual_btn.click(clear_components, inputs=[],  outputs=[land_cam_out, land_cam_json, land_single_face_label])

            create_avatar_webcam_btn = gr.Button("Create avatar")
            avatar_creation_json = gr.JSON(label="Metrics (live)")
            download_output = gr.File(
                label ="⬇️ Download FBX file",
                visible = True,
                interactive = False
            )
            create_avatar_webcam_btn.click(
                fn = create_avatar_webcam,
                inputs=[land_cam_in, session_id],
                outputs=[avatar_creation_json,download_output]
            )

    with gr.Tab("Visualizador 3D") as threeDVisualizer_tab: 
     
     
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
                interactive=False,
                #value = check_neutral_3d_image_exist(session_id, False),
            )

            threeDVisualizer_tab.select(check_neutral_3d_image_exist, inputs=[session_id], outputs=[model_in])  
            # Add a file upload component for users to upload their own 3D models
            #file_upload = gr.File(label="Upload your own 3D model (OBJ, GLTF/GLB, STL)")

            #file_upload.upload(lambda x: x, inputs=file_upload, outputs=model_in)

    with gr.Tab("Image Transformer") as imageTransformer_tab:
      
     
        with gr.Row():
            with gr.Column():
               neutral_image_path = NEUTRAL_IMAGES_ADDRESS + os.sep + f"neutral_face_{session_id}.jpg"
               print (f"Neutral image path for transform => {neutral_image_path}")        

                    
               img_transform_in = gr.Image(
                                        type ="numpy", 
                                        label ="Input Image", 
                                        sources =[], 
                                        image_mode="RGB",                                    
                                    )
               img_transform_prompt = gr.Textbox(label="Prompt", placeholder="Describe the transformation...")
               
               imageTransformer_tab.select(check_neutral_image_exist, inputs=[session_id], outputs=[img_transform_in])
              
            with gr.Column():        
                img_transform_out = gr.Image(type="numpy", label="Transformed Image", interactive=False)

        with gr.Row():
            generate_image_btn = gr.Button("Generate image with AI Tool (ex: Nano Banana)")
            create_avatar_transformed_btn = gr.Button("Create Avatar")
        with gr.Row():
          avatar_creation_json = gr.JSON(label="Metrics (live)")
          download_output = gr.File(
                label ="⬇️ Download FBX file",
                visible = True,
                interactive = False
            )
        
        generate_image_btn.click(
            fn=transform_image_with_gemini,
            inputs=[img_transform_in, img_transform_prompt],
            outputs=[img_transform_out],
        )

        create_avatar_transformed_btn.click(
                fn = unavailable_botton,
                inputs = None,
                outputs = generate_image_btn,               
                queue=False
        ).then(  
                fn = create_avatar_from_transformed_image,
                inputs=[img_transform_out,session_id],
                outputs=[avatar_creation_json, download_output],
                queue=True
        ).then(
            fn= available_botton,
            inputs = None,
            outputs = generate_image_btn,
            queue =False
        )  

    with gr.Tab("3D Object Transformation") as object_transformation_tab:
        #
        # Bibliography:
        #   https://mtw75.medium.com/generative-ai-image-to-3d-services-apis-a-benchmark-2fb119d96a95
        #
        gr.Markdown("### Transform 2D Image to 3D Object")
        with gr.Tabs():
            with gr.Tab("@Meshy"):
                with gr.Row():
                    meshy_img_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"], image_mode="RGB")
                    meshy_prompt = gr.Textbox(label="Prompt", placeholder="Describe the 3D object...")
                meshy_btn = gr.Button("Generate 3D Model (Meshy)")
                meshy_model_out = gr.Model3D(label="3D Model Output", interactive=False)
                meshy_btn.click(
                    fn=transform_image_with_meshy,
                    inputs=[meshy_img_in, meshy_prompt],
                    outputs=[meshy_model_out],
                )
                
            with gr.Tab("@Cube by CSM"):
                with gr.Row():
                    meshy_img_in = gr.Image(
                        type="numpy", 
                        label="Input Image", 
                        sources=["upload", "clipboard"], 
                        image_mode="RGB",
                        value = DEFAULT_2D_to_3D_MODEL_PATHS
                        )
                    meshy_prompt = gr.Textbox(label="Prompt", placeholder="Describe the 3D object...")
                meshy_btn = gr.Button("Generate 3D Model (Cube by CSM)")
                meshy_model_out = gr.Model3D(label="3D Model Output", interactive=False)
                meshy_btn.click(
                    fn=transform_image_with_cubeby_csm,
                    inputs=[meshy_img_in, meshy_prompt],
                    outputs=[meshy_model_out],
                )
            with gr.Tab("@Stability AI"):
                with gr.Row():
                    stability_img_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"], image_mode="RGB")
                    stability_prompt = gr.Textbox(label="Prompt", placeholder="Describe the 3D object...")
                stability_btn = gr.Button("Generate 3D Model (Stability AI)")
                stability_model_out = gr.Model3D(label="3D Model Output", interactive=False)
                stability_btn.click(
                    fn=transform_image_with_stability_ai,
                    inputs=[stability_img_in, stability_prompt],
                    outputs=[stability_model_out],
                )
            with gr.Tab("@TRELLIS"):
                with gr.Row():
                    trellis_img_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"], image_mode="RGB")
                    trellis_prompt = gr.Textbox(label="Prompt", placeholder="Describe the 3D object...")
                trellis_btn = gr.Button("Generate 3D Model (TRELLIS)")
                trellis_model_out = gr.Model3D(label="3D Model Output", interactive=False)
                trellis_btn.click(
                    fn=transform_image_with_trellis,
                    inputs=[trellis_img_in, trellis_prompt],
                    outputs=[trellis_model_out],
                )
            with gr.Tab("@Hunyuan3D-2.1"):
                with gr.Row():
                    hunyuan3d_img_in = gr.Image(type="numpy", label="Input Image", sources=["upload", "clipboard"], image_mode="RGB")
                    hunyuan3d_prompt = gr.Textbox(label="Prompt", placeholder="Describe the 3D object...")
                hunyuan3d_btn = gr.Button("Generate 3D Model (Hunyuan3D-2.1)")
                hunyuan3d_model_out = gr.Model3D(label="3D Model Output", interactive=False)
                hunyuan3d_btn.click(
                    fn=transform_image_with_hunyuan3d_2_1,
                    inputs=[hunyuan3d_img_in, hunyuan3d_prompt],
                    outputs=[hunyuan3d_model_out],
                )


    with gr.Tab("Audio Features") as audioFeatures_tab:
        gr.Markdown("### Audio Features")
        with gr.Row():
            audio_recorder = gr.Audio(label="Record Audio", sources=["microphone"], type="filepath")
            save_recorded_audio_btn = gr.Button("Save Recorded Audio")
            saved_audio_output = gr.Audio(label="Saved Audio")
        save_audio_message = gr.Textbox(label="Status")

        save_recorded_audio_btn.click(
            fn= record_audio,
            inputs=[audio_recorder, session_id],
            outputs=[saved_audio_output, save_audio_message]
        )

        gr.Markdown("### Play Audio from Prompt")
        audio_prompt_input = gr.Textbox(label="Prompt for Audio Generation", placeholder="Enter text to generate audio...")
           
        with gr.Tabs():
            with gr.Tab("@XTTS-v2"):
                with gr.Row():
                    play_audio_btn1 = gr.Button("Play with Audio with @XTTS-v2")
                    audio_output_1 = gr.Audio(label="Generated Audio 1")
            with gr.Tab("@Dia"):
                with gr.Row():
                    play_audio_btn2 = gr.Button("Play Audio with @Dia")
                    audio_output_2 = gr.Audio(label="Generated Audio 2")
            with gr.Tab("@OpenVoice_v2"):
                with gr.Row():
                    play_audio_btn3 = gr.Button("Play Audio with @OpenVoice_v2")
                    audio_output_3 = gr.Audio(label="Generated Audio 3")

        
        play_audio_btn1.click(
            fn = play_audio_from_prompt_xtts_v2,
            inputs = [audio_prompt_input, session_id],
            outputs = [audio_output_1, save_audio_message]
        )
        play_audio_btn2.click(
            fn = play_audio_from_prompt_dia,
            inputs = [audio_prompt_input, session_id],
            outputs = [audio_output_2, save_audio_message]
        )
        play_audio_btn3.click(
            fn = play_audio_from_prompt_openvoice_v2,
            inputs = [audio_prompt_input, session_id],
            outputs = [audio_output_3, save_audio_message]
        )


   
    # with gr.Tab("Text Example Image Stream") as evaluate_idea:
    #      videoWebCam = gr.Video(sources=["webcam"], format="mp4"),
    #      webcam = gr.Image( sources=["webcam"])       
    #      output = gr.Image()
    #      stop_button = gr.Button("Stop Stream")
    
    #      stop_button.click(fn=stop_stream)

    #      webcam.stream(fn=stream_handler, inputs=webcam, outputs=output)

   
   
    ## Free and delete user directory when the user close the application
    #
    # Bibliografy:
    #       https://www.gradio.app/guides/resource-cleanup
    #
    #      Request object: https://www.gradio.app/docs/gradio/request
    demo.unload(delete_directory)
    
if __name__ == "__main__":
    demo.launch(share=True)


