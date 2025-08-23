# --- generate_arkit_flame_meshes.py ---
import os, json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
import subprocess
import cv2
import shutil

if sys.version_info >= (3, 11):
    print("python version is 3.11 or higher. attempting to import bpy...")
    try:
        import bpy
        print("bpy module imported successfully!")
        # you can add more bpy-related code here if needed
    except ImportError:
        print("error: bpy module could not be imported. it's usually part of a blender installation's python environment.")
else:
    print(f"\n\nyour python version is {sys.version_info.major}.{sys.version_info.minor}, which is below 3.11. skipping bpy import validation.\n\n")



# Add project paths to the system path
# Adjust these paths if your directory structure is different
DECA_PATH = '../DECA'
MP_TO_FLAME_PATH = '../mediapipe-blendshapes-to-flame'

FLAME_HEAD_TRAKER = 'flame-head-tracker'
sys.path.append(DECA_PATH)
sys.path.append(MP_TO_FLAME_PATH)
#sys.path.append(FLAME_HEAD_TRAKER)

import trimesh

# DECA / FLAME
#from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
#from decalib.models.FLAME import FLAME
from decalib.utils import util


WORKING_DIR = '/teamspace/studios/this_studio/flame-head-tracker'
os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
print("Current Working Directory: ", os.getcwd())

## Computing Device
device = 'cuda:0'
import torch
torch.cuda.set_device(device) # this will solve the problem that OpenGL not on the same device with torch tensors

import matplotlib.pyplot as plt
import numpy as np
import time # import the time module

#if sys.version_info < (3, 11):
from tracker_base import Tracker

from submodules.decalib.deca import DECA
from submodules.flame_lib.FLAME import FLAME
from utils.image_utils import read_img

# DECA
from utils.deca_inference_utils import create_deca_model, get_flame_code_from_deca

# MICA
from utils.mica_inference_utils import create_mica_model, get_shape_code_from_mica, mica_preprocess

if sys.version_info < (3, 11):
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        PerspectiveCameras, PointLights, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, BlendParams
    )


# Mediapipe→FLAME mapper
import sys
sys.path.append('mediapipe-blendshapes-to-flame')
from mp_2_flame import MP_2_FLAME

# ARKit order required by NVIDIA ACE (52)
ARKIT_ORDER = [
 "eyeBlinkLeft","eyeLookDownLeft","eyeLookInLeft","eyeLookOutLeft","eyeLookUpLeft","eyeSquintLeft","eyeWideLeft",
 "eyeBlinkRight","eyeLookDownRight","eyeLookInRight","eyeLookOutRight","eyeLookUpRight","eyeSquintRight","eyeWideRight",
 "jawForward","jawLeft","jawRight","jawOpen","mouthClose","mouthFunnel","mouthPucker","mouthLeft","mouthRight",
 "mouthSmileLeft","mouthSmileRight","mouthFrownLeft","mouthFrownRight","mouthDimpleLeft","mouthDimpleRight",
 "mouthStretchLeft","mouthStretchRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
 "mouthPressLeft","mouthPressRight","mouthLowerDownLeft","mouthLowerDownRight","mouthUpperUpLeft","mouthUpperUpRight",
 "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
 "cheekPuff","cheekSquintLeft","cheekSquintRight","noseSneerLeft","noseSneerRight","tongueOut"
]
 
BLANDESHAPE_DIRECOTRY_NAME = "blandeshape_obj_files"
def plot(ret_dict, i=0, save_path = 'output', filename = 'image'):
    # plot some results
    plt.figure(figsize=(15,6))
    plt.subplot(1,5,1); plt.imshow(ret_dict['img'][i]); plt.title('img'); plt.axis('off')
    plt.subplot(1,5,2); plt.imshow(ret_dict['parsing'][i]); plt.title('parsing'); plt.axis('off')
    plt.subplot(1,5,3); plt.imshow(ret_dict['img_rendered'][i]); plt.title('img_rendered'); plt.axis('off')
    plt.subplot(1,5,4); plt.imshow(ret_dict['shape_rendered'][i]); plt.title('shape_rendered'); plt.axis('off')
    if 'mesh_rendered' in ret_dict:
        plt.subplot(1,5,5); plt.imshow(ret_dict['mesh_rendered'][i]); plt.title('mesh_rendered'); plt.axis('off')
    plt.show()

def save_image_file(data_3d, fig_title, filename_filepath ):
        fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        # Plot your 3D data (e.g., scatter plot, surface plot, etc.)
        # Adjust this based on how your 3D data is structured and what kind of plot you need
        #ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2])
        plt.imshow(data_3d)
        plt.title(fig_title)
        plt.axis('off')
        # Set labels and title if desired       
        #ax.set_title(fig_title)

        # Save the figure
        plt.savefig(filename_filepath) # You can change the filename and format
        plt.close(fig) # Close the figure to free up memory

def plot_save(ret_dict, i=0, save_path = 'output', filename = 'image'):

    if 'img' in ret_dict:
        filename = 'img'
        save_image_file(
            ret_dict[filename][i], 
            filename, 
            save_path + os.sep + f"figure_{filename}.png"
        )

    if 'parsing' in ret_dict:
        filename = 'parsing'
        save_image_file(
            ret_dict[filename][i], 
            filename, 
            save_path + os.sep +   f"figure_{filename}.png"
        )
    
    if 'img_rendered' in ret_dict:
        filename = 'img_rendered'
        save_image_file(
            ret_dict[filename][i],
             filename, 
             save_path + os.sep +   f"figure_{filename}.png"
        )    

    if 'shape_rendered' in ret_dict:
        filename = 'shape_rendered'
        save_image_file(
            ret_dict[filename][i], 
            filename, 
            save_path + os.sep +   f"figure_{filename}.png"
        )
    
    if 'mesh_rendered' in ret_dict:
        filename = 'mesh_rendered'
        save_image_file(
            ret_dict[filename][i], 
            filename,
             save_path + os.sep +   f"figure_{filename}.png"
        )

def save_obj_as_image(obj_path, filename = 'obj_to_img', save_path = '', device = 'cuda'):
    mesh   = load_objs_as_meshes([obj_path], device=device)

    cameras = PerspectiveCameras(device=device)
    lights  = PointLights(location=[[0, 0, 1.0]], device=device)
    raster  = RasterizationSettings(image_size=512)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                            blend_params=BlendParams(background_color=(1,1,1)))
    )

    img = renderer(mesh)[0, ..., :3].cpu().numpy()        # H×W×3 RGB, 0-1
    Image.fromarray((img*255).astype("uint8")).save( f"{save_path}{os.sep}{filename}.png")
  

def main(
    input_image="/teamspace/studios/this_studio/DECA/TestSamples/examples/000001.jpg",
    out_dir="out_arkit_flame",
    amplitude=1.0):  # 0..1; 1.0 is full strength
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Init DECA
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg, device=device)

    # Load & crop image as DECA demo does
    #td = datasets.TestData(input_image, iscrop=True, face_detector='fan')
   # data = td[0]
    #images = data['image'].to(device)[None, ...]
    img = read_img(input_image)
    images =  deca.crop_image(img).to(device)
   
    #with torch.no_grad():
    #    codedict = deca.encode(images)  # contains 'shape','exp','pose', etc.
    
    # DECA inference
    with torch.no_grad():
        codedict = deca.encode(images[None])
        opdict, visdict = deca.decode(codedict)

    # Identity (FLAME shape betas) we want to keep fixed for all frames
    shape_betas = codedict['shape']     # [1, n_shape]

     # Extract FLAME parameters
    flame_params = {
            'shape': codedict['shape'].cpu().numpy(),
            'exp': codedict['exp'].cpu().numpy(),
            'pose': codedict['pose'].cpu().numpy(),
            'cam': codedict['cam'].cpu().numpy(),
            'tex': codedict['tex'].cpu().numpy() if 'tex' in codedict else None,
            'light': codedict['light'].cpu().numpy() if 'light' in codedict else None,
        }

    ### From new library 
    flame_params['head_pose'] = flame_params['pose'][:,:3] if 'pose' in flame_params else None
    flame_params['jaw_pose'] = flame_params['pose'][:,3:]if 'pose' in flame_params else None  

    ##########################
    ## Setup Flame Tracker    #     
    ###########################

    tracker_cfg = {
        'mediapipe_face_landmarker_v2_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/face_landmarker.task",
        'flame_model_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/FLAME2020/generic_model.pkl",
        'flame_lmk_embedding_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/landmark_embedding.npy",
        'ear_landmarker_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/ear_landmarker.pth", # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
        'tex_space_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/FLAME_albedo_from_BFM.npz",
        'face_parsing_model_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/79999_iter.pth",
        'template_mesh_file_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/head_template.obj",
        'result_img_size': 512,
        'use_matting': True,           # use image/video matting to remove background
        'optimize_fov': True,          # whether to optimize the camera FOV, NOTE: this feature is still experimental
        'device': device,
    }

    tracker = Tracker(tracker_cfg)

    # optional settings
    tracker.update_init_fov(fov=20)    # this is the initial camera FOV, default is 20
    tracker.set_landmark_detector('mediapipe')

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

    ret_dict_lmks = tracker.detect_face_landmarks(img)
    
    lmks_dense, lmks_68, lmks_eyes, blend_scores = tracker.unpack_face_landmarks_result(ret_dict_lmks)

    img_aligned, arcface = mica_preprocess(img, lmks_68, image_size=224)
    img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
    img_aligned = np.array(img_aligned).astype(np.float32) / 255
    image = cv2.resize(img_aligned, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).to(device)[None]
    arcface = torch.tensor(arcface).to(device)[None]
   
    codedict_mica = tracker.mica.encode(image, arcface)
    opdict = tracker.mica.decode(codedict_mica)

    shape_data = opdict['pred_shape_code']
    shape_betas = shape_data
    
    #get_shape_code_from_mica(tracker.mica, img_bgr, lmks_68, device)

    #shape_betas = shape_data[:, :tracker.NUM_SHAPE_COEFFICIENTS]

    print("####### Processing image to evaluate ##########")

    # flame_params = {
    #         'shape': codedict['shape'],
    #         'exp': codedict['exp'],
    #         'pose': codedict['pose'],
    #         'cam': codedict['cam'],
    #         'tex': codedict['tex'] if 'tex' in codedict else None,
    #         'light': codedict['light'] if 'light' in codedict else None
    #     }
    

    # DECA pose convention: we'll override jaw/eye pose from mapper per frame
    # Instantiate a FLAME layer (same topology DECA uses)
    
    #flame_layer = FLAME(deca_cfg.model).to(device)

    # Faces/topology from FLAME (constant)
    # We can get faces from flame_layer.faces; if not, read once from DECA saved obj
   
    #faces = flame_layer.faces.cpu().numpy()
    
    # Extract mesh
    #vertices = opdict['vertices'].cpu().numpy().squeeze()

    # detect landmarks (to output)

    
    faces = deca.flame.faces_tensor.cpu().numpy()

    # Init Mediapipe→FLAME mapper
    mp2flame = MP_2_FLAME(mappings_path='/teamspace/studios/this_studio/mediapipe-blendshapes-to-flame/mappings')

   
    # Save the neutral (all zeros) as frame 0 reference (optional)
    neutral_bs = np.zeros((52,), dtype=np.float32)


    exp0, pose0, eye0 = mp2flame.convert(blendshape_scores=neutral_bs[None, :])

    exp0_t = torch.from_numpy(exp0).to(device).float()

    # pose layout in mapper: [global/jaw?] Here, we’ll use only jaw from mapper and keep neck/head neutral.
    pose0_t = torch.zeros((1, 6), device=device).float()  # rot+neck zeros
   
    #jaw_pose = torch.from_numpy(pose0[:, 3:6]).to(device).float() if pose0.shape[1] >= 6 else torch.zeros((1,3), device=device)
    
    jaw_pose = torch.from_numpy(pose0[:, 3:]).to(device).float() if pose0.shape[1] >= 6 else torch.zeros((1,3), device=device)
    
    head_pose = torch.from_numpy(pose0[:,:3]).to(device).float() 
  
    full_pose0 = torch.cat([pose0_t[:, :3], jaw_pose], dim=1)  # [1,6] (head rot 0, jaw from mapper)
  
    exp_code = codedict['exp'][:,:min(50, tracker.NUM_EXPR_COEFFICIENTS)]

    head_code =  codedict['pose'][:,:3]

    jaw_code = codedict['pose'][:,3:]

    print(shape_betas.shape)
    print(exp_code.shape)
    print(head_code.shape)
    print(jaw_code.shape)
    print(f"{exp0_t}")
    print("----- Analysis different data ----")
    print(codedict['exp'].shape)
    print(f"{codedict['exp']}")
    print(full_pose0.shape)
    print("----- New Data Analysis different data ----")
    print(f"{flame_params['head_pose']}")
    print(f"{flame_params['jaw_pose']}")


    

    #exp_transfor, = torch.from_numpy(exp_code).to(device).detach() 
    #verts_neutral, _, _ = flame_layer(shape_params=shape_betas, expression_params = codedict['exp'], pose_params=full_pose0)
  
    #verts_neutral, _, _ = tracker.flame(shape_params=shape_betas, expression_params = codedict['exp'], pose_params=full_pose0)
  
    #verts_neutral, _, _ = tracker.flame(shape_params=shape_betas, expression_params = exp0_t, head_pose_params = head_pose,  jaw_pose_params = jaw_pose)
  
    verts_neutral, _, _ = tracker.flame(shape_params = shape_betas, expression_params = exp_code, head_pose_params = head_code,  jaw_pose_params = jaw_code)
   
    #   self.flame(shape_params=shape, expression_params=exp, 
    #                                     head_pose_params=head_pose, jaw_pose_params=jaw_pose)


    #verts_neutral, _, _ = flame_layer(shape_params= codedict['shape'], expression_params = exp0_t, pose_params=  codedict['pose'])
  
    trimesh.Trimesh(vertices=verts_neutral[0].detach().cpu().numpy(), faces=faces, process=False)\
           .export(Path(out_dir)/"neutral.obj")

    # # For each ARKit blendshape (in required order), create a frame mesh
    # for idx, name in enumerate(ARKIT_ORDER, start=1):  # frames 1..52
    #     bs = np.zeros((52,), dtype=np.float32)
    #     bs[ARKIT_ORDER.index(name)] = amplitude
    #     exp, pose, eye_pose = mp2flame.convert(blendshape_scores=bs[None, :])
    #     exp_t = torch.from_numpy(exp).to(device).float()
    #     # Build pose: keep head neutral, apply mapper's jaw component
    #     jaw_pose = torch.from_numpy(pose[:, 3:6]).to(device).float() if pose.shape[1] >= 6 else torch.zeros((1,3), device=device)
    #     full_pose = torch.cat([torch.zeros((1,3), device=device), jaw_pose], dim=1)  # [1,6]
    #     with torch.no_grad():
    #         verts, _ = flame_layer(shape_params=shape_betas, expression_params=exp_t, pose_params=full_pose)
    #     mesh_path = Path(out_dir)/f"{idx:02d}_{name}.obj"
    #     trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(), faces=faces, process=False).export(mesh_path)

    # Also save a JSON manifest with the order
    #Path(out_dir, "arkit_order.json").write_text(json.dumps(ARKIT_ORDER, indent=2))
    print(f"Generated 52 pose meshes in: {out_dir}")


def main_another_example(
    img_path = "/teamspace/studios/this_studio/DECA/TestSamples/examples/000001.jpg",
    # get the filename from the path
    out_dir="out_arkit_flame",
    amplitude=1.0):  # 0..1; 1.0 is full strength

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(img_path)
    file_name, _ = os.path.splitext(file_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    ##########################
    ## Setup Flame Tracker    #     
    ###########################

    tracker_cfg = {
        'mediapipe_face_landmarker_v2_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/face_landmarker.task",
        'flame_model_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/FLAME2020/generic_model.pkl",
        'flame_lmk_embedding_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/landmark_embedding.npy",
        'ear_landmarker_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/ear_landmarker.pth", # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
        'tex_space_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/FLAME_albedo_from_BFM.npz",
        'face_parsing_model_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/79999_iter.pth",
        'template_mesh_file_path': f"/teamspace/studios/this_studio/{FLAME_HEAD_TRAKER}/models/head_template.obj",
        'result_img_size': 512,
        'use_matting': True,           # use image/video matting to remove background
        'optimize_fov': True,          # whether to optimize the camera FOV, NOTE: this feature is still experimental
        'device': device,
    }

    tracker = Tracker(tracker_cfg)

    # optional settings
    tracker.update_init_fov(fov=20)    # this is the initial camera FOV, default is 20
    tracker.set_landmark_detector('mediapipe')

    # if realign == True, 
    # img will be replaced by the realigned image
    start_time = time.time() # record the start time

    ret_dict = tracker.load_image_and_run(img_path, realign=True, photometric_fitting = True) 
    
    print("*** !! OK Size of data in Deca dict for get texture Ok !!!")
    print(tracker.deca.ret_dict.keys())
   
    os.makedirs(os.path.join(out_dir, file_name), exist_ok=True)

    tracker.deca.save_obj(
        os.path.join(out_dir, file_name, file_name + '.obj'), 
        tracker.deca.ret_dict
        )
    #print(tracker.deca.ret_dict.keys)

    end_time = time.time() # record the end time

    elapsed_time = end_time - start_time # calculate the elapsed time
    print(f"time taken to run load_image_and_run: {elapsed_time:.2f} seconds\n\n") # print the elapsed time


    
    shape = torch.from_numpy(ret_dict['shape']).to(device).detach()           # [N, D_shape]
    exp = torch.from_numpy(ret_dict['exp']).to(device).detach()               # [N, D_exp]
    head_pose = torch.from_numpy(ret_dict['head_pose']).to(device).detach()   # [N, 3]
    jaw_pose = torch.from_numpy(ret_dict['jaw_pose']).to(device).detach() 

    faces = tracker.deca.flame.faces_tensor.cpu().numpy()

    print("####### Processing image to evaluate ##########")

    # Init Mediapipe→FLAME mapper
    mp2flame = MP_2_FLAME(mappings_path='/teamspace/studios/this_studio/mediapipe-blendshapes-to-flame/mappings')  
    

    print(shape.shape)
    print(exp.shape)
    print(head_pose.shape)
    print(jaw_pose.shape)
   
    print("----- Analysis different data ----")

    print("---- Texture Information -----")
    print(ret_dict['tex'].shape)
   

    verts_neutral, _, _ = tracker.flame(shape_params = shape, expression_params = exp, head_pose_params = head_pose,  jaw_pose_params = jaw_pose)
   
    #   self.flame(shape_params=shape, expression_params=exp, 
    #                                     head_pose_params=head_pose, jaw_pose_params=jaw_pose)


    #verts_neutral, _, _ = flame_layer(shape_params= codedict['shape'], expression_params = exp0_t, pose_params=  codedict['pose'])
  
    trimesh.Trimesh(
        vertices = verts_neutral[0].detach().cpu().numpy(), 
        faces = faces, 
        process = False
    ).export(Path(out_dir)/"neutral.obj", include_texture=True, write_texture=True)

    #save_obj_as_image(out_dir + os.sep + "neutral.obj",  save_path = out_dir, device = device)

    #plot_save(ret_dict, i = 0, save_path = out_dir)

    # For each ARKit blendshape (in required order), create a frame mesh

    #Validate if exit 52 bladshape generation directory
    blandshape52Directory = Path(out_dir) / BLANDESHAPE_DIRECOTRY_NAME
    os.makedirs(blandshape52Directory, exist_ok=True)

    for idx, name in enumerate(ARKIT_ORDER, start=1):  # frames 1..52
        bs = np.zeros((52,), dtype=np.float32)
        bs[ARKIT_ORDER.index(name)] = amplitude
        print(f" The blandshape {name} has value for find out : { bs[ARKIT_ORDER.index(name)]}")

        exp, pose, eye_pose = mp2flame.convert(blendshape_scores=bs[None, :])
       
        exp_t = torch.from_numpy(exp).to(device).float()
        # Build pose: keep head neutral, apply mapper's jaw component
        #jaw_pose = torch.from_numpy(pose[:, 3:6]).to(device).float() if pose.shape[1] >= 6 else torch.zeros((1,3), device=device)
       
        #full_pose = torch.cat([torch.zeros((1,3), device=device), jaw_pose], dim=1)  # [1,6]

        jaw_pose = torch.from_numpy(pose[:, 3:]).to(device).float() if pose.shape[1] >= 6 else torch.zeros((1,3), device=device)
    
        head_pose = torch.from_numpy(pose[:,:3]).to(device).float() 
        
        tracker.deca.ret_dict['exp'] = torch.from_numpy(exp).to(device).float()
        tracker.deca.ret_dict['pose'] = torch.from_numpy(pose).to(device).float()  
        tracker.deca.ret_dict['eye_pose'] = torch.from_numpy(eye_pose).to(device).float()    
             
       # [N, D_shape]

        #About texture extraction 
        #https://github.com/mikedh/trimesh/issues/1064
        
        new_file_name = f"{name}_neutral.obj"
        
        tracker.deca.save_obj(
          os.path.join(out_dir, file_name),
          new_file_name, 
          tracker.deca.ret_dict
        )

        with torch.no_grad():
           # verts, _ = flame_layer(shape_params=shape_betas, expression_params=exp_t, pose_params=full_pose)

            #verts_neutral, _, _ = tracker.flame(shape_params = shape, expression_params = exp, head_pose_params = head_pose,  jaw_pose_params = jaw_pose)
            verts_neutral, _, _ = tracker.flame(shape_params = shape, expression_params = exp_t, head_pose_params = head_pose,  jaw_pose_params = jaw_pose)
        
        trimesh.Trimesh(
            vertices = verts_neutral[0].detach().cpu().numpy(), 
            faces = faces, 
            process = False
         ).export(Path(blandshape52Directory)/f"{name}_neutral.obj", include_texture=True, write_texture=True)
        #mesh_path = Path(out_dir)/f"{idx:02d}_{name}.obj"
        
        #trimesh.Trimesh(vertices=verts_neutral[0].detach().cpu().numpy(), faces=faces, process=False).export(mesh_path)

    
    Path(out_dir, "arkit_order.json").write_text(json.dumps(ARKIT_ORDER, indent=2))
    print(f"Generated 52 pose meshes in: {out_dir} for create a FBX files")
    
    ###Return adress to texture files and texture file name
    return (os.path.join(out_dir, "neutral.obj" ),  os.path.join(out_dir, file_name), f"{file_name}.png" )
    

#
#
def create_textured_material(material_name, texture_path):
    """
     Create a material and a node tree to assign a texture.
    """
    # Create a new material or access an existing one.
    mat = bpy.data.materials.get(material_name)
    if not mat:
        mat = bpy.data.materials.new(name = material_name)
    
    mat.use_nodes = True
    
    # Clean up the existing node tree.
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create the Principled BSDF node and the output node.
    principled_bsdf = nodes.new(type = 'ShaderNodeBsdfPrincipled')
    material_output = nodes.new(type = 'ShaderNodeOutputMaterial')
    
    # Create the image texture node.
    image_texture = nodes.new('ShaderNodeTexImage')
    
    # Load the texture image.
    if os.path.exists(texture_path):
        print(f" Extract image features in: {texture_path}")

        image_texture.image = bpy.data.images.load(texture_path, check_existing=True)

    else:
        print(f"Warning: Texture file not found in {texture_path}")
        return None
    
    # Arrange the position of nodes for better visualization.
    principled_bsdf.location = (-200, 0)
    image_texture.location = (-400, 0)
    
    # Link the nodes.
    links = mat.node_tree.links
    links.new(image_texture.outputs[0], principled_bsdf.inputs[0])
    links.new(principled_bsdf.outputs[0], material_output.inputs[0])
    
    #mat.node_tree.nodes["Principled BSDF"].inputs['Specular'].default_value = 0
    #mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.5
    


    return mat



def export_blender_scene_fbl(output_dir = 'out_arkit_flame', filename = 'arkit_52_animation'):
    # Habilitar el complemento glTF 2.0 si no está activo (opcional, pero recomendado)
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    
    IN_DIR = Path(output_dir if output_dir == '' or output_dir == None else "out_arkit_flame")       # where the OBJs are
    OUT_GBL = IN_DIR / f"{filename}.glb"


    # Exportar la escena como GLB
    bpy.ops.export_scene.gltf(filepath= str(OUT_GBL), export_format='GLB')

    print(f"Exported: {OUT_GBL}")

def export_from_objs_to_fbx(output_dir = 'out_arkit_flame', texture_files_dir = '', texture_filename = '', blandeshape_directory = ''):
        # CONFIG
    IN_DIR = Path(output_dir if output_dir == '' or output_dir == None else "out_arkit_flame")       # where the OBJs are
    OUT_FBX = IN_DIR / "arkit_52_animation.fbx"
    SCENE_FPS = 30
    
    TEXTURE_DIR = Path(texture_files_dir if texture_files_dir != '' else IN_DIR)  # where the texture files are
   
    #Scene Cleanup: All objects in the default scene (camera, light, cube) 
    # are removed to start with a clean scene.
    if bpy.ops.object.mode_set.poll():
      bpy.ops.object.mode_set(mode='OBJECT')

    # Select all objects in the scene.
    bpy.ops.object.select_all(action='SELECT')

    # Delete the selected objects.
    bpy.ops.object.delete()
    
    #--------------------------------------------------------

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.context.scene.render.fps = SCENE_FPS

    neutral_path = IN_DIR / "neutral.obj"
    assert neutral_path.exists(), f"Missing {neutral_path}"
    # Import neutral mesh
    #bpy.ops.import_scene.obj(filepath=str(neutral_path))
    
    texture_file = TEXTURE_DIR / texture_filename
    assert texture_file.exists(), f"Missing texture file in {texture_file}"
    print(f"Importing neutral mesh from: {neutral_path} with texture: {texture_file}") 
   
    # Copy file from texture_file to IN_DIR
    dest_texture_path = IN_DIR / "neutral.png"
    shutil.copy(str(texture_file), str(dest_texture_path))   
    
    
    
 
    
    

    #bpy.ops.wm.obj_import(filepath=str(neutral_path))
    
    bpy.ops.import_scene.obj(filepath=str(neutral_path), use_edges=True, use_image_search=True)
    
    # Assigns the material to the active object.
    active_object = bpy.context.active_object
    if active_object and active_object.type == 'MESH':
      new_material = create_textured_material("DECA_Material", str(texture_file))
    
   # Delete existing material slots and add a new one.
    active_object.data.materials.clear()
    active_object.data.materials.append(new_material)
    

    obj = bpy.context.selected_objects[0]
    obj.name = "Head"
    bpy.context.view_layer.objects.active = obj
    # Ensure a Basis key exists
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis", from_mix=False)

    # Load ARKit order
    ARKIT_ORDER = json.loads((IN_DIR/"arkit_order.json").read_text())

    # For each pose OBJ, add a shape key from it
    shape_key_names = []
    for i, name in enumerate(ARKIT_ORDER, start=1):
        #blandshape_filename = f"{i:02d}_{name}.obj"
        blandshape_filename = f"{name}_neutral.obj"

        path = IN_DIR / BLANDESHAPE_DIRECOTRY_NAME / blandshape_filename
        #bpy.ops.import_scene.obj(filepath=str(path))

        bpy.ops.wm.obj_import(filepath=str(path))

        poser = [o for o in bpy.context.selected_objects if o.type == 'MESH'][-1]
        # Add shape key from the poser geometry
        sk = obj.shape_key_add(name=name, from_mix=False)
        # Transfer vertex positions
        obj.data.shape_keys.key_blocks[name].value = 0.0
        # Copy verts (assumes identical topology)
        for v_src, v_dst in zip(poser.data.vertices, obj.data.vertices):
            sk.data[v_dst.index].co = v_src.co
        # Cleanup poser mesh
        bpy.data.objects.remove(poser, do_unlink=True)
        shape_key_names.append(name)

    # Create 52-frame animation, one key per frame
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 52

    for f, name in enumerate(shape_key_names, start=1):
        # zero all keys
        for kb in obj.data.shape_keys.key_blocks:
            kb.value = 0.0
        obj.data.shape_keys.key_blocks[name].value = 1.0
        obj.data.shape_keys.key_blocks[name].keyframe_insert(data_path="value", frame=f)

    ###---------------------------------------------------------
    #
    #  A clean processs before export
    #
    ###----------------------------------------------------------
    # Enter 'EDIT' mode and select the entire mesh for cleaning.
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='SELECT')

    # # Remove duplicate vertices, correcting non-manifold geometry.
    # # The threshold may need adjustment depending on the scale of the model.
    # bpy.ops.mesh.remove_doubles(threshold=0.0001)

    # # Return to 'OBJECT' mode.
    # bpy.ops.object.mode_set(mode='OBJECT')

    # # Scales the object to correct non-uniform transformations.
    # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    
    # FBX export (axes for Unreal: -Z forward, Y up)
    bpy.ops.export_scene.fbx(
        filepath=str(OUT_FBX),
        use_selection = False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z', 
        axis_up='Y',
        
        object_types={'MESH'},  # Export only meshes
        mesh_smooth_type='FACE',   # Smooth shading
        use_mesh_modifiers=True,  # Apply modifiers
        use_triangles=True,  # Export as triangles
        global_scale=1.0,  # Scale factor
        path_mode='COPY', 
        embed_textures = True,  # Automatically handle paths for textures
    )
    print(f"Exported: {OUT_FBX}")

def step_1_reconstruct_3d_from_image(image_path: str ="/teamspace/studios/this_studio/DECA/TestSamples/examples/000001.jpg", output_dir: str ="out_arkit_flame"):
    """
    Step 1: Convert a 2D facial image into a 3D one using DECA.

    This function calls the DECA demo script to reconstruct a 3D face model
    from a single image. It saves the resulting mesh and FLAME parameters.

    Args:
        image_path (str): Path to the input 2D facial image.
        output_dir (str): Directory to save the reconstruction results.
    Returns:
        dict: A dictionary containing the path to the base OBJ and FLAME parameters.
    """
    print("Step 1: Converting 2D image to 3D with DECA...")

    output_dir = f"/teamspace/studios/this_studio/{output_dir}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the DECA reconstruction demo.
    # The --saveObj True flag ensures a detailed OBJ file is created.
    # The script will save a `.pkl` file with the FLAME parameters.
    
    # NOTE: The DECA project structure uses a command-line interface.
    # This simulates running the command.
    deca_reconstruct_command = [
        'python', f'/teamspace/studios/this_studio/DECA/demos/demo_reconstruct.py',
        '--inputpath', image_path,
        '--savefolder', output_dir,
        '--saveDepth', 'True',
        '--saveObj', 'True',
        '--extractTex', 'True',
        '--useTex', 'True',
        '--rasterizer_type','pytorch3d'
        ]
    
    try:
        subprocess.run(deca_reconstruct_command, check=True)
        print("DECA reconstruction successful.")
        
        # Find the generated OBJ and parameter file
        # The file names are based on the input image name.
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        base_obj_path = os.path.join(output_dir, f'{image_name}_detailed.obj')
        
        # The FLAME parameters are saved in a .pkl file
        param_file_path = os.path.join(output_dir, f'{image_name}_detailed.pkl')
        
        if not os.path.exists(base_obj_path) or not os.path.exists(param_file_path):
            raise FileNotFoundError("DECA output files not found.")
        
        # Load the FLAME parameters from the .pkl file
        # The DECA project provides a utility function for this
        flame_params = util.load_pkl(param_file_path)
        
        return {
            "base_obj_path": base_obj_path,
            "flame_params": flame_params
        }
        
    except Exception as e:
        print(f"Error during DECA reconstruction: {e}")
        return None

if __name__ == "__main__":
    #Get 3D information and convert to 52 Blandshep from Arkit
    neutral_obj_address, texture_file_dir, texture_filename = main_another_example()
   
   #Create fbx file with information
    export_from_objs_to_fbx( output_dir = 'out_arkit_flame', 
                             texture_files_dir = texture_file_dir, 
                             texture_filename = texture_filename, 
                             blandeshape_directory = BLANDESHAPE_DIRECOTRY_NAME
                             )

   #Clean and free resources
   
   
    #main()
    #step_1_reconstruct_3d_from_image()
