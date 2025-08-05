## Enviroment Setup
import os, sys
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

from tracker_base import Tracker

def plot(ret_dict, i=0):
    # plot some results
    plt.figure(figsize=(15,6))
    plt.subplot(1,5,1); plt.imshow(ret_dict['img'][i]); plt.title('img'); plt.axis('off')
    plt.subplot(1,5,2); plt.imshow(ret_dict['parsing'][i]); plt.title('parsing'); plt.axis('off')
    plt.subplot(1,5,3); plt.imshow(ret_dict['img_rendered'][i]); plt.title('img_rendered'); plt.axis('off')
    plt.subplot(1,5,4); plt.imshow(ret_dict['shape_rendered'][i]); plt.title('shape_rendered'); plt.axis('off')
    if 'mesh_rendered' in ret_dict:
        plt.subplot(1,5,5); plt.imshow(ret_dict['mesh_rendered'][i]); plt.title('mesh_rendered'); plt.axis('off')
    plt.show()


###########################
## Setup Flame Tracker    #     
###########################

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'use_matting': True,           # use image/video matting to remove background
    'optimize_fov': True,          # whether to optimize the camera FOV, NOTE: this feature is still experimental
    'device': device,
}

tracker = Tracker(tracker_cfg)

# optional settings
tracker.update_init_fov(fov=20)    # this is the initial camera FOV, default is 20
tracker.set_landmark_detector('mediapipe')


print("####### Processing image to evaluate ##########")

img_path = './assets/FFHQ/00002.png'

# if realign == True, 
# img will be replaced by the realigned image
start_time = time.time() # record the start time
ret_dict = tracker.load_image_and_run(img_path, realign=True, photometric_fitting=False) 
end_time = time.time() # record the end time
elapsed_time = end_time - start_time # calculate the elapsed time
print(f"time taken to run load_image_and_run: {elapsed_time:.2f} seconds\n\n") # print the elapsed time

plot(ret_dict)

# check the shapes of returned results
for key in ret_dict:
    if key == 'fov': print('FOV = ', ret_dict[key])
    else: print(f'{key} {ret_dict[key].shape}')