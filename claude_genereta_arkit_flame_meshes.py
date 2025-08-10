import os
import sys
import torch
import numpy as np
import cv2
import mediapipe as mp
import trimesh
from PIL import Image
import json
from pathlib import Path

# Add DECA to path
sys.path.append('./DECA')
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ARKit blendshape names (52 total)
ARKIT_BLENDSHAPES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft',
    'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
    'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
    'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
    'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut'
]


class FacialAnimationPipeline:
    def __init__(self, deca_model_path='./DECA'):
        """Initialize the pipeline with DECA model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DECA
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d'
        self.deca = DECA(config=deca_cfg, device=self.device)
        
        # MediaPipe face mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def extract_3d_face(self, image_path):
        """Extract 3D face using DECA"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Preprocess for DECA
        h, w = image_array.shape[:2]
        image_tensor = torch.tensor(image_array).float().permute(2,0,1).unsqueeze(0) / 255.
        image_tensor = image_tensor.to(self.device)
        
        # DECA inference
        with torch.no_grad():
            codedict = self.deca.encode(image_tensor)
            opdict, visdict = self.deca.decode(codedict)
        
        # Extract FLAME parameters
        flame_params = {
            'shape': codedict['shape'].cpu().numpy(),
            'exp': codedict['exp'].cpu().numpy(),
            'pose': codedict['pose'].cpu().numpy(),
            'cam': codedict['cam'].cpu().numpy(),
            'tex': codedict['tex'].cpu().numpy() if 'tex' in codedict else None,
            'light': codedict['light'].cpu().numpy() if 'light' in codedict else None
        }
        
        # Extract mesh
        vertices = opdict['vertices'].cpu().numpy().squeeze()
        faces = self.deca.flame.faces_tensor.cpu().numpy()
        
        return flame_params, vertices, faces, opdict