"""
CosFace Feature Extraction
Best alternative to AdaFace - Consistent & Stable
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, ROOT_DIR)

class CosFaceExtractor:
    def __init__(self, model_path='./models/pretrained_models/cosface.pth', device='cpu'):
        """
        Initialize CosFace model
        
        CosFace benefits:
        - Sister model to ArcFace
        - Very consistent embeddings
        - 512-dimensional output
        - Stable across iterations
        """
        from core.backbones import iresnet50
        
        self.device = device
        self.model = iresnet50()
        
        # Load weights
        statedict = torch.load(model_path, map_location=device)
        
        if 'state_dict' in statedict:
            statedict = statedict['state_dict']
        
        model_statedict = {key.replace("module.", ""): val for key, val in statedict.items()}
        self.model.load_state_dict(model_statedict, strict=False)
        
        self.model.to(device)
        self.model.eval()
        
        # Disable dropout
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        print("CosFace model loaded.")
    
    
    def extract(self, aligned_face):
        """Extract CosFace embedding"""
        
        # Preprocess
        
        with torch.no_grad():
            output = self.model(aligned_face)
            
            if isinstance(output, tuple):
                embedding = output[0]
            else:
                embedding = output
          
            embedding = embedding.cpu().numpy().flatten().astype(np.float32)
            
      
       
        return embedding
