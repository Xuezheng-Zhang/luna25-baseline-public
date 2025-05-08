"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.model_2d import ResNet18
import os
import math
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule using an ensemble of 2D and 3D models
    """

    def __init__(self, suppress_logs=False):
        self.size_px = 64
        self.size_mm = 50
        self.suppress_logs = suppress_logs
        self.model_root = "/opt/app/resources"

        if not self.suppress_logs:
            logging.info("Initializing the ensemble system")

        # Load 2D model
        self.model_2d = ResNet18(weights=None).cuda()
        ckpt_2d = torch.load(os.path.join(self.model_root, "best_model_2d.pth"))
        self.model_2d.load_state_dict(ckpt_2d)
        self.model_2d.eval()

        # Load 3D model
        self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()
        ckpt_3d = torch.load(os.path.join(self.model_root, "best_model_3d.pth"))
        self.model_3d.load_state_dict(ckpt_3d)
        self.model_3d.eval()

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):
        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        patch = patch.astype(np.float32)
        patch = dataloader.clip_and_scale(patch)
        return patch

    def predict(self):
        patches_2d = []
        patches_3d = []

        for _coord in self.coords:
            patches_2d.append(self.extract_patch(_coord, [1, self.size_px, self.size_px], mode="2D"))
            patches_3d.append(self.extract_patch(_coord, [self.size_px]*3, mode="3D"))

        # Convert to tensors
        patches_2d = torch.from_numpy(np.array(patches_2d)).cuda()
        patches_3d = torch.from_numpy(np.array(patches_3d)).cuda()

        # Forward pass
        with torch.no_grad():
            logits_2d = self.model_2d(patches_2d).cpu().numpy()
            logits_3d = self.model_3d(patches_3d).cpu().numpy()

        # Apply sigmoid to get probabilities
        probs_2d = torch.sigmoid(torch.from_numpy(logits_2d)).numpy()
        probs_3d = torch.sigmoid(torch.from_numpy(logits_3d)).numpy()

        # Average predictions
        probs_ensemble = (probs_2d + probs_3d) / 2.0
        logits_ensemble = (logits_2d + logits_3d) / 2.0

        return probs_ensemble, logits_ensemble