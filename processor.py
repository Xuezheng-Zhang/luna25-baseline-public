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

from models.timm_model import TIMMPretrainedModel
from experiment_config import Configuration


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, suppress_logs=False, model_name="LUNA25-ensemble"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        # self.model_2d = ResNet18(weights=None).cuda()
        
        # self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()

        cfg_eff = Configuration()
        cfg_eff.MODEL_NAME = "efficientnet_b0"
        cfg_eff.PRETRAINED = False

        cfg_vit = Configuration()
        cfg_vit.MODEL_NAME = "vit_base_patch16_224"
        cfg_vit.PRETRAINED = False

        self.model_eff = TIMMPretrainedModel(cfg_eff)
        self.model_vit = TIMMPretrainedModel(cfg_vit)


        self.model_root = "/opt/app/resources/"

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

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode, name, model):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).cuda()

        ckpt = torch.load(
            os.path.join(
                self.model_root,
                self.model_name,
                f"{name}_best_metric_model.pth",
            )
        )
        model.load_state_dict(ckpt)
        model = model.cuda()
        model.eval()
        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):

        logits_efi = self._process_model("2D", "EfficientNet", self.model_eff)
        logits_vit = self._process_model("2D", "ViT", self.model_vit)

        logits = (logits_efi + logits_vit) / 2.0

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits
