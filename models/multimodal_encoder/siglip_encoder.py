import torch
import torch.nn as nn

from transformers import SiglipModel, SiglipProcessor, CLIPVisionConfig


class SiGLIPVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"  # Set to auto for now, will set up args later
        self.dtype = torch.bfloat16
        model_name = "google/siglip-base-patch16-224"
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        self.vision_tower = SiglipModel.from_pretrained(
            model_name, device_map=self.device_map
        )
        self.vision_tower.requires_grad_(False)  # Freeze encoder

    @torch.no_grad()
    def forward(self, images):
        # Modified from LLaVA/llava/model/multimodal_encoder/clip_encoder.py
        if type(images) is list:
            image_feats = []
            for image in images:
                image_forward_out = self.vision_tower.get_image_features(
                    **image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feats.append(image_forward_out)
        else:
            image_feats = self.vision_tower.get_image_features(
                **images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
        return image_feats
