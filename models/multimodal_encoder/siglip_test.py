from siglip_encoder import SiGLIPVisionTower
import torch
from transformers import SiglipProcessor
import requests
from PIL import Image

model = SiGLIPVisionTower()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(inputs)
    print(logits.shape)
