import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import os

class BoyoVAEDecode:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        self.alternative_image_path = os.path.join(os.path.dirname(__file__), "images", "example.png")
        if not os.path.exists(self.alternative_image_path):
            raise FileNotFoundError(f"Alternative image not found at {self.alternative_image_path}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_and_check_nsfw"
    CATEGORY = "Boyonodes"

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def decode_and_check_nsfw(self, samples, vae):
        decoded_images = vae.decode(samples["samples"])
        
        transform = T.ToPILImage()
        alternative_image = Image.open(self.alternative_image_path).convert('RGB')
        
        for i in range(len(decoded_images)):
            pil_image = transform(decoded_images[i].permute(2, 0, 1))
            result = self.classifier(pil_image)
            
            for r in result:
                if r["label"] == "nsfw" and r["score"] > 0.9:  # Fixed NSFW threshold of 0.9
                    width, height = pil_image.size
                    alt_pil = alternative_image.resize((width, height), resample=Image.LANCZOS)
                    decoded_images[i] = self.pil2tensor(alt_pil)
        
        return (decoded_images,)
