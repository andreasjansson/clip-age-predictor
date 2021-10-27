from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
import cog

AGES = list(range(1, 100))


class Predictor(cog.Predictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        texts = [f"this person is {age} years old" for age in AGES]
        prompts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            self.prompt_features = self.model.encode_text(prompts)
        self.prompt_features /= self.prompt_features.norm(dim=-1, keepdim=True)

    @cog.input("image", type=Path, help="Input image")
    def predict(self, image):
        pil_image = Image.open(image)
        with torch.no_grad():
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (
            (100.0 * image_features @ self.prompt_features.T)
            .softmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        age = AGES[np.argmax(similarity[0])]
        return f"CLIP thinks you are {age} years old"
