from PIL import Image
from typing import List
import torch

def load_images(image_path: List[str]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in image_path]

def embed_images(images, clip_model, preprocess) -> List[List[float]]:
    """images: List[PIL.Image]"""

    processed = torch.stack([preprocess(img) for img in images]).to(clip_model.device)

    with torch.no_grad():
        embeddings = clip_model.encode_image(processed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().tolist()
