import torch
import clip
from typing import List
from PIL import Image

def load_clip_model(device: str="cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def embed_images(images: List[Image.Image], clip_model, preprocess, device) -> List[List[float]]:
    """images: List[PIL.Image]"""

    processed = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        embeddings = clip_model.encode_image(processed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().tolist()

def embed_text_for_clip(text:str, model, device: str):
    tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().tolist()[0]
