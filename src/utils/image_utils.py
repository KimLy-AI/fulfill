from PIL import Image
import torch
import numpy as np
from typing import Optional, Tuple, List

def load_and_preprocess_image(img_path: str, preprocess=None) -> Optional[torch.Tensor]:
    """Load and preprocess a single image"""
    try:
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = preprocess(image.resize((512, 512), Image.Resampling.LANCZOS)) if preprocess else image
        return image_tensor
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def process_batch(image_batch: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, np.ndarray]]:
    """Process a batch of images through CLIP model"""
    if not image_batch:
        return []
    try:
        paths = [item[0] for item in image_batch]
        tensors = torch.stack([item[1] for item in image_batch])
        with torch.no_grad():
            features = tensors
        features_np = features.cpu().numpy()
        return [(paths[i], features_np[i]) for i in range(len(paths))]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []