import torch
import clip
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clip_model(model_name: str = "ViT-B/32", device: str = None) -> tuple:
    """
    Load the CLIP model and preprocessing function.
    
    Args:
        model_name: CLIP model name to load
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
        Tuple of (model, preprocess) objects
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading CLIP model '{model_name}' on device: {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()  # Set to evaluation mode
    logger.info("CLIP model loaded successfully")
    return model, preprocess, device