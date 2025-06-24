import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
from typing import List, Tuple, Optional
import pathlib
from src.models.load_models import load_clip_model
from src.utils.image_utils import load_and_preprocess_image, process_batch

device = None  # Will be set by load_clip_model
model = None   # Will be set by load_clip_model
preprocess = None  # Will be set by load_clip_model

def initialize_model():
    global model, preprocess, device
    model, preprocess, device = load_clip_model("ViT-B/32")

def create_embeddings(dataset_folder: str = 'design_crop_flattened', output_file: str = 'embedding_vector.csv') -> None:
    initialize_model()
    # Collect all image paths
    project_base_dir = pathlib.Path(__file__).parent.parent.resolve() 

    input_path = project_base_dir.joinpath(f'data/images/{dataset_folder}')
    output_path  =project_base_dir.joinpath(f'data/database/{output_file}')
    images = [os.path.join(root, file) for root, dirs, files in os.walk(input_path)
              for file in files if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]
    print(f"Found {len(images)} images to process")

    # Results storage with thread-safe access
    image_features_dict = {}
    dict_lock = Lock()
    error_count = 0
    error_lock = Lock()

    def worker_thread(image_paths: List[str], batch_size: int = 8):
        batch = []
        results = []
        for img_path in image_paths:
            processed = load_and_preprocess_image(img_path, preprocess)
            if processed is not None:
                batch.append((img_path, processed))
            if len(batch) >= batch_size:
                batch_results = process_batch(batch)
                results.extend(batch_results)
                batch = []
        if batch:
            batch_results = process_batch(batch)
            results.extend(batch_results)
        return results

    print("Processing images with optimized multithreading and batching...")

    # Configure threading parameters
    num_threads = min(4, max(1, len(images) // 50)) if device == "cuda" else min(8, max(1, len(images) // 20))
    batch_size = 16 if device == "cuda" else 4
    print(f"Using {num_threads} threads with batch size {batch_size}")

    # Split images among threads
    chunk_size = len(images) // num_threads
    image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    if len(image_chunks) > num_threads:
        image_chunks[-2].extend(image_chunks[-1])
        image_chunks = image_chunks[:-1]

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_thread, chunk, batch_size) for chunk in image_chunks]
        for i, future in enumerate(as_completed(futures)):
            try:
                thread_results = future.result()
                with dict_lock:
                    for img_path, features in thread_results:
                        image_features_dict[img_path] = features
                print(f"Thread {i+1} completed: {len(thread_results)} images processed")
            except Exception as e:
                print(f"Thread {i+1} failed: {e}")

    print("Image processing complete. Saving embedding vectors...")

    # Create DataFrame and save results
    paths = list(image_features_dict.keys())
    vectors = [feat.tolist() for feat in image_features_dict.values()]
    df_embeddings = pd.DataFrame({"img_path": paths, "embedded": vectors})
    df_embeddings.to_csv(output_path, index=False)

    print(f"Successfully processed {len(image_features_dict)} out of {len(images)} images")
    print(f"Errors encountered: {error_count}")
    if image_features_dict:
        feature_shape = list(image_features_dict.values())[0].shape
        print(f"Feature vector shape: {feature_shape}")
        print(f"Total embedding size: {len(image_features_dict) * feature_shape[0]} features")

if __name__ == "__main__":
    create_embeddings()