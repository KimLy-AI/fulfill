from src.configs.config import Config
# from src.data_collection.scraper import GoogleDriveScraper
from src.logging_setup.logging_setup import setup_logging
import pathlib
from pathlib import Path
import logging
import json
from src.models.clip_model import CLIPSimilaritySearcher
from src.data_uploading.create_embedding import create_embeddings
from src.data_uploading.upload_data_to_vectorDB import EmbeddingUploader
from src.data_preprocessing.image_preprocess import ImagePreprocessor

import os
import torch
import multiprocessing
from src.data_setup.setup_data import SetupData

logger = setup_logging()


def process_mini_batch(mini_batch_name, config, batch_name, embedding_func):
    try:
        # Embedding
        script_location = pathlib.Path(__file__).parent.resolve()

        project_base_dir = pathlib.Path(__file__).parent.parent.resolve()
        output_directory = project_base_dir.joinpath("data/images/design_flattended")

        config.logger.info(f"Start embedding data for {mini_batch_name}...")
        ImagePreprocessor(
            input_name=mini_batch_name,
            output_directory = output_directory
        ).run()

        return mini_batch_name, "success"
    except Exception as e:
        config.logger.error(f"Error processing {mini_batch_name}: {str(e)}")
        return mini_batch_name, "failed", str(e)

def worker_function(mini_batch_name):
    # Each worker loads its own copy of the model
    config = Config()  # Each worker creates its own config
    embedding_func = create_embeddings

    return process_mini_batch(
        mini_batch_name=mini_batch_name,
        config=config,
        batch_name=config.batch_name,  # Get from config
        embedding_func=embedding_func
    )
def main():
    # # 0. Load config and params
    config = Config(stage="MAIN")

    # 1. Set up data if needed
    config.logger.info(f"Start setting up data...")
    SetupData().setup()

    # 2. Create and start processes with initializer for embedding parallely
    config.logger.info(f"Start processing and embedding data...")

    if config.is_test:
        config.logger.info("Running in test mode, using a single mini batch for testing.")
        mini_batch_names = ["mini_batch1"]
    else:
        config.logger.info("Running in production mode, processing all mini batches.")
        mini_batch_names = get_immediate_subfolder_names(config.base_dir / "data/images")

    with multiprocessing.Pool(processes=config.num_processes) as pool:
        results = pool.map(worker_function, mini_batch_names)

    config.logger.info("\nProcessing summary:")
    for result in results:
        if result[1] == "success":
            config.logger.info(f"{result[0]}: Success")
        else:
            config.logger.error(f"{result[0]}: Failed - {result[2]}")

    # 3. Load embedding data
    config.logger.info("Starting embedding uploader...")
    EmbeddingUploader(mini_batch_names).run()

if __name__ == "__main__":

    # Run pipline
    main()