from src.config.config import Config
# from src.data_collection.scraper import GoogleDriveScraper
from src.logging_setup.logging_setup import setup_logging
from pathlib import Path
import logging
import json
from src.models.clip_model import CLIPSimilaritySearcher
import os
import torch 
logger = setup_logging()

def main():
    context = "Main"
    try:
        # config = Config(
        #     folder_url="https://drive.google.com/drive/folders/1_UeVZ_VhfB3yjDwTMAPOf9xygnN5uvT_",
        #     output_csv="metadata/drive_folder_images.csv",
        #     chrome_profile_path=str(Path.home() / "AppData/Local/Google/Chrome/User Data/Profile 1"),
        #     download_dir="downloaded_images_api",
        #     client_secret_file="src/credentials/client_secret.json"
        # )
        # config.validate()

    #     logger.info("Starting Google Drive image scraper...", extra={'context': context})
    #     scraper = GoogleDriveScraper(config)
    #     scraper.all_images = scraper.scrape_folder_recursively(config.folder_url, folder_name="root")
    #     scraper.save_to_csv()
    #     # scraper.download_images()
    #     logger.info("Scraping, CSV saving, and downloading completed.", extra={'context': context})

    
        # Step 1: Initialize the searcher
        searcher = CLIPSimilaritySearcher(model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

        # Step 2: Load precomputed embeddings
        embedding_file = "embedding_vector_cropped.csv"
        searcher.load_embeddings(embedding_file)

        # Step 3: Process a single image
        image_path = "src\data\images\design_cropped\ JJ sau đen_cropped.png"
        single_result = searcher.process_single_image(image_path, top_k=5)
        print(f"Single Image Result{single_result}")
        # # Step 4: Process a directory of images
        # input_dir = "random_labels_cropped"
        # directory_results = searcher.process_directory(input_dir, top_k=5)

        # # Step 5: Save results to CSV
        # results_file = searcher.save_results_to_csv(directory_results)
        # print(f"Results saved to: {results_file}")

        # # Step 6: Get and save summary statistics
        # stats = searcher.get_summary_stats(directory_results)
        # summary_file = searcher.save_summary_to_csv(stats, directory_results)
        # print(f"Summary saved to: {summary_file}")

        # # Optional: Print summary statistics
        # print("\nSummary Statistics:")
        # for key, value in stats.items():
        #     if key.endswith('_pct'):
        #         print(f"{key.replace('_pct', '')}: {value:.1f}%")
        #     else:
        #         print(f"{key}: {value}")
    except FileNotFoundError as e:
        logger.error(f"Configuration or credentials file not found: {e}", extra={'context': context})
    except Exception as e:
        logger.error(f"Error in main execution: {e}", extra={'context': context})
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'context': context})
        


if __name__ == "__main__":
    main()