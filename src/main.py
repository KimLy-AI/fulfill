from src.config.config import Config
from src.data_collection.scraper import GoogleDriveScraper
from src.logging_setup.logging_setup import setup_logging
from pathlib import Path
import logging
import json

logger = setup_logging()

def main():
    context = "Main"
    try:
        config = Config(
            folder_url="https://drive.google.com/drive/folders/1_UeVZ_VhfB3yjDwTMAPOf9xygnN5uvT_",
            output_csv="metadata/drive_folder_images.csv",
            chrome_profile_path=str(Path.home() / "AppData/Local/Google/Chrome/User Data/Profile 1"),
            download_dir="downloaded_images_api",
            client_secret_file="src/credentials/client_secret.json"
        )
        config.validate()

        logger.info("Starting Google Drive image scraper...", extra={'context': context})
        scraper = GoogleDriveScraper(config)
        scraper.all_images = scraper.scrape_folder_recursively(config.folder_url, folder_name="root")
        scraper.save_to_csv()
        # scraper.download_images()
        logger.info("Scraping, CSV saving, and downloading completed.", extra={'context': context})

    except FileNotFoundError as e:
        logger.error(f"Configuration or credentials file not found: {e}", extra={'context': context})
    except Exception as e:
        logger.error(f"Error in main execution: {e}", extra={'context': context})
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'context': context})

if __name__ == "__main__":
    main()