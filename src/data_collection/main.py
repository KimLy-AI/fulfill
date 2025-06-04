from config import Config
from scraper import GoogleDriveScraper
from logging_setup import setup_logging
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
            client_secret_file="code/datalutech-4e66001df488.json"
        )
        config.validate()

        logger.info("Starting Google Drive image scraper...", extra={'context': context})
        scraper = GoogleDriveScraper(config)
        scraper.all_images = scraper.scrape_folder_recursively(config.folder_url, folder_name="root")
        scraper.save_to_csv()
        scraper.download_images()
        logger.info("Scraping, CSV saving, and downloading completed.", extra={'context': context})

    except FileNotFoundError as e:
        logger.error(f"Configuration or credentials file not found: {e}", extra={'context': context})
    except Exception as e:
        logger.error(f"Error in main execution: {e}", extra={'context': context})
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'context': context})

if __name__ == "__main__":
    secret_path = Path("code/datalutech-4e66001df488.json")
    if not secret_path.exists():
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        with secret_path.open("w", encoding='utf-8') as f:
            json.dump({
                "installed": {
                    "client_id": "YOUR_CLIENT_ID",
                    "project_id": "YOUR_PROJECT_ID",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": "YOUR_CLIENT_SECRET",
                    "redirect_uris": ["http://localhost"]
                }
            }, f)
        logger.warning("Created a DUMMY client_secret.json. Replace with actual credentials.",
                      extra={'context': 'Pre-Main'})
    main()