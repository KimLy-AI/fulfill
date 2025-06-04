from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re
from datetime import datetime
from typing import List, Dict, Optional, Set
from pathlib import Path
import requests
import logging
import time
from config import Config
from google_drive_service import GoogleDriveService

logger = logging.getLogger(__name__)

class GoogleDriveScraper:
    """Scraper for Google Drive folders using the Drive API."""
    def __init__(self, config: Config):
        self.config = config
        self.drive_service = GoogleDriveService(
            config.client_secret_file, config.api_name, config.api_version, config.scopes
        )
        self.all_images: List[Dict] = []

    def scrape_folder_recursively(
        self, folder_url: str, folder_name: str = "root",
        visited_urls: Optional[Set[str]] = None, depth: int = 0
    ) -> List[Dict]:
        """Recursively scrape folders for images using Google Drive API."""
        if visited_urls is None:
            visited_urls = set()

        if folder_url in visited_urls:
            logger.info(f"{'  ' * depth}Already visited {folder_url}, skipping...", extra={'context': folder_name})
            return []

        visited_urls.add(folder_url)
        indent = "  " * depth
        context = f"ScrapeRecursive:{folder_name}"
        logger.info(f"{indent}Processing folder: {folder_name}", extra={'context': context})

        folder_id_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', folder_url)
        if not folder_id_match:
            logger.error(f"Could not extract folder ID from URL: {folder_url}", extra={'context': context})
            return []

        current_folder_id = folder_id_match.group(1)
        current_run_images = []

        try:
            # Collect images in current folder
            folder_images = self._collect_images_in_current_folder(folder_url, folder_name)
            current_run_images.extend(folder_images)
            logger.info(f"{indent}Found {len(folder_images)} images", extra={'context': context})

            # Get subfolders
            subfolders_data = self._get_subfolders(current_folder_id, context)
            logger.info(f"{indent}Found {len(subfolders_data)} subfolders", extra={'context': context})

            # Recursively process subfolders
            for i, subfolder_item in enumerate(subfolders_data, 1):
                sub_folder_context = f"{folder_name}/{subfolder_item['name']}"
                logger.info(f"{indent}Processing subfolder {i}/{len(subfolders_data)}: {subfolder_item['name']}",
                            extra={'context': sub_folder_context})
                images_from_subfolder = self.scrape_folder_recursively(
                    subfolder_item['url'], sub_folder_context, visited_urls, depth + 1
                )
                current_run_images.extend(images_from_subfolder)

        except Exception as e:
            logger.error(f"Error scraping folder {folder_name}: {e}", extra={'context': context})
            import traceback
            logger.debug(traceback.format_exc(), extra={'context': context})

        logger.info(f"{indent}Completed folder: {folder_name} (Found {len(current_run_images)} images)",
                    extra={'context': context})
        return current_run_images

    def _get_subfolders(self, folder_id: str, context: str) -> List[Dict]:
        """Retrieve subfolders using Google Drive API."""
        query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        subfolders = []
        page_token = None
        try:
            while True:
                response = self.drive_service.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name)',
                    pageSize=100,
                    pageToken=page_token
                ).execute()
                for folder in response.get('files', []):
                    subfolders.append({
                        "name": folder['name'],
                        "url": f"https://drive.google.com/drive/folders/{folder['id']}",
                        "id": folder['id']
                    })
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
        except Exception as e:
            logger.error(f"Error retrieving subfolders for {folder_id}: {e}", extra={'context': context})
        return subfolders

    def _collect_images_in_current_folder(self, folder_url: str, folder_name: str) -> List[Dict]:
        """Collect image files in the current folder using Google Drive API."""
        context = f"CollectImages:{folder_name}"
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', folder_url)
        if not match:
            logger.error(f"Could not extract folder ID from URL: {folder_url}", extra={'context': context})
            return []
        folder_id = match.group(1)

        files = self.drive_service.list_files(folder_id)
        images = []
        for file_item in files:
            if self._is_image_file(file_item.get('name', '')):
                file_id = file_item.get('id', 'N/A')
                direct_url = file_item.get('webViewLink', f"https://drive.google.com/file/d/{file_id}/view")
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}" if file_id != 'N/A' else "N/A"
                images.append({
                    "filename": file_item.get('name', 'N/A'),
                    "file_extension": self._get_file_extension(file_item.get('name', '')),
                    "folder_name": folder_name,
                    "folder_url": folder_url,
                    "direct_url": direct_url,
                    "preview_url": None,
                    "download_url": download_url,
                    "file_id": file_id,
                    "file_size": file_item.get('size', 'N/A'),
                    "modified_date": file_item.get('modifiedTime', 'N/A'),
                    "collection_timestamp": datetime.utcnow().isoformat() + "Z"
                })
        logger.info(f"Collected {len(images)} images in folder {folder_name}", extra={'context': context})
        return images

    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image based on its extension."""
        if not filename or not isinstance(filename, str):
            return False
        return any(filename.lower().endswith(ext) for ext in self.config.image_extensions)

    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        return Path(filename).suffix.lower() if '.' in filename else ''

    def save_to_csv(self) -> None:
        """Save collected image metadata to CSV."""
        context = "GoogleDriveScraper:save_to_csv"
        if not self.all_images:
            logger.warning("No images to save to CSV.", extra={'context': context})
            return

        output_path = Path(self.config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                "filename", "file_extension", "folder_name", "folder_url",
                "direct_url", "preview_url", "download_url", "file_id",
                "file_size", "modified_date", "collection_timestamp"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_images)
        logger.info(f"Saved {len(self.all_images)} image metadata entries to {self.config.output_csv}",
                    extra={'context': context})

    def download_images(self) -> None:
        """Download images using requests."""
        context = "GoogleDriveScraper:download_images"
        download_dir = Path(self.config.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        downloadable_images = [
            img for img in self.all_images
            if img.get('download_url') and img.get('download_url') != "N/A" and img.get('file_id') != "N/A"
        ]
        if not downloadable_images:
            logger.warning("No images with valid download_url to download.", extra={'context': context})
            return

        logger.info(f"Starting download of {len(downloadable_images)} images...", extra={'context': context})
        success_count = 0
        with ThreadPoolExecutor(max_workers=self.config.max_download_workers) as executor:
            future_to_image = {
                executor.submit(self._download_worker, img): img
                for img in downloadable_images
            }
            for future in as_completed(future_to_image):
                img = future_to_image[future]
                file_context = f"File: {img.get('filename', 'N/A')} ({img.get('file_id', 'N/A')})"
                try:
                    success, message = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"SUCCESS: {message}", extra={'context': file_context})
                    else:
                        logger.error(f"FAILED: {message}", extra={'context': file_context})
                except Exception as e:
                    logger.error(f"Exception during download: {e}", extra={'context': file_context})
        logger.info(f"Download Summary: {success_count}/{len(downloadable_images)} images downloaded.",
                    extra={'context': context})

    def _download_worker(self, image_info: Dict) -> tuple[bool, str]:
        """Download a single image using requests."""
        filename = image_info.get('filename', 'Unknown')
        download_url = image_info.get('download_url')
        folder_name = image_info.get('folder_name', 'root')
        context = f"DownloadWorker:{filename}"

        if not download_url or download_url == "N/A":
            return False, "Missing or invalid download_url"

        try:
            path_parts = folder_name.replace("root/", "", 1).split('/')
            sanitized_path_parts = [re.sub(r'[<>:"/\\|?*]', '_', part) for part in path_parts if part]
            target_sub_dir = Path(self.config.download_dir).joinpath(*sanitized_path_parts)
            target_sub_dir.mkdir(parents=True, exist_ok=True)

            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            save_path = target_sub_dir / safe_filename

            if save_path.exists() and save_path.stat().st_size > 0:
                api_size_str = image_info.get('file_size', '0')
                try:
                    api_size = int(api_size_str)
                    if save_path.stat().st_size == api_size:
                        return True, f"Already exists and size matches: {save_path.relative_to(self.config.download_dir)}"
                except ValueError:
                    return True, f"Already exists (size not compared): {save_path.relative_to(self.config.download_dir)}"

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with requests.Session() as session:
                        response = session.get(download_url, stream=True, verify=False, timeout=60)
                        response.raise_for_status()
                        with save_path.open('wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    if save_path.exists() and save_path.stat().st_size > 0:
                        return True, f"Downloaded to {save_path.relative_to(self.config.download_dir)}"
                    else:
                        save_path.unlink(missing_ok=True)
                        raise Exception("Download resulted in an empty file.")
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}", extra={'context': context})
                        time.sleep(2 ** attempt)
                    else:
                        save_path.unlink(missing_ok=True)
                        return False, f"Failed to download after {max_retries} attempts: {e}"
        except Exception as e:
            save_path.unlink(missing_ok=True)
            return False, f"Error downloading {filename}: {e}"