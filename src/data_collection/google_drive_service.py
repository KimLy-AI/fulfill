from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import pickle
import io
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GoogleDriveService:
    """Service for interacting with Google Drive API."""
    def __init__(self, client_secret_file: str, api_name: str, api_version: str, scopes: List[str]):
        self.client_secret_file = client_secret_file
        self.api_name = api_name
        self.api_version = api_version
        self.scopes = scopes
        self.service = self._create_service()

    def _create_service(self):
        context = "GoogleDriveService:init"
        cred = None
        pickle_file = Path(f"token_{self.api_name}_{self.api_version}.pickle")
        if pickle_file.exists():
            with pickle_file.open("rb") as token:
                cred = pickle.load(token)
        if not cred or not cred.valid:
            if cred and cred.expired and cred.refresh_token:
                try:
                    cred.refresh(Request())
                except Exception as e:
                    logger.warning(f"Failed to refresh token: {e}", extra={'context': context})
                    cred = None
            if not cred:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, self.scopes)
                cred = flow.run_local_server(port=0)
            with pickle_file.open("wb") as token:
                pickle.dump(cred, token)
        try:
            return build(self.api_name, self.api_version, credentials=cred, cache_discovery=False)
        except Exception as e:
            logger.error(f"Failed to create Google Drive service: {e}", extra={'context': context})
            raise

    def list_files(self, folder_id: str, page_size: int = 1000) -> List[Dict]:
        """List files in a Google Drive folder."""
        context = f"GoogleDriveService:folder_id={folder_id}"
        results = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"
        try:
            while True:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, webContentLink)',
                    pageToken=page_token,
                    pageSize=page_size
                ).execute()
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
        except Exception as e:
            logger.error(f"Error listing files: {e}", extra={'context': context})
        return results

    def download_file(self, file_id: str, retries: int = 3) -> Optional[bytes]:
        """Download a file from Google Drive using API."""
        context = f"GoogleDriveService:file_id={file_id}"
        for attempt in range(retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%", extra={'context': context})
                fh.seek(0)
                return fh.read()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries}: {e}", extra={'context': context})
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to download file after {retries} attempts: {e}", extra={'context': context})
                    return None
        return None