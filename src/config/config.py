from dataclasses import dataclass, field
from typing import Set, List
import json
from pathlib import Path

@dataclass
class Config:
    """Configuration for the Google Drive scraper."""
    folder_url: str
    output_csv: str
    chrome_profile_path: str  # Kept for potential Selenium use
    download_dir: str
    client_secret_file: str
    max_download_workers: int = 5
    scroll_pause_time: float = 2.5  # Selenium-related
    action_delay: float = 1.0  # Selenium-related
    max_scroll_attempts: int = 10  # Selenium-related
    image_extensions: Set[str] = field(
        default_factory=lambda: {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
            '.tiff', '.tif', '.svg', '.ico', '.heic', '.avif'
        }
    )
    api_name: str = "drive"
    api_version: str = "v3"
    scopes: List[str] = field(
        default_factory=lambda: ["https://www.googleapis.com/auth/drive"]
    )

    @classmethod
    def from_json(cls, config_file: str) -> 'Config':
        """Load configuration from a JSON file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def validate(self) -> None:
        """Validate configuration settings."""
        if not Path(self.client_secret_file).is_file():
            raise FileNotFoundError(f"Client secret file not found: {self.client_secret_file}")
        if not self.folder_url.startswith("https://drive.google.com/drive/folders/"):
            raise ValueError(f"Invalid folder URL: {self.folder_url}")