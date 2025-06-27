from dataclasses import dataclass, field
from typing import Set, List
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
import os

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

    def __init__(self, stage=None):
        self._load_environment()
        self._setup_paths()
        self._setup_db_config()
        self.logger = self._setup_logger(stage=stage)
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

    def _load_environment(self):
        """Load environment variables from .env file"""
        load_dotenv()
        self.bucket_name = os.getenv("BUCKET_NAME")
        self.env = os.getenv("ENV")
        self.batch_name = os.getenv("BATCH_NAME")
        self.scan_type_is_all = os.getenv("SCAN_TYPE_IS_ALL").strip().lower() == "true"  # convert str to bool
        self.model_download_enabled = os.getenv(
            "MODEL_DOWNLOAD_ENABLED").strip().lower() == "true"  # convert str to bool
        self.data_download_enabled = os.getenv("DATA_DOWNLOAD_ENABLED").strip().lower() == "true"  # convert str to bool
        self.version = os.getenv("VERSION")
        self.num_processes = int(os.getenv("NUM_PROCESSES"))
        self.key = os.getenv("GEMINI_API_KEY")
        self.is_test = os.getenv("IS_TEST").strip().lower() == "true"  # convert str to bool

    def _setup_paths(self):
        """Setup directory paths"""
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.src_dir = Path(__file__).resolve().parent.parent
        self.setup_data_config_path = self.src_dir / "configs/setup_data_config.yaml"
        # self.file_path = self.base_dir / f"data/information/{self.batch_name}/{self.batch_name}.csv"
        # self.processed_data_path = self.base_dir / f"data/information/{self.batch_name}/1_preprocessed_data.csv"
        # self.embedded_data_path = self.base_dir / f"data/information/{self.batch_name}/2_embedded_data.csv"
        # self.image_folder = self.base_dir / f"data/images/{self.batch_name}"

    def _setup_db_config(self):
        """Setup database configuration from environment variables"""
        self.db_config = {
            'project_id': os.getenv("PROJECT_ID"),
            'region': os.getenv("REGION"),
            'instance': os.getenv("INSTANCE"),
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'host': os.getenv("HOST")
        }

    @staticmethod
    def _setup_logger(stage) -> logging.Logger:
        """Configure and return a logger instance with icons for log levels."""
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Add icons for different log levels
        logging.addLevelName(logging.INFO, f"ℹ️ INFO - [{stage}]")
        logging.addLevelName(logging.WARNING, f"⚠️ WARNING - [{stage}]")
        logging.addLevelName(logging.ERROR, f"❌ ERROR - [{stage}]")
        logging.addLevelName(logging.CRITICAL, f"🔥 CRITICAL - [{stage}]")
        logging.addLevelName(logging.DEBUG, f"🐞 DEBUG - [{stage}]")

        return logger
@dataclass
class AppConfig:
    """Application configuration settings."""
    BASE_FOLDERS: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    EMBEDDING_FILE: str = 'embedding_vector_cropped.csv'
    CLIP_MODEL: str = "ViT-B/32"
    COLS_PER_ROW: int = 3
    TOP_K: int = 30
    IMAGE_THUMBNAIL_SIZE: Tuple[int, int] = (300, 300)
    PREVIEW_THUMBNAIL_SIZE: Tuple[int, int] = (300, 300)
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        self.EMBEDDING_PATH = self.BASE_FOLDERS / 'data' / 'database' / self.EMBEDDING_FILE
        self.IMAGE_PATH = self.BASE_FOLDERS / 'data' / 'images'
        self.METADATA_PATH = self.BASE_FOLDERS / 'data' / 'clean_metadata'
        self.SEARCH_PATHS = [
            self.IMAGE_PATH / 'Design_crop_flattened',
            self.IMAGE_PATH,
            self.BASE_FOLDERS / 'data' / 'database',
            Path('./data/images/Design_crop_flattened'),
            Path('./data/images'),
            Path('images/Design_crop_flattened'),
            Path('images'),
            Path('.')
        ]

@dataclass
class SimilarityResult:
    """Container for similarity search results."""
    image_path: Path
    similarity_score: float
    rank: int
    
    @property
    def filename(self) -> str:
        """Get the filename of the image."""
        return self.image_path.name