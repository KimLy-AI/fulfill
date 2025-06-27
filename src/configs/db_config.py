from src.utils.utils import Constant as CONST
from src.configs.config import Config

VERSION = Config().version
TABLE_NAME = f"design_embeddings_{VERSION}"

COLUMN_DEFINITIONS = [  
    ("file_id", "TEXT"),
    ("filename", "TEXT"),
    ("file_extension", "TEXT"),
    ("folder_name", "TEXT"),
    ("folder_url", "TEXT"),
    ("direct_url", "TEXT"),
    ("download_url", "TEXT"),
    ("file_size", "TEXT"),
    ("embedded_design_images", f"VECTOR({CONST.EMBEDED_SIZE})"),
    ("modified_date", "TEXT"),
    ("collection_timestamp", "TEXT")
]