import re
from pathlib import Path

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def get_file_extension(filename: str) -> str:
    """Extract file extension from filename."""
    return Path(filename).suffix.lower() if '.' in filename else ''