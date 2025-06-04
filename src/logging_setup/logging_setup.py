import logging
from typing import Any

class ContextFilter(logging.Filter):
    """Add context to log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.context = getattr(record, 'context', 'N/A')
        return True

def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(context)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.addFilter(ContextFilter())
    return logger