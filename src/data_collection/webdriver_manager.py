from selenium import webdriver
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class WebDriverManager:
    """Context manager for Selenium WebDriver."""
    def __init__(self, chrome_profile_path: str):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument(f"--user-data-dir={chrome_profile_path}")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("--disable-webgl")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.driver: Optional[webdriver.Chrome] = None

    def __enter__(self) -> webdriver.Chrome:
        try:
            self.driver = webdriver.Chrome(options=self.options)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            logger.info("WebDriver initialized.", extra={'context': 'WebDriverManager'})
            return self.driver
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}", extra={'context': 'WebDriverManager'})
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed.", extra={'context': 'WebDriverManager'})