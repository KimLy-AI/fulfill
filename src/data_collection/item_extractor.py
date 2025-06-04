from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Optional
import logging
import time
from config import Config

logger = logging.getLogger(__name__)

class ItemExtractor:
    """Extracts item information from Google Drive using Selenium."""
    def __init__(self, driver: WebDriver, config: Config):
        self.driver = driver
        self.config = config
        self.wait = WebDriverWait(driver, 30)
        self.short_wait = WebDriverWait(driver, 10)

    def scroll_to_load_content(self, context: str) -> None:
        """Scroll to load all content on the page."""
        logger.info("Performing dynamic scroll...", extra={'context': context})
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        max_attempts = self.config.max_scroll_attempts
        consecutive_no_change = 0
        max_no_change = 3

        for attempt in range(max_attempts):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.config.scroll_pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                consecutive_no_change += 1
                if consecutive_no_change >= max_no_change:
                    logger.info(f"Scroll height stable after {consecutive_no_change} attempts.", extra={'context': context})
                    break
            else:
                consecutive_no_change = 0
            last_height = new_height
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

    def find_elements(self, selectors: List[str], context: str) -> List:
        """Find elements using a list of CSS selectors."""
        for selector in selectors:
            try:
                self.short_wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                logger.debug(f"Found {len(elements)} elements with selector: {selector}", extra={'context': context})
                return elements
            except Exception:
                logger.debug(f"No elements found with selector: {selector}", extra={'context': context})
        logger.warning("No elements found with any selector.", extra={'context': context})
        return []

    def get_item_name(self, element, context: str) -> str:
        """Extract item name from an element."""
        name_selectors = ['div.Q5txwe', 'span.N0BeEc.qs41qe', 'div[aria-label][title]']
        for selector in name_selectors:
            try:
                name_elem = element.find_element(By.CSS_SELECTOR, selector)
                if name_elem.is_displayed():
                    name = name_elem.text.strip() or name_elem.get_attribute("title")
                    if name:
                        return name
            except Exception:
                continue
        aria_label = element.get_attribute("aria-label") or ""
        if aria_label:
            parts = aria_label.split(',')[0].strip()
            for suffix in [" Google Drive Folder", " Google Drive File", " Thư mục Google Drive", " Tệp Google Drive"]:
                parts = parts.replace(suffix, "").strip()
            if parts:
                return parts
        return "Unknown"