import requests
from bs4 import BeautifulSoup
import json
import re
import os
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Set
import time

# ✅ Bổ sung Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 🔹 Headers giả lập Chrome để bypass 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}

# ================== BỔ SUNG ==================
def fetch_html(url: str, retries: int = 3, delay: int = 3) -> str:
    """Fetch raw HTML with retry + headers"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logging.warning(f"⚠️ Fetch {url} failed ({attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise

def crawl_url(url: str):
    """Tải dữ liệu từ 1 URL bằng requests (fallback)"""
    try:
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return [Document(page_content=text, metadata={"url": url})]
    except Exception as e:
        logging.error(f"❌ Lỗi khi crawl {url}: {e}")
        return []

def crawl_url_selenium(url: str):
    """Tải dữ liệu bằng Selenium (giả lập browser thật)"""
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        options.add_argument("--disable-blink-features=AutomationControlled")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return [Document(page_content=text, metadata={"url": url})]
    except Exception as e:
        logging.error(f"❌ Selenium cũng thất bại khi crawl {url}: {e}")
        return []

def safe_load_url(url: str):
    """Thử WebBaseLoader → requests+BS4 → Selenium"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        logging.info(f"✅ WebBaseLoader loaded {len(docs)} docs from {url}")
        return docs
    except Exception as e:
        logging.warning(f"⚠️ WebBaseLoader failed for {url}: {e} → thử requests+BS4")
        docs = crawl_url(url)
        if docs:
            return docs
        logging.warning(f"⚠️ requests+BS4 cũng fail → thử Selenium")
        return crawl_url_selenium(url)
    
# ================== END BỔ SUNG ==================

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class UXOCrawler:
    def __init__(self):
        self.sources = {
            "mag_vietnam": "https://maginternational.org/vietnam",
            "unicef": "https://www.unicef.org/vietnam/",
            "undp": "https://www.undp.org/vietnam",
            "vnmac": "https://vnmac.gov.vn/"
        }

    def get_all_links(self, base_url: str, limit: int = 20) -> List[str]:
        """Lấy toàn bộ link con trong cùng domain (giới hạn limit để tránh quá tải)."""
        try:
            resp = requests.get(base_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logging.error(f"❌ Error fetching {base_url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        base_domain = urlparse(base_url).netloc
        links: Set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.add(full_url)

        links = list(links)
        if len(links) > limit:
            links = links[:limit]
        logging.info(f"🔗 Found {len(links)} links in {base_url}")
        return links

    def crawl_domain(self, source_name: str, base_url: str, limit: int = 20) -> List[Document]:
        """Crawl toàn bộ link con trong 1 domain"""
        docs = []
        urls = self.get_all_links(base_url, limit=limit)
        if base_url not in urls:
            urls.insert(0, base_url)

        for url in urls:
            try:
                loaded = safe_load_url(url)  # 🔹 Dùng safe_load_url thay vì crawl_url trực tiếp
                for d in loaded:
                    d.metadata["source"] = source_name
                    d.metadata["url"] = url
                    d.metadata["length"] = len(d.page_content.split())
                docs.extend(loaded)
                logging.info(f"✅ Crawled {len(loaded)} docs from {url}")
            except Exception as e:
                logging.warning(f"⚠️ Skipped {url}: {e}")
        return docs
