import asyncio
import logging
import re
from typing import List, Optional
import httpx
from selectolax.lexbor import LexborHTMLParser
from src.models import WarningLetterMetadata, WarningLetterDocument
import json
import random

# Setup logging for production-ready visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FDAIngestor:
    def __init__(self):
        # The base AJAX URL from your network tab
        self.ajax_url = "https://www.fda.gov/datatables/views/ajax"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Apple)",
            "X-Requested-With": "XMLHttpRequest"
        }
        self.semaphore = asyncio.Semaphore(3) # Rate limiting prevention

    async def get_all_metadata(self, client: httpx.AsyncClient, total_to_fetch: int = 100) -> List[WarningLetterMetadata]:
        """Fetches metadata for FDA warning letters from the FDA's AJAX endpoint.

        Args:
            client (httpx.AsyncClient): Async HTTP client for making requests
            total_to_fetch (int, optional): Maximum number of letters to fetch. Defaults to 100.

        Returns:
            List[WarningLetterMetadata]: List of warning letter metadata objects containing
                                        company name, issue date, URL, issuing office, and subject
        """
        all_meta = []
        start = 0
        page_size = 50

        while len(all_meta) < total_to_fetch:
            params = {
                "view_name": "warning_letter_solr_index",
                "view_display_id": "warning_letter_solr_block",
                "start": start,
                "length": page_size,
                # Note: We likely don't even need _drupal_ajax=1 if it's returning pure DataTables JSON
            }

            resp = await client.get(self.ajax_url, params=params)
            resp.raise_for_status()

            # The structure is: {"data": [[col0, col1, col2, ...], [...]]}
            json_data = resp.json()
            rows = json_data.get("data", [])

            if not rows:
                break

            for cols in rows:
                if len(cols) < 5: continue

                # Since these elements are small HTML snippets,
                # we still use Lexbor to pull the text/attributes out quickly.

                # 1. Parse Issue Date (Index 1)
                issue_date_parser = LexborHTMLParser(cols[1])
                issue_date = issue_date_parser.text(strip=True)

                # 2. Parse Company & URL (Index 2)
                company_parser = LexborHTMLParser(cols[2])
                link_node = company_parser.css_first("a")
                if not link_node: continue

                company_name = link_node.text(strip=True)
                url_path = link_node.attributes.get("href", "")

                # 3. Plain Text Fields
                issuing_office = cols[3]
                subject = cols[4]

                all_meta.append(WarningLetterMetadata(
                    issue_date=issue_date,
                    company_name=company_name,
                    url=f"https://www.fda.gov{url_path}",
                    issuing_office=issuing_office,
                    subject=subject
                ))

            start += page_size
            if len(rows) < page_size: break
            await asyncio.sleep(0.2)

        return all_meta


    async def fetch_full_text(self, client: httpx.AsyncClient, meta: WarningLetterMetadata) -> Optional[str]:
        """Fetches text with exponential backoff to bypass rate limits.

        Args:
            client (httpx.AsyncClient): Async HTTP client for making requests
            meta (WarningLetterMetadata): Metadata object containing the URL to fetch

        Returns:
            Optional[str]: Extracted text content from the warning letter page,
                          or None if fetching fails after retries
        """
        max_retries = 3
        backoff_base = 5  # Start with a 5s wait on failure

        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    # Add a baseline 'politeness' delay even on success
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                    res = await client.get(str(meta.url), timeout=20.0)

                    if res.status_code == 403:
                        wait_time = (backoff_base ** (attempt + 1)) + random.uniform(0, 5)
                        logger.warning(f"403 Forbidden for {meta.company_name}. Retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        continue # Try the next attempt

                    res.raise_for_status()
                    parser = LexborHTMLParser(res.text)
                    article = parser.css_first("article") or parser.css_first(".field-name-body")
                    if not article: return None

                    text = article.text(separator="\n", strip=True)
                    return re.sub(r'\n{3,}', '\n\n', text)

                except Exception as e:
                    logger.error(f"Attempt {attempt} failed for {meta.company_name}: {e}")
                    if attempt == max_retries - 1:
                        return None
                    await asyncio.sleep(2)
        return None


async def main():
    """Main orchestration function for FDA warning letter ingestion.

    Coordinates the complete ingestion pipeline:
    1. Fetches metadata for all warning letters
    2. Retrieves full text content for each letter
    3. Persists data to JSONL file

    Args:
        None

    Returns:
        None: Writes ingested data to warning_letters_raw.jsonl file
    """
    ingestor = FDAIngestor()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Metadata Collection (Breadth)
        all_meta = await ingestor.get_all_metadata(client, total_to_fetch=3387) # 3387 is the reported total number of warning letters

        # 3. Content Extraction (Depth)
        tasks = [ingestor.fetch_full_text(client, m) for m in all_meta]
        texts = await asyncio.gather(*tasks)

        # 4. Persistence
        # Save as JSONL for downstream schematization by the LLM
        with open("warning_letters_raw.jsonl", "w") as f:
            for meta, text in zip(all_meta, texts):
                if text:
                    record = {"metadata": meta.model_dump(mode="json"), "content": text}
                    f.write(json.dumps(record) + "\n")

        logger.info(f"Ingestion complete. Saved {len(all_meta)} letters to warning_letters_raw.jsonl")

if __name__ == "__main__":
    asyncio.run(main())