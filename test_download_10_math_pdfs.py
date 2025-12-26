import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUTPUT_DIR = "math_pdfs"
BASE_OAI_URL = "https://oaipmh.arxiv.org/oai"
PDF_BASE_URL = "https://arxiv.org/pdf"
SET_SPEC = "math"

MAX_PDFS = 10
MAX_WORKERS = 5
TIMEOUT = 30
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_first_math_ids(limit):
    params = {
        "verb": "ListRecords",
        "set": SET_SPEC,
        "metadataPrefix": "arXiv",
    }

    r = requests.get(BASE_OAI_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/"
    }

    ids = []
    for record in root.findall(".//oai:record", ns):
        arxiv_id = record.find(".//arxiv:id", ns)
        if arxiv_id is not None:
            ids.append(arxiv_id.text.strip())
        if len(ids) >= limit:
            break

    return ids


def build_pdf_url(arxiv_id):
    # Old-style IDs contain a slash
    if "/" in arxiv_id:
        return f"{PDF_BASE_URL}/{arxiv_id}.pdf"
    else:
        return f"{PDF_BASE_URL}/{arxiv_id}.pdf"


def safe_filename(arxiv_id):
    # Replace slashes so Windows is happy
    return arxiv_id.replace("/", "_") + ".pdf"


def download_pdf(session, arxiv_id):
    filename = safe_filename(arxiv_id)
    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):
        return

    url = build_pdf_url(arxiv_id)
    r = session.get(url, stream=True, timeout=TIMEOUT)

    if r.status_code != 200:
        print(f"FAILED: {arxiv_id} ({r.status_code})")
        return

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)


def main():
    print("Fetching first 10 mathematics paper IDs...")
    ids = get_first_math_ids(MAX_PDFS)
    print("IDs:", ids)

    session = requests.Session()

    print("Downloading PDFs...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_pdf, session, arxiv_id)
            for arxiv_id in ids
        ]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Test download complete.")


if __name__ == "__main__":
    main()
