import os
import time
import random
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUTPUT_DIR = "math_pdfs"
BASE_OAI_URL = "https://oaipmh.arxiv.org/oai"
PDF_BASE_URL = "https://arxiv.org/pdf"
SET_SPEC = "math"

BATCH_SIZE = 100
MAX_WORKERS = 10
REQUEST_TIMEOUT = 45

OAI_BASE_SLEEP = 2
OAI_MAX_RETRIES = 12
BACKOFF_CAP = 180
BATCH_PAUSE = 2

CHECKPOINT_FILE = "oai_checkpoint_token.txt"
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- utility helpers ----------

def safe_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_") + ".pdf"


def id_from_filename(filename: str) -> str:
    return filename.replace(".pdf", "").replace("_", "/")


def build_pdf_url(arxiv_id: str) -> str:
    return f"{PDF_BASE_URL}/{arxiv_id}.pdf"


def load_existing_ids() -> set[str]:
    """
    Scan OUTPUT_DIR once and record all already-downloaded arXiv IDs.
    """
    existing = set()
    for fname in os.listdir(OUTPUT_DIR):
        if fname.lower().endswith(".pdf"):
            existing.add(id_from_filename(fname))
    print(f"Found {len(existing)} already-downloaded PDFs — will skip them.")
    return existing


def load_checkpoint_token() -> str | None:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            tok = f.read().strip()
            return tok or None
    return None


def save_checkpoint_token(token: str | None):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(token or "")


# ---------- OAI harvesting ----------

def oai_get_with_retry(session: requests.Session, params: dict) -> str:
    for attempt in range(1, OAI_MAX_RETRIES + 1):
        try:
            r = session.get(BASE_OAI_URL, params=params, timeout=REQUEST_TIMEOUT)

            if r.status_code == 200:
                return r.text

            # Treat 403 like rate-limiting / backoff
            if r.status_code in (403, 406, 429, 500, 502, 503, 504):
                sleep_s = min(
                    BACKOFF_CAP,
                    (2 ** (attempt - 1)) + random.uniform(0, 2.0)
                )
                print(
                    f"OAI HTTP {r.status_code}. "
                    f"Retry {attempt}/{OAI_MAX_RETRIES} in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
                continue

            # Any other status is unexpected → fail loudly
            r.raise_for_status()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            sleep_s = min(
                BACKOFF_CAP,
                (2 ** (attempt - 1)) + random.uniform(0, 2.0)
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        "OAI request failed after maximum retries. "
        "Rerun the script to resume from checkpoint."
    )



def harvest_math_ids_resumable():
    session = requests.Session()
    token = load_checkpoint_token()

    if token:
        print("Resuming OAI harvest from checkpoint.")
        params = {"verb": "ListRecords", "resumptionToken": token}
    else:
        print("Starting fresh OAI harvest.")
        params = {"verb": "ListRecords", "set": SET_SPEC, "metadataPrefix": "arXiv"}

    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/"
    }

    while True:
        xml_text = oai_get_with_retry(session, params)
        root = ET.fromstring(xml_text)

        for record in root.findall(".//oai:record", ns):
            arxiv_id = record.find(".//arxiv:id", ns)
            if arxiv_id is not None:
                yield arxiv_id.text.strip()

        token_el = root.find(".//oai:resumptionToken", ns)
        next_token = None if token_el is None or token_el.text is None else token_el.text.strip()
        save_checkpoint_token(next_token)

        if not next_token:
            print("OAI harvest complete.")
            break

        params = {"verb": "ListRecords", "resumptionToken": next_token}
        time.sleep(OAI_BASE_SLEEP)


# ---------- downloading ----------

def download_pdf(session: requests.Session, arxiv_id: str, existing_ids: set[str]) -> bool:
    if arxiv_id in existing_ids:
        return False

    filename = safe_filename(arxiv_id)
    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):  # safety net
        existing_ids.add(arxiv_id)
        return False

    url = build_pdf_url(arxiv_id)

    for attempt in range(1, 6):
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    existing_ids.add(arxiv_id)
                    return True

                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, attempt * 2))
                else:
                    return False
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(min(60, attempt * 2))

    return False


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------- main ----------

def main():
    existing_ids = load_existing_ids()

    pdf_session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=MAX_WORKERS)
    pdf_session.mount("https://", adapter)

    batch_num = 1
    for batch in batched(harvest_math_ids_resumable(), BATCH_SIZE):
        batch = [aid for aid in batch if aid not in existing_ids]
        if not batch:
            continue

        print(f"\n=== Batch {batch_num} | {len(batch)} new papers ===")

        success = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(download_pdf, pdf_session, aid, existing_ids) for aid in batch]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                if fut.result():
                    success += 1

        print(f"Batch {batch_num}: {success}/{len(batch)} downloaded")
        batch_num += 1
        time.sleep(BATCH_PAUSE)

    print("All done.")


if __name__ == "__main__":
    main()
