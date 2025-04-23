import os
import re
import csv
import base64
import requests
import concurrent.futures
from lxml import html
from PIL import Image
from io import BytesIO

###############################################################################
# Configuration
###############################################################################

CSV_FILENAME = "perfumes_updated.csv"  # CSV with a "Perfumes" column
SAVE_DIR = os.path.join("static", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

# Optionally resize images to EXACTLY 375×500
RESIZE_TO = (375, 500)

# Concurrency limit
MAX_WORKERS = 1

# A user-agent to reduce blocking
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.9"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "DNT": "1",
    "Referer": "https://www.google.com/",
}


# Absolute XPath you provided:
XPATH_EXPR = (
    "/html/body/div[3]/div/div[14]/div/div[2]/div[2]/div/div/div/div/"
    "div[1]/div/div/div[1]/div[2]/h3/a/div/div/div/g-img/img"
)

###############################################################################
# Helper Functions
###############################################################################

def sanitize_filename(name: str) -> str:
    """
    Convert 'Jean Paul ???' -> 'Jean_Paul'
    """
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w_\-]", "", name)
    return name

def build_google_images_url(query: str) -> str:
    """
    e.g. https://www.google.com/search?q=Jean+Paul&tbm=isch
    """
    from urllib.parse import quote_plus
    encoded = quote_plus(query)
    return f"https://www.google.com/search?q={encoded}&tbm=isch"

def resize_if_needed(img_bytes: bytes) -> bytes:
    """
    If RESIZE_TO is set, resize the image to EXACT dimensions (375x500),
    then save to JPEG. Convert to 'RGB' if not already, because Pillow 
    can't write 'P' mode directly to JPEG.
    """
    if not RESIZE_TO:
        return img_bytes

    try:
        pil_img = Image.open(BytesIO(img_bytes))
        
        # If the image is in 'P' mode (indexed color), convert it to 'RGB'
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Use LANCZOS for high-quality resizing
        pil_img = pil_img.resize(RESIZE_TO, resample=Image.Resampling.LANCZOS)

        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception as e:
        print(f"[ERROR] Resizing failed: {e}")
        return None


def process_data_url(src: str) -> bytes:
    """
    src might be data:image/png;base64,iVBOR...
    Decode the base64 portion to raw bytes.
    Returns None if something fails.
    """
    try:
        split_base64 = src.split("base64,")
        if len(split_base64) < 2:
            return None
        b64_part = split_base64[1]
        raw = base64.b64decode(b64_part)
        return raw
    except Exception:
        return None

def download_and_save_image(perfume_name: str, src: str, index: int):
    """
    If `src` starts with data:, decode & resize
    Else, treat as normal URL and GET -> resize
    Save under static/images/<Perfume>_<Index>.jpg
    """
    img_bytes = None

    if src.startswith("data:"):
        raw = process_data_url(src)
        if not raw:
            print(f"[WARN] Base64 data broken for '{perfume_name}' (index={index})")
            return
        final_data = raw
        if final_data:
            img_bytes = final_data
    else:
        # Normal URL
        if src.startswith("//"):
            src = "https:" + src
        try:
            r = requests.get(src, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Download failed for '{perfume_name}' -> {src}: {e}")
            return
        final_data = r.content
        if final_data:
            img_bytes = final_data

    if not img_bytes:
        return

    safe_p = sanitize_filename(perfume_name)
    filename = f"{safe_p}_{index}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        print(f"[INFO] Saved '{perfume_name}' => {filepath}")
    except Exception as e:
        print(f"[ERROR] Writing file '{filepath}': {e}")

def fetch_img_elements_by_xpath(perfume_name: str):
    """
    1) Build Google Images URL
    2) GET page
    3) Parse lxml
    4) Return list of <img> elements via XPATH_EXPR
    """
    url = build_google_images_url(perfume_name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Request failed for '{perfume_name}': {e}")
        return []

    doc = html.fromstring(resp.text)
    elements = doc.xpath(XPATH_EXPR)
    if not elements:
        print(f"[WARN] No images found for '{perfume_name}' via given XPath.")
    return elements

def process_perfume(perfume_name: str):
    """
    - Find <img> elements by the given XPath
    - For each, handle data: or normal URL
    - Download & save
    """
    img_elems = fetch_img_elements_by_xpath(perfume_name)
    for i, elem in enumerate(img_elems, start=1):
        src = elem.get("src")
        if not src:
            print(f"[WARN] <img> missing src for '{perfume_name}' (index={i})")
            continue
        download_and_save_image(perfume_name, src, i)

def main():
    # Read the CSV using semicolon delimiter & latin-1 encoding
    if not os.path.exists(CSV_FILENAME):
        print(f"[ERROR] CSV file '{CSV_FILENAME}' not found.")
        return

    perfumes = []
    with open(CSV_FILENAME, mode="r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p = row.get("Perfumes")
            if p:
                perfumes.append(p)

    print(f"[INFO] Found {len(perfumes)} perfumes in '{CSV_FILENAME}'")

    # Concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_perfume, p): p for p in perfumes}
        for fut in concurrent.futures.as_completed(future_map):
            perfume = future_map[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] Unexpected error for '{perfume}': {e}")

    print("[INFO] Finished all.")

if __name__ == "__main__":
    main()


## //*[@id="dimg_7Sv2Z6LFDoiMxc8PzavpwAs_367"]
## //*[@id="dimg_TDL2Z-W2LMehi-gPsryuuQo_381"]

## /html/body/div[3]/div/div[14]/div/div[2]/div[2]/div/div/div/div/div[1]/div/div/div[11]/div[2]/h3/a/div/div/div/g-img/img

## /html/body/div[3]/div/div[14]/div/div[2]/div[2]/div/div/div/div/div[1]/div/div/div[1]/div[2]/h3/a/div/div/div/g-img/img

## /html/body/div[3]/div/div[14]/div/div[2]/div[2]/div/div/div/div/div[1]/div/div/div[1]/div[2]/h3/a/div/div/div/g-img/img



## normal search
# https://www.google.com/search?client=opera&q=versace+eros&sourceid=opera&ie=UTF-8&oe=UTF-8
# https://www.google.com/search?client=opera&hs=jkV&sca_esv=ad3da75f3216c388&sxsrf=AHTn8zpHMTKZAJbpVHJDPZBh2kiSTYGLbQ:1744188189878&q=versace+eros&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWp5u5rQsC2YJafWTbyNSy6G3Vsi155b_IyTtSTnvQaXi8pqHe_O2hfOzwUB6ozx76J1AC2lHBIr4M8ARN1C70pOeyHAdsDpUGp3-Y1xPZx8QoWgi0znScFIjnaGCqvZPHbYhHD7Mx9PX6QPxI13i6sXch_PxwUb0h94PIKUZzp_O6XOEEzRYv7AQtcvB0sNK0hJ6S8lg&sa=X&ved=2ahUKEwiRseqrx8qMAxUNwAIHHSlhLvsQtKgLegQIDxAB&biw=1482&bih=792&dpr=1.25
# https://www.google.com/search?q=jean+paul+ultramale&client=opera