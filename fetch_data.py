"""
==============================================================================
FETCH THE ONLINE RETAIL II DATASET
==============================================================================
One-time setup for the RFM and CLTV lessons.

    uv run python fetch_data.py

Downloads the UCI "Online Retail II" archive (~45 MB, no login required),
converts it to a CSV the lessons can read quickly, and leaves it in
``data/retail/``. Run it once; ``rfm.py`` and ``cltv.py`` read the CSV directly
from then on.

Why the conversion
------------------
UCI ships the data as a two-sheet ``.xlsx``. Reading a million rows out of an
Excel workbook takes the better part of a minute and holds a lot of memory,
because the file has to be un-zipped and parsed as XML before pandas sees a
single row. Doing that once here — instead of on every lesson run, and instead
of on every page load in the teaching app — is the whole point of this script.

The two sheets are converted **one at a time**, and each is freed before the
next is read. That halves the peak memory of the only step in this repository
big enough to matter.

If you would rather not run this at all, don't: every lesson falls back to
synthetic data and says so.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
==============================================================================
"""

from __future__ import annotations

import gc
import resource
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "retail"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ARCHIVE_PATH = DATA_DIR / "online_retail_II.zip"
EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"
CSV_PATH = DATA_DIR / "online_retail_II.csv"

# The columns we keep. `Description` is the product name — useful for a market
# basket lesson, dead weight for segmentation, and by far the widest column in
# the file, so it is dropped before it ever reaches a DataFrame.
KEEP_COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def peak_memory_mb() -> float:
    """Peak resident memory of this process so far, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def download(url: str = ARCHIVE_URL, target: Path = ARCHIVE_PATH) -> Path:
    """Download the archive, unless it is already here."""
    if target.exists():
        print(f"  archive already present ({target.stat().st_size / 1e6:.0f} MB), skipping download")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1 << 20):  # 1 MB at a time, never the whole file in RAM
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / 1e6:6.1f} / {total / 1e6:.0f} MB", end="", flush=True)
    print(f"\r  downloaded {downloaded / 1e6:.0f} MB" + " " * 20)
    return target


def unzip(archive: Path = ARCHIVE_PATH, target: Path = EXCEL_PATH) -> Path:
    """Extract the workbook from the archive."""
    if target.exists():
        print("  workbook already extracted, skipping")
        return target

    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        print(f"  extracting {name}")
        with zf.open(name) as source, open(target, "wb") as handle:
            while chunk := source.read(1 << 20):
                handle.write(chunk)
    return target


def convert_to_csv(workbook: Path = EXCEL_PATH, target: Path = CSV_PATH) -> Path:
    """Convert the workbook to a single CSV, one sheet at a time.

    The two sheets are the 2009-2010 and 2010-2011 trading years. They share a
    schema and are two halves of one transaction log, so they are concatenated —
    but sequentially, appending the second to the CSV rather than holding both
    in memory at once.
    """
    sheets = pd.ExcelFile(workbook, engine="openpyxl").sheet_names
    print(f"  {len(sheets)} sheet(s): {', '.join(sheets)}")

    total_rows = 0
    for position, sheet in enumerate(sheets):
        started = time.time()
        frame = pd.read_excel(workbook, sheet_name=sheet, engine="openpyxl", usecols=KEEP_COLUMNS)
        # Append after the first sheet; write the header only once.
        frame.to_csv(target, index=False, mode="w" if position == 0 else "a",
                     header=position == 0)
        total_rows += len(frame)
        print(f"  {sheet}: {len(frame):,} rows in {time.time() - started:.0f}s "
              f"(peak {peak_memory_mb():.0f} MB)")

        # Free the sheet before reading the next one — this is what keeps peak
        # memory at one sheet rather than the whole workbook.
        del frame
        gc.collect()

    print(f"  wrote {total_rows:,} rows -> {target.name} ({target.stat().st_size / 1e6:.0f} MB)")
    return target


def main() -> int:
    started = time.time()
    print("=" * 78)
    print("FETCHING ONLINE RETAIL II")
    print("=" * 78)

    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists — nothing to do.")
        print("Delete it and re-run if you want a fresh copy.")
        return 0

    print("\n1. Download")
    download()
    print("\n2. Extract")
    unzip()
    print("\n3. Convert to CSV (the slow step — a minute or so)")
    convert_to_csv()

    # The intermediates are ~90 MB of duplicate data. The CSV is all the lessons
    # need, so offer the space back rather than silently keeping three copies.
    for leftover in (ARCHIVE_PATH, EXCEL_PATH):
        if leftover.exists():
            leftover.unlink()
            print(f"  removed intermediate {leftover.name}")

    print("\n" + "=" * 78)
    print(f"Done in {time.time() - started:.0f}s, peak memory {peak_memory_mb():.0f} MB")
    print(f"Data ready: {CSV_PATH}")
    print("Now run:  uv run python rfm.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
