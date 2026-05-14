
#Scrapes the Wayback Machine (CDX API) for a list of URLs, grouping snapshots by ISO week,
#downloads only the earliest snapshot per (url, year, week), and tracks:
#  - Which snapshots fail to download (all retries exhausted),
#  - Which ISO weeks got no snapshot at all (missing).
#Prints a summary at the end so you can manually handle or re-scrape.


import requests
import datetime
import time
import os
from collections import defaultdict
from urllib.parse import quote

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

URLS = [
    "https://www.bbc.co.uk/sport",
    "https://www.bbc.co.uk/sport/football",
    "https://www.bbc.co.uk/sport/rugby-union",
    "https://www.bbc.co.uk/sport/cricket",
    "https://www.bbc.co.uk/sport/tennis",
    "https://www.bbc.co.uk/sport/golf",
    "https://www.skysports.com",
    "https://www.skysports.com/football",
    "https://www.skysports.com/rugby-union",
    "https://www.skysports.com/cricket",
    "https://www.skysports.com/tennis",
    "https://www.skysports.com/golf",
    "https://www.mirror.co.uk/sport/",
    "https://www.mirror.co.uk/sport/football/",
    "https://www.mirror.co.uk/sport/rugby-union/",
    "https://www.mirror.co.uk/sport/cricket/",
    "https://www.mirror.co.uk/sport/tennis/",
    "https://www.mirror.co.uk/sport/golf/"
]

START_DATE = "20240401"  # YYYYMMDD
END_DATE = "20240630"  # YYYYMMDD

BASE_DIR = "cdx_archives"
CDX_API = "https://web.archive.org/cdx/search/cdx"

DOWNLOAD_SLEEP = 5  # seconds after each successful download
RETRY_SLEEP = 20  # seconds to wait before re-trying on failure
MAX_RETRIES  = 3  # attempts

# Keep track of failed snapshots & missing weeks
failures = defaultdict(list)     # failures[url] -> list of (year, week, timestamp, archived_url)
missing_weeks = defaultdict(list)  # missing_weeks[url] -> list of (year, week)


# -------------------------------------------------------
# CDX Query & Grouping
# -------------------------------------------------------

"""
  Query the CDX API for all snapshots of `url` between `start` and `end`.
  Returns a list of dicts with fields like:
       {
         "urlkey": ...,
         "timestamp": "20250127073213",
         "original": "https://www.bbc.co.uk/sport/football",
         "mimetype": ...,
         "statuscode": "200",
         "digest": ...,
         "length": ...
       }
"""

def fetch_cdx_snapshots(url, start, end):
    #Fetch snapshot metadata for *url* between *start* and *end* (YYYYMMDD).

    # Build query parameters understood by the CDX API
    params = {
        "url": url,                # Page of interest
        "from": start,             # Inclusive start date
        "to": end,                 # Inclusive end date
        "output": "json",         # Ask for JSON lines
        "collapse": "digest",     # Collapse identical content hashes
        "filter": "statuscode:200"  # Only keep successful captures
    }

    try:
        # Issue HTTP GET with a safety timeout
        resp = requests.get(CDX_API, params=params, timeout=30)
        resp.raise_for_status()  # Raise on HTTP error codes
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] CDX request failed for {url}: {e}")
        return []  # Give caller an empty list on failure

    try:
        data = resp.json()  # Parse the JSON payload
    except ValueError as e:
        print(f"[ERROR] Could not parse JSON for {url}: {e}")
        return []

    # The first row holds field names; remaining rows hold values
    if not data or len(data) < 2:
        return []

    headers = data[0]  # Example: ["urlkey", "timestamp", "original", ...]
    rows = data[1:]

    results = []
    for row in rows:
        record = dict(zip(headers, row))  # Map field names to values
        results.append(record)
    return results


def group_snapshots_by_week(snapshots):
    """Return a dictionary keyed by (year, iso_week) with lists of snapshots."""
    grouped = defaultdict(list)
    for snap in snapshots:
        ts_str = snap["timestamp"]                # Timestamp as string
        dt = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")  # To datetime
        year, iso_week, _ = dt.isocalendar()       # Extract ISO week
        grouped[(year, iso_week)].append(snap)     # Append snapshot to bucket
    return grouped


# Download html func

def download_with_retries(snapshot):
    """Download archived HTML for *snapshot* with retry logic."""

    ts = snapshot["timestamp"]
    original_url = snapshot["original"]
    archived_url = f"https://web.archive.org/web/{ts}/{quote(original_url, safe='')}"  # Fully qualified archived link

    print(f"Downloading {archived_url} ...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(archived_url, timeout=20)
            resp.raise_for_status()
            time.sleep(DOWNLOAD_SLEEP)  # Polite delay after success
            return resp.text, archived_url  # Return HTML and its source URL
        except requests.exceptions.RequestException as exc:
            print(f"[ERROR] Attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_SLEEP} seconds ...")
                time.sleep(RETRY_SLEEP)

    # All retries exhausted; caller handles the None result
    return None, archived_url

# Main
# ------

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # function to get the full range of weeks from date range if you want

    for link in URLS:

        print(f"\n=== Processing {link} ===")
        snapshots = fetch_cdx_snapshots(link, START_DATE, END_DATE)

        if not snapshots:
            print(f"  No snapshots found for {link} in that date range.")
            # Mark all weeks "missing"?
            # Or just store a single message that entire link is missing?

            missing_weeks[link].append(("NO_SNAPSHOTS", "N/A"))
            continue

        # Group
        grouped = group_snapshots_by_week(snapshots)

        # Find all (year, week) pairs that exist in the data

        all_weeks = list(grouped.keys())
        # Sort them by (year, iso_week)
        all_weeks.sort()

        # For each (year, week), pick earliest snapshot

        # We'll download for each group
        for (yr, wnum) in all_weeks:
            snap_list = grouped[(yr, wnum)]
            snap_list.sort(key=lambda x: x["timestamp"])
            earliest_snap = snap_list[0]

            # Build folder name
            folder_name = f"{yr}-week_{wnum}"
            folder_path = os.path.join(BASE_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Try to download
            html, archived_url = download_with_retries(earliest_snap)
            if html is None:
                # record this failure
                failures[link].append((yr, wnum, earliest_snap["timestamp"], archived_url))
                continue

            # Save success
            ts = earliest_snap["timestamp"]
            original_url = earliest_snap["original"]
            safe_name = (
                original_url.replace("https://", "")
                            .replace("/", "_")
                            .replace("?", "_")
                            .replace(":", "")
            )
            filename = f"{safe_name}__{ts}.html"
            out_path = os.path.join(folder_path, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"    -> Saved to {out_path}")

        # Identify gaps in ISO weeks for this URL
        if len(all_weeks) > 1:
            # gather the iso_week numbers
            sorted_weeks = [w for (y, w) in all_weeks]
            min_w, max_w = min(sorted_weeks), max(sorted_weeks)
            # any missing in [min_w..max_w]?
            for w in range(min_w, max_w+1):
                if (all_weeks[0][0], w) not in grouped:  # all_weeks[0][0] is the year (assuming single year)
                    missing_weeks[link].append((all_weeks[0][0], w))

    # End of for link in URLS

    # -------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------
    print("\n=======================")
    print("SUMMARY OF FAILURES:")
    print("=======================")
    if not failures:
        print("No download failures encountered.")
    else:
        for link, fail_list in failures.items():
            print(f"URL: {link}")
            for (yr, wnum, ts, arch_url) in fail_list:
                print(f"  - Week={wnum}, Timestamp={ts}, archived_url={arch_url}")
            print()

    print("=======================")
    print("SUMMARY OF MISSING WEEKS:")
    print("=======================")
    if not missing_weeks:
        print("No missing weeks recorded.")
    else:
        for link, missing_list in missing_weeks.items():
            print(f"URL: {link}")
            for tup in missing_list:
                print(f"  - {tup}")
            print()

    print("\nDone.")
    print("Scrape these missing/failing items manually")

if __name__ == "__main__":
    main()
