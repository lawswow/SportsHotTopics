import csv                       # write the keyword table to disk
import pathlib                   # handle filesystem paths independent of OS
import re                        # detect ISO week folder names via regex
from collections import defaultdict, OrderedDict  # collect docs and keep order
from datetime import datetime    # convert folder names to calendar weeks

from bs4 import BeautifulSoup    # strip HTML tags quickly
from rake_nltk import Rake       # keyword extraction algorithm
from nltk.corpus import stopwords  # base English stop‑word list

# Configuration ---
ROOT = pathlib.Path(__file__).parent            # project root directory
ARCHIVE_DIR = ROOT / "data"                    # folders named by week live here
OUT_CSV = ROOT / "rake_keywords.csv"           # final output file

TOP_N = 10                                      # number of phrases kept per week
MIN_WORDS = 2                                   # ignore single‑word phrases
MAX_WORDS = 6                                   # ignore very long phrases

# Extra stop words ---------------------------
# Custom domain‑specific words that should never appear in a key phrase
CUSTOM_STOP = {
    # topical words removed because they dominate without adding meaning
    "black", "british", "south", "asian", "asians", "lives", "matter", "football",
    "posted", "ago", "today", "yesterday", "tomorrow", "tonight",
    "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "attribution",
    # presentation/ filler words
    "update", "updates", "updated", "live", "latest", "breaking", "highlight",
    "highlights", "video", "videos", "watch", "read", "click", "share", "story",
    "stories", "report", "reports", "reaction", "analysis", "recap", "edition",
    "news", "homepage", "comment", "following", "follow", "close", "panel",
    "app", "collection", "preview", "topic", "appear",
    # site markers
    "bbc", "skysports", "guardian", "mirror", "com", "co", "uk", "http", "https",
    "sky", "twitter", "facebook",
    # age markers
    "1d", "2d", "3d", "4d", "5d", "6d", "1h", "2h", "3h", "4h", "5h", "6h", "7h",
    "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h",
    "19h", "20h", "21h", "22h", "23h", "commented", "comments",
}

# Helpers -------
# Regular expression that matches folder names such as "2024-week_14"
WEEK_RE = re.compile(r"^\d{4}-week_\d{2}$")

def week_of(name: str) -> str:
    """Return an ISO week label for a folder name.

    If *name* already matches the pattern "YYYY-week_XX" it is returned
    unchanged. If *name* is an ISO date in the form "YYYY-MM-DD" the function
    converts it to the corresponding ISO week formatted as "YYYY-WXX". Any
    other input is returned unchanged to avoid a hard failure on unexpected
    directory names.
    """
    if WEEK_RE.match(name):
        return name
    try:
        dt = datetime.strptime(name, "%Y-%m-%d")
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    except ValueError:
        return name  # leave odd folder names untouched

def html_to_text(html: str) -> str:
    """Convert raw HTML into plain text suitable for RAKE.

    Script, style and common page chrome tags are removed before text
    extraction to reduce noise.
    """
    soup = BeautifulSoup(html, "lxml")
    for tg in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tg.decompose()
    return soup.get_text(separator=" ", strip=True)#

# Collect pages per week -----------
print(f"Scanning {ARCHIVE_DIR} ...")
weekly_docs = defaultdict(list)  # maps week label to list of page texts

for week_dir in ARCHIVE_DIR.iterdir():
    if not week_dir.is_dir():
        continue
    week = week_of(week_dir.name)
    for html in week_dir.rglob("*.html"):
        try:
            txt = html_to_text(html.read_text(encoding="utf-8", errors="ignore"))
            if txt:
                weekly_docs[week].append(txt)
        except Exception as e:
            print(f"[WARN] {html} skipped ({e})")

print(f"Collected text for {len(weekly_docs)} calendar weeks.")

# Process week by week --------
rows = []
# combined stop list: built‑in English words plus the custom additions above
base_stop = set(stopwords.words("english")) | CUSTOM_STOP

for week, pages in sorted(weekly_docs.items()):
    # concatenate all pages from the week into one text blob
    full = "\n".join(pages)

    # create a fresh RAKE instance so internal state does not leak between weeks
    rake = Rake(stopwords=base_stop, min_length=MIN_WORDS, max_length=MAX_WORDS)

    # run keyword extraction
    rake.extract_keywords_from_text(full)

    # RAKE returns unique phrases but normalisation is applied for safety
    unique: dict[str, float] = OrderedDict()
    for score, phrase in rake.get_ranked_phrases_with_scores():
        key = phrase.lower().strip()
        unique.setdefault(key, score)  # keep the first (highest‑scoring) hit

    best = list(unique.items())[:TOP_N]

    # diagnostic console output
    print(f"\n{week} - top {TOP_N} phrases:")
    for phrase, score in best:
        print(f"{score:6.2f} : {phrase}")
        rows.append([week, phrase, f"{score:.2f}"])

# Save CSV ------
print(f"\nWriting -> {OUT_CSV}")
with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["week", "phrase", "score"])
    writer.writerows(rows)

print("Process completed successfully")
