# Clean HTML snapshots, apply lemmatisation or stemming, and collect named
# entities. Output is written to cleaned_tokens.csv (plus an optional Parquet
# file).

import re
import csv
import pathlib
import itertools
from typing import List

import pandas as pd  # Tabular data handling
from bs4 import BeautifulSoup  # HTML parsing to plain text
import nltk  # Tokenisation and linguistic resources
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer


# spaCy is used for named entity recognition. function loads the model
# and downloads it if necessary.

def load_spacy():
    # Return a spaCy `nlp` object for English or `None` if loading fails.
    import spacy
    from spacy.util import is_package
    from spacy.cli import download

    model = "en_core_web_sm"
    try:
        if not is_package(model):
            print("[INFO] downloading spaCy model ...")
            download(model)
        return spacy.load(model, disable=["parser", "textcat"])
    except Exception as err:
        print("[WARN] spaCy unavailable:", err)
        return None

#
# Initialise the spaCy pipeline once at import time
NLP = load_spacy()

# Ensure required NLTK resources are present. Download silently if missing.
for res in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(res if res.startswith("token") else f"corpora/{res}")
    except LookupError:
        nltk.download(res)

# Ready to use objects from NLTK
TOKENISER = nltk.word_tokenize
BASE_STOPWORDS = set(stopwords.words("english"))

# Additional stop words specific to news and sports domains
EXTRA_STOP = {
    "attribution", "posted", "update", "updates", "updated", "live",
    "latest", "breaking", "highlight", "highlights", "video", "videos",
    "watch", "read", "click", "share", "story", "stories", "report",
    "reports", "reaction", "analysis", "recap", "edition", "news",
    "homepage", "comment", "following", "follow", "close", "panel",
    "app", "collection", "preview", "topic", "appear",

    # Site or domain names
    "bbc", "skysports", "guardian", "mirror", "com", "co", "uk", "http",
    "https", "sky", "twitter", "facebook",

    # Generic sports vocabulary
    "sport", "sports", "division", "score", "scores", "result", "results",
    "table", "standings", "match", "matches", "fixture", "rugby", "union",
    "football", "cricket", "golf", "tennis", "athletics", "darts",
    "snooker", "podcasts", "mobile", "contact", "work", "term", "condition",

    # Temporal words
    "ago", "today", "yesterday", "tomorrow", "tonight", "minute", "minutes",
    "hour", "hours", "day", "days", "week", "weeks", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",

    # Miscellaneous
    "amp", "bet", "odds", "cookies", "privacy", "consent", "sign", "sign‑in",
    "signin", "boxing", "science", "racing", "betting", "crime", "travel",

    # Mirror specific filler
    "blog", "opinion", "online", "u", "irish", "world", "weird", "real",
    "life", "hopeful", "teamdogs", "area", "politics", "health", "weather",
    "royal", "money", "tech", "ufc", "ireland", "europe", "usa", "canada",
    "caribbean", "africa", "cruise", "cheap", "flight", "asia", "middle",
    "east", "australia", "zealand", "central", "south", "america", "lifestyle",
    "family", "fashion", "beauty", "sex", "relationship", "food", "drink",
    "gaming", "gardening", "celebs", "tv", "film", "celebrity", "partner",
    "bingo", "cartoon", "competition", "crossword", "dating", "funeral",
    "notice", "horoscope", "offer", "newsletter", "signup", "voucher",
    "code", "asos", "nike", "argo", "john", "lewis", "curry", "booking",
    "shein", "prettylittlething", "groupon", "boohoo", "choice", "search",
    "social", "got", "shop", "bull", "return", "topics", "gossip"
}

LEMMATISER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# Toggle to switch between stemming and lemmatisation
USE_STEMMING = False

# Path configuration ---------------------------------------------------------
DATA_ROOT = pathlib.Path(r"C:\Users\user\Documents\SportsHotTopics\src\data").expanduser()
OUTPUT_CSV = pathlib.Path(__file__).with_name("cleaned_tokens.csv")


# Helper functions -----------------------------------------------------------

def html_to_text(html: str) -> str:
    """Convert raw HTML to plain text suitable for tokenisation."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non content tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # collapse whitespace to single spaces
    return soup.get_text(separator=" ", strip=True)


ALL_STOPWORDS = BASE_STOPWORDS | EXTRA_STOP

NON_ALPHA = re.compile(r"[^a-zA-Z\s]")  # Anything not a letter or space
TOKEN_RE = re.compile(r"[a-z]{2,}")  # Tokens must be at least two letters


def normalise_token(tok: str) -> str:
    """Return the lemma or stem of a token (input is already lower‑case)."""
    if USE_STEMMING:
        return STEMMER.stem(tok)
    # WordNet lemmatiser needs POS, but default=‘n’ works
    return LEMMATISER.lemmatize(tok)##


def clean_and_tokenize(text: str) -> List[str]:
    text = NON_ALPHA.sub(" ", text.lower())  # keep only letters & spaces
    raw_tokens = TOKENISER(text)

    tokens: List[str] = []
    for tok in raw_tokens:
        if not TOKEN_RE.fullmatch(tok):  # throw away 1-char tokens, etc
            continue
        norm = normalise_token(tok)  # lemmatise/stem
        if norm in ALL_STOPWORDS:  # then test the *normalised* form
            continue
        tokens.append(norm)

    return tokens


def extract_entities(text: str) -> List[str]:
    """Return unique named entities of selected types using spaCy."""
    if NLP is None:
        return []

    doc = NLP(text)
    wanted = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
    ents = [ent.text for ent in doc.ents if ent.label_ in wanted]
    return list(dict.fromkeys(ents))  # Preserve original order while deduplicating


# Main processing pipeline ---------------------------------------------------

def process_all() -> None:
    html_files = sorted(DATA_ROOT.rglob("*.html"))
    total = len(html_files)
    print_every = 3  # Progress message frequency

    print(f"Found {total} HTML files.\n")

    rows = []  # Each row becomes one CSV line, # each row: week, outlet, file, tokens_str, entities_str

    for i, html_path in enumerate(html_files, 1):
        # progress message
        if i == 1 or i % print_every == 0 or i == total:
            pct = (i / total) * 100
            print(f"Processing {i:>4}/{total} ({pct:5.1f}%) -> {html_path.name}")

        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            plain = html_to_text(html)
            tokens = clean_and_tokenize(plain)
            ents = extract_entities(plain)
        except Exception as exc:
            print(f"[WARN] {html_path} skipped ({exc})")
            continue

        week = html_path.parent.name
        outlet = html_path.stem.split("__")[0]

        rows.append([
            week,
            outlet,
            html_path.name,
            " ".join(tokens),  # Tokens separated by spaces
            "|".join(ents)  # Entities separated by pipes
        ])

    # write CSV
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["week", "outlet", "file", "tokens", "entities"])
        writer.writerows(rows)

    print(f"\nDone - {len(rows)} rows written to {OUTPUT_CSV}")

    # Parquet version for faster downstream loading
    df = pd.DataFrame(rows, columns=["week", "outlet", "file", "tokens", "entities"])
    parquet_path = OUTPUT_CSV.with_suffix(".parquet")
    df.to_parquet(parquet_path, compression="snappy")
    print(f"Parquet corpus written -> {parquet_path}")


# Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    process_all()
