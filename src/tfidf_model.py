"""
1. Read a CSV file of cleaned tokens (one row per article with a week label).
2. Combine rows that belong to the same week into a single document string.
3. Fit a scikit-learn TfidfVectorizer over the weekly corpus.
4. Print the top terms for a manual sanity check.
5. Build a word cloud image for every week using the TF-IDF weights.
6. Pickle the trained vectoriser and store the TF-IDF matrix on disk.
"""

# Standard library imports
import pathlib  # Path operations independent of operating system
import pickle   # Serialise Python objects
import warnings # Silence specific library warnings
from collections import Counter  # Quick top-N frequency extraction

# Third‑party imports
import numpy as np  # Numerical arrays and helpers
import pandas as pd  # Data loading and grouping
import scipy.sparse as sp  # Efficient storage for sparse matrices
from sklearn.feature_extraction.text import TfidfVectorizer  # TF‑IDF model
from wordcloud import WordCloud  # Word cloud generation from term weights
import matplotlib.pyplot as plt  # Image preview
from matplotlib import font_manager  # Locate a TrueType font

# Configuration values
ROOT = pathlib.Path(__file__).resolve().parent  # Folder containing this script
DATA_CSV = ROOT / "cleaned_tokens.csv"  # Source tokens with columns week,tokens

MODEL_PATH = ROOT / "tfidf_weekly.pkl"   # Output for the fitted vectoriser
MATRIX_PATH = ROOT / "tfidf_weekly.npz"  # Output for the TF-IDF matrix
WC_DIR = ROOT / "wordclouds"              # Folder where PNG images are saved

TOP_N = 15          # Number of terms shown per week and kept in each cloud
MIN_DF = 2          # Term must appear in at least this many weeks
MAX_FEATS = 10000   # Safety cap on the vocabulary size
WC_SIZE = (480, 320)  # Width and height of word cloud images in pixels

# Load the token data and build one document per calendar week
print(f"Reading {DATA_CSV} …")
df = pd.read_csv(DATA_CSV, usecols=["week", "tokens"])##


# Combine all rows that share the same week into a single whitespace string
after_group = (
    df.groupby("week")["tokens"].apply(lambda col: " ".join(col.dropna().astype(str)))
)
weekly_docs = after_group.sort_index()  # Keep chronological order

print(f"Aggregated into {len(weekly_docs)} weekly documents")

# Fit the TF-IDF vectoriser; silencing an irrelevant tokenizer warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The parameter 'token_pattern'"###
)

vec = TfidfVectorizer(
    tokenizer=str.split,      # Tokens already separated by space
    preprocessor=lambda x: x, # Input strings remain unchanged
    lowercase=False,          # Tokens are already lower case
    min_df=MIN_DF,
    max_features=MAX_FEATS
)

print("Fitting TF-IDF …")
tfidf_mat = vec.fit_transform(weekly_docs)  # Rows map to weeks
terms = vec.get_feature_names_out()
print(f"Corpus = {tfidf_mat.shape[0]} weeks * {tfidf_mat.shape[1]} terms")

# Print the top TF-IDF weighted terms for each week as a sanity check
for row, week_lbl in zip(tfidf_mat, weekly_docs.index):
    pairs = zip(row.indices, row.data)  # (column_id, weight) tuples
    top = sorted(pairs, key=lambda t: t[1], reverse=True)[:TOP_N]
    pretty = ", ".join(f"{terms[i]} ({w:.3f})" for i, w in top)
    print(f"\n{week_lbl} top {TOP_N} terms")
    print(pretty or "(empty)")

# Create one word cloud per week using TF-IDF weights
print("\nGenerating word clouds …")
WC_DIR.mkdir(exist_ok=True)

# WordCloud requires a TrueType font; DejaVu Sans ships with matplotlib
font_path = font_manager.findfont("DejaVu Sans")

for row, week_lbl in zip(tfidf_mat, weekly_docs.index):
    bow = {terms[i]: float(w) for i, w in zip(row.indices, row.data)}
    if not bow:
        print(f"  {week_lbl} skipped (no terms after min_df filter)")
        continue

    bow_topN = dict(Counter(bow).most_common(TOP_N))  # Reduce clutter

    wc = WordCloud(
        width=WC_SIZE[0],
        height=WC_SIZE[1],
        background_color="white",
        prefer_horizontal=1.0,
        max_words=TOP_N,
        font_path=font_path
    ).generate_from_frequencies(bow_topN)

    out_png = WC_DIR / f"{week_lbl}.png"
    wc.to_file(out_png)
    print(f"  saved {out_png.name}")

    # preview window; comment out if running headless
    plt.figure(figsize=(WC_SIZE[0] / 96, WC_SIZE[1] / 96), dpi=96)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(week_lbl)
    plt.tight_layout()
    plt.show(block=False)

plt.close("all")  # Close all preview figures

# Persist the trained vectoriser and sparse matrix
print(f"\nPickling vectoriser -> {MODEL_PATH.name}")
MODEL_PATH.write_bytes(pickle.dumps(vec))

print(f"Saving TF-IDF matrix -> {MATRIX_PATH.name}")
sp.save_npz(MATRIX_PATH, tfidf_mat)

print("All done, check the 'wordclouds' folder")
