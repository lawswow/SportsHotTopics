
# 1. Read token data grouped by week.
# 2. Build a CountVectorizer on the weekly corpus.
# 3. Collect per week top words and statistics.
# 4. Save reports to CSV.
# 5. Create and save a corpus word cloud image.


import pathlib  # Path operations
import csv  # Writing CSV files
import numpy as np  # Numerical operations
import pandas as pd  # Data handling
from sklearn.feature_extraction.text import CountVectorizer  # Bag-of-words model
from wordcloud import WordCloud  # Word cloud generation
import matplotlib.pyplot as plt  # preview of the word cloud

# User parameters
ROOT = pathlib.Path(__file__).resolve().parent  # Directory where the script lives
DATA_CSV = ROOT / "cleaned_tokens.csv"  # Source file with weekly tokens
TOP_N = 15  # Number of high frequency words kept per week
WC_SIZE = (640, 400)  # Image dimensions for the word cloud in pixels

#Load data
print(f"Reading {DATA_CSV}")
df = pd.read_csv(DATA_CSV, usecols=["week", "tokens"])  # Keep only required columns

# CountVectorizer expects one string per document. Merge tokens for each week into one string.
weekly_docs = (
    df.groupby("week")["tokens"]
    .apply(lambda s: " ".join(s.dropna().astype(str)))
    .sort_index()
)

print(f"Found {len(weekly_docs)} weekly docs")

# Fit CountVectorizer
cv = CountVectorizer(
    tokenizer=str.split,  # Each token already separated by spaces
    preprocessor=lambda x: x,  # No additional preprocessing
    lowercase=False,  # Tokens are already lower case
    min_df=1  # Keep every token
)

X = cv.fit_transform(weekly_docs)  # Sparse matrix with shape (weeks x vocabulary)
words = cv.get_feature_names_out()  # Vocabulary list indexed by column position

# Prepare containers for output
rows_top = []  # Records top tokens per week
rows_stats = []  # Records basic counts per week

# Iterate through each week
for row_idx, week in enumerate(weekly_docs.index):
    row = X.getrow(row_idx)  # Sparse vector for this week
    inds = row.indices  # Indices of non zero counts
    freqs = row.data  # Frequencies at those indices

    if freqs.size == 0:  # Skip weeks without content
        continue

    unique = len(inds)  # Distinct tokens this week
    total = int(freqs.sum())  # Total token count
    rows_stats.append([week, unique, total])

    # Identify the most frequent tokens
    top_idx = np.argsort(freqs)[::-1][:TOP_N]
    for j in top_idx:
        word = words[inds[j]]
        cnt = int(freqs[j])##
        rows_top.append([week, word, cnt])

#  Write CSV outputs
with open(ROOT / "weekly_top_words.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["week", "word", "freq"], *rows_top])

with open(ROOT / "weekly_stats.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["week", "unique_words", "total_words"], *rows_stats])

print("CSV files written")

# Build corpus level word cloud
corpus_counts = np.asarray(X.sum(axis=0)).ravel()  # Total counts for each token
freq_dict = {word: int(cnt) for word, cnt in zip(words, corpus_counts) if cnt > 0}

wc = WordCloud(
    width=WC_SIZE[0],
    height=WC_SIZE[1],
    background_color="white",
    max_words=250,
    prefer_horizontal=1.0
).generate_from_frequencies(freq_dict)

png_path = ROOT / "corpus_wordcloud.png"
wc.to_file(png_path)
print(f"Word cloud saved: {png_path.name}")

# preview of the generated wordcloud image
plt.figure(figsize=(WC_SIZE[0] / 96, WC_SIZE[1] / 96), dpi=96)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Corpus Word Cloud (raw counts)")
plt.tight_layout()
plt.show(block=False)

print("All done")
