
# lda_weekly_model.py  -  mini LDA per ISO week + bar and heat maps
import csv, pathlib
from collections import defaultdict

import pandas as pd
from gensim import corpora, models
import matplotlib.pyplot as plt #barcharts
import seaborn as sns                      # heatmaps

# -------------- CONFIG ---------------
DATA_CSV = pathlib.Path(__file__).with_name("cleaned_tokens.csv")
OUT_CSV = pathlib.Path(__file__).with_name("lda_weekly_topics.csv")

NUM_TOPICS = 5           # topics per week
WORDS_PER = 10          # how many words to list in console and csv
MIN_DOCS = 2           # skip weeks with fewer pages
MIN_TOKENS = 20          # skip pages with too few tokens

FIG_DIR    = pathlib.Path(__file__).with_name("figures")
FIG_DIR.mkdir(exist_ok=True)


#            Load cleaned tokens
print(f"Reading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV, usecols=["week", "tokens"])
df["tokens"] = df["tokens"].fillna("").apply(str.split)
df = df[df["tokens"].apply(len) >= MIN_TOKENS]

#         collect pages by week
docs_by_week: dict[str, list[list[str]]] = defaultdict(list)
for _, row in df.iterrows():
    docs_by_week[row["week"]].append(row["tokens"])

print(f"Found {len(docs_by_week)} calendar weeks\n")

rows_for_csv = []     # lines for lda_weekly_topics.csv
topic_sizes = {}     # week -> list of topic totals
processed = 0



# Fit one tiny LDA per week and draw a bar chart
for week, pages in sorted(docs_by_week.items()):

    if len(pages) < MIN_DOCS:
        print(f"{week}: skipped - only {len(pages)} page(s)")
        continue

    # Build gensim dictionary and corpus
    dictionary = corpora.Dictionary(pages)
    dictionary.filter_extremes(no_below=2,
                               no_above=0.9 if len(pages) > 1 else 1.0)
    if len(dictionary) == 0:
        print(f"{week}: skipped - dictionary empty after filtering")
        continue
    corpus = [dictionary.doc2bow(doc) for doc in pages]

    # pass args by keyword, no duplicate num_topics
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=30,
        random_state=42,
        minimum_probability=0      # list all words in show_topic
    )

    # Show and store top words
    print(f"\n=== {week} - top {WORDS_PER} words per topic ===")
    for tid in range(NUM_TOPICS):
        words = lda.show_topic(tid, topn=WORDS_PER)
        nice = ", ".join(w for w, _ in words)###
        print(f"Topic {tid:>2}: {nice}")
        rows_for_csv.append([week, tid, " ".join(w for w, _ in words)])

    # Topic prevalence for bar chart
    sizes = [0.0] * NUM_TOPICS
    for bow in corpus:
        for tid, prob in lda.get_document_topics(bow, minimum_probability=0):
            sizes[tid] += prob
    topic_sizes[week] = sizes

    # Draw and save bar chart
    plt.figure(figsize=(6, 3))
    plt.bar(range(NUM_TOPICS), sizes, color="skyblue")
    plt.title(f"Topic prevalence - {week}")
    plt.xlabel("topic id")
    plt.ylabel("sum gamma(topic | doc)")
    plt.xticks(range(NUM_TOPICS))
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{week}_bars.png")
    plt.close()

    processed += 1



#  Write summary CSV
# -------------------------
if rows_for_csv:
    print(f"\nWriting -> {OUT_CSV}")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["week", "topic_id", f"top_{WORDS_PER}_words"])
        wr.writerows(rows_for_csv)
else:
    print("\n[WARN] No weeks were processed - CSV not written")

#  Heat map across weeks
# -----------------------
if topic_sizes:
    heat_df = pd.DataFrame.from_dict(topic_sizes, orient="index") \
                           .sort_index()          # rows sorted chronologically

    plt.figure(figsize=(8, 4))
    sns.heatmap(heat_df.T, cmap="YlGnBu", cbar_kws=dict(label="sum gamma"))
    plt.title("Topic prevalence heat map across weeks")
    plt.xlabel("calendar week")
    plt.ylabel("topic id")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap.png")
    plt.close()

    print(f"Figures saved into folder: {FIG_DIR}")
else:
    print("[WARN] No visual data collected - heat map skipped")

print(f"\nDone - {processed} weeks processed OK")
