import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Standard‑library modules for CLI handling, paths, dates and opening the browser
import argparse
import pathlib
import datetime
import sys
import webbrowser

# Third‑party libraries for data handling and topic modelling
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore

# Paths / filenames (edit these if you move the input or output files)
ROOT = pathlib.Path(__file__).resolve().parent
TOK_PARQUET = ROOT / "cleaned_tokens.parquet"   # Fast parquet version of token table
TOK_CSV = ROOT / "cleaned_tokens.csv"           # CSV fallback if parquet is missing

#  Helper functions

def load_tokens() -> pd.DataFrame:
    """Load the tokenised corpus produced by preprocess.py.

    The function first attempts to load the parquet file for speed.
    If that is not present it falls back to the CSV.
    The program terminates with an error message if neither file exists.
    """
    if TOK_PARQUET.exists():
        return pd.read_parquet(TOK_PARQUET)
    if TOK_CSV.exists():
        return pd.read_csv(TOK_CSV)
    print("No token file found", file=sys.stderr)
    sys.exit(1)


def build_corpus(df: pd.DataFrame):
    """Convert the DataFrame of space‑separated token strings into a Gensim corpus.

    Returns
    -------
    dictionary : gensim.corpora.Dictionary
        Maps each unique word to an integer id.
    corpus : list of list of (int, int)
        Bag‑of‑words representation expected by Gensim topic models.
    """
    token_lists = [str(t).split() for t in df["tokens"].fillna("")]
    dictionary = Dictionary(token_lists)

    # Filter out very rare words (appear in <5 documents) and very common words (appear in >50% documents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    return dictionary, corpus


def save_topics_csv(model, dictionary, path: pathlib.Path, topn: int = 20):
    """Write the top 'topn' words for every topic to a flat CSV file."""
    rows = [
        {"topic": topic_id, "word": word, "weight": round(weight, 4)}
        for topic_id in range(model.num_topics)
        for word, weight in model.show_topic(topic_id, topn=topn)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

#Main script##

def main():
    """Train an LDA topic model and open an interactive pyLDAvis visualisation."""
    # Parse small command‑line interface
    ap = argparse.ArgumentParser(description="Train LDA and visualise with pyLDAvis")
    ap.add_argument("-k", "--topics", type=int, default=15, help="Number of topics")
    ap.add_argument("-p", "--passes", type=int, default=50, help="Number of training passes (epochs)")
    ap.add_argument("-w", "--workers", type=int, default=4, help="Number of CPU cores to use")
    args = ap.parse_args()

    #Load pre‑tokenised documents and build the BoW corpus
    df = load_tokens()
    print(f"Loaded {len(df):,} documents...")
    dictionary, corpus = build_corpus(df)
    print(f"Corpus: {len(dictionary):,} unique tokens -> {len(corpus):,} BoW rows")

    # Choose single‑core or multi‑core LDA implementation based on --workers parameter
    Model = LdaMulticore if args.workers > 1 else LdaModel
    lda = Model(
        corpus=corpus,
        id2word=dictionary,
        num_topics=args.topics,
        passes=args.passes,
        workers=args.workers,
        random_state=42,  # Reproducible results
    )

    # Create a timestamped output folder for the model artefacts
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = ROOT / "models" / f"lda_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    lda.save(str(out_dir / "model.gensim"))

    # Export topic-word table for quick inspection in a spreadsheet
    save_topics_csv(lda, dictionary, out_dir / "topics.csv")

    # Show a short textual preview of each topic in the console
    for topic_id, topic in lda.print_topics():
        print(f"{topic_id:02d}: {topic}")

    # Build and save the interactive pyLDAvis HTML page
    print("Building pyLDAvis panel...")
    vis_data = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)

    html_path = out_dir / "ldavis.html"
    pyLDAvis.save_html(vis_data, str(html_path))

    print(f"pyLDAvis saved to {html_path}")
    webbrowser.open_new_tab(html_path.as_uri())


if __name__ == "__main__":
    main()
