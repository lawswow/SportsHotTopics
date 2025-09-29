"""
Creates a weekly timeline of RAKE‑extracted keyword topics.
1. Reads a CSV file containing RAKE phrases and scores.
2. Maps each phrase to a high‑level topic based on regular expression patterns.
3. Counts how many top phrases of each topic occur in every week.
4. Saves the counts table and plots a stacked bar chart showing topic volume over time.
"""

# Standard library imports
import re  # Regular expressions for pattern matching
import pathlib  # Path handling that is OS independent

# Third‑party imports
import pandas as pd  # Data loading and manipulation
import matplotlib.pyplot as plt  # Chart creation

# 1. File locations ------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent  # Folder where the script resides
IN_CSV = ROOT / "rake_keywords.csv"  # Input produced by rake_model.py; expected columns: week, phrase, score
OUT_CSV = ROOT / "rake_tag_counts.csv"  # Output table with weekly tag counts
FIG_PNG = ROOT / "rake_timeline.png"  # File path for the generated plot

# 2. Tagging table (regex pattern mapped to tag label) -------------------
# Each tuple holds a single combined pattern and the tag that should be
# assigned when the pattern is found inside a phrase. Patterns are kept in
# lowercase so each phrase is converted to lowercase before matching.
TAG_RULES = [
    # Ball sports
    (r"premier|league|arsenal|liverpool|man utd|utd|chelsea|city|football|alexander|birmingham|brentford|brighton|burnley|crystal|fc|gunners|havertz|icons|jim|ratcliffe|rooney|torres|trent|villa|wayne|wright|dijk|wanderers|wolverhampton",
     "FOOTBALL"),

    (r"cricket|wicket|bowler|stokes|root|anderson|ashes|mitchell|stuart",
     "CRICKET"),

    (r"rugby|scrum|try|six nation|premiership|zammit|vunipola|chris|harris",
     "RUGBY"),

    (r"nba|basketball|march|madness",
     "BASKETBALL"),

    # Racket sports
    (r"wimbledon|djokovic|nadal|alcaraz|raducanu|tennis|wozniacki",
     "TENNIS"),

    (r"pga|master|mcilroy|scheffler|birdie|bogey|liv|golf|open|courses|dragged|dustin|johnson|masters|mcilroy|rory",
     "GOLF"),

    # Fight sports
    (r"tyson|fury|usyk|joshua|boxing|coma|conor|induced|logan|looking|mcgregor|mike|paul|touching|tyson",
     "BOXING"),

    (r"ufc|mma|octagon|conor mcgregor|khabib|fight card|caroline|daniel|mcgregor|mma|stoppage",
     "MMA"),

    # Motorsport
    (r"nascar|daytona|500|verstappen|hamilton|prix|formula|f1|car|hamilton|lanka|lewis|martin|max|ricciardo",
     "MOTORSPORT"),

    # Olympics
    (r"olympic|gold|medal|paris|2024",
     "OLYMPICS"),

    # Transfers and contracts
    (r"transfer|window|signing|contract|bid|retirement|5m|new|signs|transfers",
     "TRANSFERS"),
]

DEFAULT_TAG = "OTHER"  # Tag assigned when no rule matches

# 3. Load RAKE data ------------------------------------------------------
print(f"Reading {IN_CSV} ...")
df = pd.read_csv(IN_CSV)  # Load all rows and columns

# Convert phrases to lowercase for case‑insensitive matching
# Using str accessor keeps operation vectorised and fast
df["phrase"] = df["phrase"].str.lower()

# 4. Phrase‑to‑tag function ---------------------------------------------
def tag_phrase(phrase: str) -> str:
    """Return the first tag whose pattern matches the input phrase."""
    for pattern, tag in TAG_RULES:
        if re.search(pattern, phrase):
            return tag
    return DEFAULT_TAG

# Apply tagging function across the DataFrame; result stored in a new column
# This step enriches each phrase with its semantic category
df["tag"] = df["phrase"].apply(tag_phrase)

# 5. Build week‑by‑tag count table --------------------------------------
# Group by both week and tag, count rows, convert to matrix form where each
# column is a tag and each row is a week label. Missing combinations are set
# to zero for cleaner analysis.
counts = (
    df.groupby(["week", "tag"]).size()
      .unstack(fill_value=0)
      .sort_index()
)

# Persist counts table for use in spreadsheets or further scripts
counts.to_csv(OUT_CSV)
print(f"Tag counts written to {OUT_CSV}")

# 6. Plot stacked bar timeline ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))  # Figure size in inches

# Pandas built‑in plot produces stacked bars when stacked=True
counts.plot(kind="bar", stacked=True, ax=ax)

# Axis labels and title clarify the chart content
ax.set_title("Top RAKE Phrases per Week grouped by Topic")
ax.set_ylabel("Number of phrases among weekly top 10")
ax.set_xlabel("Week label")

# Place the legend outside the plot area to reduce clutter
ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()  # Adjust layout so everything fits without overlap
fig.savefig(FIG_PNG, dpi=300)  # High‑resolution save for print use
print(f"Timeline chart saved to {FIG_PNG}")

plt.show()  # Display the figure when running interactively
