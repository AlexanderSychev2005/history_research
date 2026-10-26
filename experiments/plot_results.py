import csv
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def read_csv(path):
    with open(ROOT / path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_figure(name):
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: experiments/figures/{name}.png and .pdf")


# ------------------------------------------------
# Figure 1: Comparison of normalization experiments
# ------------------------------------------------

alias = read_csv("01_alias_diagnostic/results.csv")
unicode_rows = read_csv("02_unicode_normalization/results.csv")

def count_correct(rows, column):
    return sum(row[column].lower() == "true" for row in rows)

labels = [
    "Original\nmentions",
    "Unicode\nnormalization",
    "Canonical YAGO\nnames (oracle)",
]

values = [
    count_correct(alias, "baseline_correct"),
    count_correct(unicode_rows, "normalized_correct"),
    count_correct(alias, "modified_correct"),
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    values,
    color=["#A5ACB8", "#5B8DB8", "#3B9278"],
    width=0.55,
)

ax.set_title("Historical Name Linking Experiments", fontsize=15, pad=16)
ax.set_ylabel("Correctly linked mentions")
ax.set_ylim(0, 5.8)
ax.set_yticks(range(6))
ax.spines[["top", "right"]].set_visible(False)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.15,
        f"{value}/5",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

fig.text(
    0.5, -0.02,
    "Five selected mentions; canonical replacements use gold-standard knowledge.",
    ha="center", fontsize=9, color="#555555"
)

save_figure("01_historical_name_experiments")


# ------------------------------------------------
# Figure 2: Candidate generation error analysis
# ------------------------------------------------

rows = read_csv("03_candidate_analysis/results.csv")
linkable = [r for r in rows if r["gold_entity"]]

counts = Counter()

for row in linkable:
    if row["detected"] != "True":
        counts["Mention not detected"] += 1
    elif row["correct"] == "True":
        counts["Correctly linked"] += 1
    elif row["gold_in_candidates"] == "True":
        counts["Gold candidate present,\nnot selected"] += 1
    else:
        counts["Gold candidate missing"] += 1

categories = [
    "Correctly linked",
    "Gold candidate present,\nnot selected",
    "Gold candidate missing",
    "Mention not detected",
]

values = [counts[c] for c in categories]

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.barh(
    categories,
    values,
    color=["#3B9278", "#E0AE50", "#C86661", "#A5ACB8"],
    height=0.6,
)

ax.invert_yaxis()
ax.set_title("Entity Linking Outcomes", fontsize=15, pad=16)
ax.set_xlabel("Number of gold-linkable mentions")
ax.set_xlim(0, max(values) + 3)
ax.spines[["top", "right"]].set_visible(False)

for bar, value in zip(bars, values):
    ax.text(
        value + 0.2,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
        fontsize=12,
        fontweight="bold",
    )

fig.text(
    0.5, 0.01,
    f"LELA baseline | {len(linkable)} gold-linkable mentions | "
    "Alford 2026 + Nechaeva 2017",
    ha="center", fontsize=9, color="#555555"
)

save_figure("02_entity_linking_outcomes")

print("\nAll figures generated successfully.")
