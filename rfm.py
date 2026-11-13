"""
==============================================================================
RFM SEGMENTATION — RECENCY, FREQUENCY, MONETARY
==============================================================================
Course: Big Data and Machine Learning
Topic: Value-based customer segmentation with Python

Business question
-----------------
"We know what our customers buy. We do not know how best to talk to them, or
which of them are worth investing in."

RFM answers that with three numbers per customer and no machine learning at all:

    Recency    — how long ago was their last purchase?   (lower is better)
    Frequency  — how often do they buy?                  (higher is better)
    Monetary   — how much have they spent?               (higher is better)

Each customer is scored 1 (poor) to 5 (excellent) on each dimension, and the
three scores together form a customer profile. It is the cheapest useful
segmentation there is, and in practice a good predictor of who will respond to
the next campaign.

The lesson starts from a **transaction log** — one row per line item on one
invoice, the shape data actually arrives in — because turning that log into
three numbers per customer is most of the work and all of the judgement.

What RFM cannot do
------------------
It measures **outcomes** (purchases) only. It says nothing about *why* someone
behaves the way they do — no age, no region, no channel preference, no
psychographics. For that you need clustering (see ``clustering.py``), and to
decide *how much* a customer is worth going forward you need CLTV (``cltv.py``).

Learning objectives
-------------------
1. Aggregate a transaction log into R, F and M — the step that decides everything
2. Score each dimension on a 1-5 quintile scale, with recency reversed
3. Understand why the three scores are a *profile*, not a sum
4. Read the 2x2x2 RFM cube and name each of its eight octants
5. Apply a named business-rule ladder (Champions, At Risk, Lost, ...)
6. See how two reasonable rule sets disagree on the same customers
7. State the limits of RFM, and what to reach for next

Run it
------
    uv sync
    uv run python fetch_data.py     # once, to download the dataset
    uv run python rfm.py

    SHOW_FIGURES=0 uv run python rfm.py     # numbers only, no browser tabs

Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
==============================================================================
"""

# %% 1. SETUP
# ==============================================================================

import inspect
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rfm_scoring import (  # noqa: E402
    CUBE_NAMES,
    LADDER_RULES,
    build_rfm,
    rfm_cube,
    rfm_ladder,
    score_rfm,
)
from src.retail_data import prepare  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Figures open in a browser tab. Set SHOW_FIGURES=0 to run the numbers only —
# handy when you just want the printed output, or when presenting from a terminal.
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "1") != "0"


def show(fig) -> None:
    """Display a figure unless SHOW_FIGURES=0."""
    if SHOW_FIGURES:
        fig.show()


def show_source(fn) -> None:
    """Print the real source of a function, so the lesson can't drift from it.

    The scoring functions live in ``src/rfm_scoring.py`` (``cltv.py`` needs them
    too, and importing this script would re-run the whole lesson). Printing them
    here means the code you read is the code that just ran.
    """
    print(f"\n--- {fn.__module__}.{fn.__name__} " + "-" * (48 - len(fn.__name__)))
    print(inspect.getsource(fn))


# %% 2. LOAD THE TRANSACTION LOG
# ==============================================================================

print("=" * 78)
print("RFM SEGMENTATION")
print("=" * 78)

transactions, source = prepare()

print(
    "\nThat last step is a modelling decision, not housekeeping. This retailer sells to\n"
    "the public and to shops that resell its stock, and the two behave nothing alike —\n"
    "a reseller orders 96 of one item every fortnight. Left in, a few hundred businesses\n"
    "would dominate every euro figure and the 'customer segments' would end up describing\n"
    "the trade counter. Nothing in the data says which is which, so it is inferred from\n"
    "buying behaviour (see `is_wholesale`), and the cost is printed above rather than\n"
    "hidden: these lessons describe the consumers, who are roughly half the turnover."
)

print("\nWhat the data looks like — one row per product on one invoice:")
print(transactions.head(5).to_string(index=False))


# %% 3. FROM A TRANSACTION LOG TO THREE NUMBERS PER CUSTOMER
# ==============================================================================
# This is the step RFM lives or dies on. Everything after it is arithmetic on
# three columns; everything wrong with an RFM analysis usually happened here.

print("\n" + "-" * 78)
print("3. AGGREGATING THE LOG INTO R, F AND M")
print("-" * 78)
show_source(build_rfm)

SNAPSHOT = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = build_rfm(transactions, SNAPSHOT)

print(f"{len(transactions):,} line items  ->  {len(rfm):,} customers")
print(f"Snapshot date ('today'): {SNAPSHOT:%Y-%m-%d}")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1).to_string())

# Follow one customer through the aggregation, so the groupby is not a black box.
# Pick a busy one — a single-order customer would not show the invoice-vs-line-item
# distinction that the docstring above warns about.
example_id = rfm.nlargest(20, "Frequency").iloc[10]["Id"]
example_rows = transactions[transactions["Customer ID"] == example_id]
example_rfm = rfm[rfm["Id"] == example_id].iloc[0]

print(f"\nOne customer, worked through by hand — customer {example_id:.0f}:")
print(f"  {len(example_rows):>6,} line items in the log")
print(f"  {example_rows['Invoice'].nunique():>6,} distinct invoices        -> Frequency = "
      f"{example_rfm['Frequency']:.0f}")
print(f"  last invoice {example_rows['InvoiceDate'].max():%Y-%m-%d}, snapshot "
      f"{SNAPSHOT:%Y-%m-%d}  -> Recency  = {example_rfm['Recency']:.0f} days")
print(f"  sum of quantity x price          -> Monetary = {example_rfm['Monetary']:,.0f}")
print(f"\nNote the first two numbers: {len(example_rows):,} rows but only "
      f"{example_rows['Invoice'].nunique()} orders. Counting rows instead of")
print("invoices would have made this customer look "
      f"{len(example_rows) / example_rows['Invoice'].nunique():.0f}x more loyal than they are.")


# %% 4. SCORE EACH DIMENSION 1-5
# ==============================================================================


print("\n" + "-" * 78)
print("4. SCORING (1 = poor ... 5 = excellent)")
print("-" * 78)
show_source(score_rfm)

scored = score_rfm(rfm)
print(scored[["Id", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score",
              "RFM_Cell", "RFM_Sum"]].head(8).to_string(index=False))

print("\nEach dimension is split into five equal-sized groups:")
for dim, score_col in [("Recency", "R_Score"), ("Frequency", "F_Score"), ("Monetary", "M_Score")]:
    ranges = scored.groupby(score_col)[dim].agg(["min", "max", "count"])
    spans = ", ".join(f"{s}:[{r['min']:.0f}-{r['max']:.0f}]" for s, r in ranges.iterrows())
    print(f"  {dim:<10} {spans}")

# Why the profile beats the sum: same total, opposite customers. Pick the most
# crowded sum, then contrast its most-recent cell against its least-recent one.
print("\nWhy keep three digits instead of adding them up?")
busiest_sum = int(scored["RFM_Sum"].value_counts().idxmax())
tied = scored[scored["RFM_Sum"] == busiest_sum]
print(f"  {len(tied)} customers score a total of {busiest_sum}, spread over "
      f"{tied['RFM_Cell'].nunique()} different cells:")
for _, row in tied.loc[[tied["R_Score"].idxmax(), tied["R_Score"].idxmin()]].iterrows():
    print(f"    cell {row['RFM_Cell']}: last purchase {row['Recency']:>2.0f} days ago, "
          f"{row['Frequency']:>2.0f} purchases, EUR {row['Monetary']:>6.0f}")
print("  Same sum, opposite customers — and opposite next actions: one needs a thank-you,")
print("  the other needs a win-back. Adding the scores up destroys exactly that distinction.")
print(f"  {scored['RFM_Cell'].nunique()} of the {5**3} possible cells are occupied.")


# %% 5. THE RFM CUBE — EIGHT OCTANTS
# ==============================================================================
# Split each dimension in two (high/low) instead of five, and RFM becomes a cube
# with eight corners. Coarser than the 125 cells above, but each corner has a
# name and an obvious action attached, which is what makes it usable in a
# marketing meeting.

print("\n" + "-" * 78)
print("5. THE RFM CUBE (median split on each dimension)")
print("-" * 78)
print("The eight corners of the cube, as (Recency, Frequency, Monetary) go up or down:")
for (r_up, f_up, m_up), name in CUBE_NAMES.items():
    arrows = "".join(
        f"{dim}{'^' if up else 'v'} " for dim, up in zip("RFM", (r_up, f_up, m_up))
    )
    print(f"  {arrows}  {name}")

scored["Cube"] = rfm_cube(scored)
cube_table = (
    scored.groupby("Cube")
    .agg(Customers=("Id", "count"), Recency=("Recency", "mean"),
         Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"))
    .round(1)
    .sort_values("Monetary", ascending=False)
)
cube_table["Share"] = (cube_table["Customers"] / len(scored) * 100).round(1).astype(str) + "%"
print(cube_table.to_string())

# The octants are nowhere near equally populated, and that is worth a minute.
correlations = scored[["Recency", "Frequency", "Monetary"]].corr()
fm_corr = correlations.loc["Frequency", "Monetary"]
rm_corr = correlations.loc["Recency", "Monetary"]
rf_corr = correlations.loc["Recency", "Frequency"]
biggest_four = cube_table.nlargest(4, "Customers")["Customers"].sum() / len(scored)

print(f"\nThe eight octants are far from equally filled: the largest four hold "
      f"{biggest_four:.0%} of the base.")
print("The reason is in the correlation matrix — the cube assumes three independent")
print("dimensions, and they are not independent:")
print(correlations.round(2).to_string())
print(
    f"\nFrequency and Monetary correlate at {fm_corr:+.2f}: customers who buy often are "
    "usually the\nones who spend most, so 'frequent but cheap' and 'rare but expensive' are "
    "genuinely\nrare corners rather than quarters of the base.\n"
    f"\nRecency is the more independent axis ({rm_corr:+.2f} against Monetary, "
    f"{rf_corr:+.2f} against\nFrequency) — weak, and negative, meaning customers who bought "
    "recently do also tend to\nbuy more often. It is the dimension carrying the most new "
    "information, which is exactly\nwhy an 'At Risk' segment is worth defining at all."
)


# %% 6. THE BUSINESS LADDER — NAMED SEGMENTS
# ==============================================================================
# The other common approach: a ladder of business rules over the 1-5 scores.
# More expressive than the cube (rules can use all five levels), but the rules
# are ordered, so the first match wins and the ordering *is* the model.

print("\n" + "-" * 78)
print("6. THE BUSINESS LADDER (first matching rule wins)")
print("-" * 78)
print("The rules, in the order they are tested — the first match wins:")
for position, (name, rule) in enumerate(LADDER_RULES, start=1):
    # Each rule is a one-line lambda inside a tuple, so the tail is exactly "),".
    rule_source = inspect.getsource(rule).split("lambda r, f, m:")[1].strip().removesuffix("),")
    print(f"  {position}. {name:<20} {rule_source}")
print(f"  {len(LADDER_RULES) + 1}. Hibernating          (everyone the rules above did not claim)")

scored["Segment"] = rfm_ladder(scored)
ladder_table = (
    scored.groupby("Segment")
    .agg(Customers=("Id", "count"), R=("R_Score", "mean"), F=("F_Score", "mean"),
         M=("M_Score", "mean"), Recency=("Recency", "mean"), Monetary=("Monetary", "mean"))
    .round(1)
    .sort_values("Customers", ascending=False)
)
ladder_table["Share"] = (ladder_table["Customers"] / len(scored) * 100).round(1).astype(str) + "%"
print(ladder_table.to_string())
print(f"\nRevenue concentration: the customers scored M=5 are "
      f"{(scored['M_Score'] == 5).mean():.0%} of the base and "
      f"{scored.loc[scored['M_Score'] == 5, 'Monetary'].sum() / scored['Monetary'].sum():.0%} "
      f"of all spending.")


# %% 7. TWO RULE SETS, ONE CUSTOMER BASE
# ==============================================================================
# Both segmentations above are defensible. They do not agree. That is not a bug
# in either of them — it is the central caveat of rule-based segmentation: the
# rules are a business decision dressed up as an analysis.

print("\n" + "-" * 78)
print("7. CUBE vs LADDER — where the two disagree")
print("-" * 78)
overlap = pd.crosstab(scored["Segment"], scored["Cube"])
print(overlap.to_string())

top_cube = scored["Cube"] == "High-Value"
top_ladder = scored["Segment"] == "Champions"
both = int((top_cube & top_ladder).sum())
print(f"\n'High-Value' (cube): {int(top_cube.sum())} customers")
print(f"'Champions' (ladder): {int(top_ladder.sum())} customers")
print(f"In both:              {both} customers")
print("The cube's median split is more generous; the ladder's top rule demands 4+ "
      "on all three\nquintile scores. Which one you ship depends on how big a "
      "campaign you can afford.")


# %% 8. VISUALISATION
# ==============================================================================

# 8a. The three dimensions, coloured by ladder segment. Rotate it: the segments
# occupy visibly different corners of the space, which is the whole idea.
fig_3d = px.scatter_3d(
    scored,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Segment",
    opacity=0.75,
    title="RFM space, coloured by business-rule segment",
    labels={"Recency": "Recency (days)", "Frequency": "Purchases", "Monetary": "Spending (EUR)"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_3d.update_traces(marker=dict(size=3.5, line=dict(width=0)))
fig_3d.update_layout(height=700)
show(fig_3d)

# 8b. Segment sizes side by side — the two rule sets, same base.
fig_sizes = make_subplots(
    rows=1, cols=2, subplot_titles=("Business ladder", "RFM cube (8 octants)")
)
ladder_counts = scored["Segment"].value_counts().sort_values()
cube_counts = scored["Cube"].value_counts().sort_values()
fig_sizes.add_trace(
    go.Bar(x=ladder_counts.values, y=ladder_counts.index, orientation="h",
           marker_color="#4C78A8", showlegend=False),
    row=1, col=1,
)
fig_sizes.add_trace(
    go.Bar(x=cube_counts.values, y=cube_counts.index, orientation="h",
           marker_color="#F58518", showlegend=False),
    row=1, col=2,
)
fig_sizes.update_layout(height=450, title_text="Two segmentations of the same customers")
fig_sizes.update_xaxes(title_text="Customers")
show(fig_sizes)

# 8c. Mean score per segment. Reading down a column tells you what the segment
# name actually means in scores — useful for sanity-checking the rules.
score_map = scored.groupby("Segment")[["R_Score", "F_Score", "M_Score"]].mean().round(2)
fig_heat = px.imshow(
    score_map.T,
    text_auto=True,
    color_continuous_scale="RdYlGn",
    aspect="auto",
    title="Average R / F / M score by segment (1 = poor, 5 = excellent)",
    labels=dict(x="Segment", y="Dimension", color="Mean score"),
)
fig_heat.update_layout(height=380)
show(fig_heat)

# 8d. How many customers sit in each cell of the R x F grid, and what they are
# worth. This is the view that shows where the money actually is.
grid = scored.pivot_table(index="F_Score", columns="R_Score", values="Monetary", aggfunc="mean")
fig_grid = px.imshow(
    grid.round(0),
    text_auto=True,
    origin="lower",
    color_continuous_scale="Blues",
    title="Average spending per R x F cell (the M dimension, revealed)",
    labels=dict(x="Recency score", y="Frequency score", color="Mean EUR"),
)
fig_grid.update_layout(height=450)
show(fig_grid)


# %% 9. WHAT RFM CANNOT SEE
# ==============================================================================
# RFM ranked every customer using three numbers derived from one thing: the till
# receipt. That is its strength — every business has receipts — and its ceiling.

print("\n" + "-" * 78)
print("9. WHAT RFM DID NOT USE")
print("-" * 78)

# It cannot say *when* — only "long ago" or "recently", never "expect them back
# in March". And it has no notion of a customer's future at all.
one_timers = (scored["Frequency"] == 1).mean()
print(f"Everything above came from {len(transactions.columns)} columns of transaction data.")
print("What the log does not contain: who these people are. No age, no income, no")
print("household, no channel preference, no reason for any of it.")
print(f"\nAnd RFM has no notion of the future. {one_timers:.0%} of these customers ordered")
print("exactly once — RFM can tell you they are inactive, but not whether they are worth")
print("chasing, nor how much to spend doing it.")

# One concrete limitation, quantified: recency scores treat every quintile as
# equal-sized, so "at risk" is defined by rank, not by the business.
lapsed_days = scored.loc[scored["R_Score"] == 1, "Recency"].min()
print(f"\nNote too that the scores are *relative*: an R_Score of 1 here means 'has not "
      f"bought\nin {lapsed_days:.0f}+ days' only because that is where the bottom fifth "
      "happens to fall.\nRe-run this next quarter and the same customer can change score "
      "without changing\nbehaviour at all.")

print(
    "\n  -> cltv.py       how much is each customer worth going forward?\n"
    "  -> clustering.py  who are they, using everything we know about them"
)

print("\n" + "=" * 78)
print(f"Done. RFM segmented {len(scored):,} customers from {len(transactions):,} "
      f"transactions  (data: {source})")
print("=" * 78)
