"""
==============================================================================
CUSTOMER LIFETIME VALUE (CLTV) — WHAT IS A CUSTOMER WORTH?
==============================================================================
Course: Big Data and Machine Learning
Topic: Value-based customer segmentation with Python

Business question
-----------------
"Which customers should we invest in, and how much may we spend to win one?"

RFM (``rfm.py``) ranks customers by what they have already done. CLTV puts a
**number in euros** on what a customer is worth over the whole relationship —
which is what you need before you can decide a discount, a service level, or an
acquisition budget.

The formulas
------------
Basis form — the discounted stream of contributions::

                 T
        LTV_i = SUM  GC_it * (1 / (1 + d))^t
                t=1

With a constant contribution and a constant retention rate, that infinite sum
collapses into the closed form this lesson centres on::

                          Rr
        LTV_i = GC_i * ---------  -  AC_i
                       1 - Rr + d

        GC_i  gross contribution per year (revenue minus attributable cost)
        Rr    retention rate — the share of customers who stay another year
        d     discount rate — future euros are worth less than today's
        AC_i  acquisition cost

Where Rr comes from
-------------------
``Rr`` is the term the whole formula turns on, and it is the one people most
often invent. It cannot be read off a customer table — it needs **cohorts**: take
everyone who first bought in a given month, and track how many of them come back
in each later month. Section 5 builds that retention matrix from the transaction
log and measures ``Rr`` instead of assuming it.

Having measured it, section 6 then shows how much the answer would have moved had
we guessed differently — which is the point of measuring.

Historical vs predictive
------------------------
*Historical* CLTV looks backwards: what has this customer already contributed?
*Predictive* CLTV looks forwards: what will they contribute from here? Section 7
computes both for every customer and shows where the two disagree.

Learning objectives
-------------------
1. Derive a per-customer gross contribution from a transaction log
2. Compute historical CLTV as a discounted sum over observed tenure
3. Build a cohort retention matrix and measure the annual retention rate
4. Apply the closed-form predictive CLTV and see it as an infinite-horizon limit
5. Quantify how sensitive the result is to the retention assumption
6. Read customer equity and the acquisition-cost ceiling off the CLTV
7. Combine CLTV with RFM to decide who to invest in

Run it
------
    uv sync
    uv run python fetch_data.py     # once, to download the dataset
    uv run python cltv.py

    SHOW_FIGURES=0 uv run python cltv.py     # numbers only, no browser tabs

Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
==============================================================================
"""

# %% 1. SETUP
# ==============================================================================

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.retail_data import (  # noqa: E402
    annual_retention,
    cohort_retention,
    customer_table,
    prepare,
    retention_by_recency_band,
)
from src.cltv_core import (  # noqa: E402
    ACQUISITION_COST,
    DISCOUNT_RATE,
    GROSS_MARGIN,
    MIN_TENURE_DAYS,
    annual_contribution,
    clv_multiplier,
    clv_segments,
)
from src.rfm_scoring import segment_rfm  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "1") != "0"


def show(fig) -> None:
    """Display a figure unless SHOW_FIGURES=0."""
    if SHOW_FIGURES:
        fig.show()


def show_source(fn) -> None:
    """Print the real source of a function, so the lesson can't drift from it.

    The core CLTV formulas live in ``src/cltv_core.py`` — importing this script
    would re-run the whole lesson, so anything that wants the formulas (the
    teaching app, a notebook) needs them somewhere importable. Printing them here
    means the code you read is the code that just ran.
    """
    print(f"\n--- {fn.__module__}.{fn.__name__} " + "-" * (48 - len(fn.__name__)))
    print(inspect.getsource(fn))


# ------------------------------------------------------------------------------
# The business assumptions, named once and used everywhere.
#
# These are *inputs to the model*, not facts about the data. Every euro this
# lesson reports depends on them, so they are named constants you can see and
# argue about — never buried inside a formula: GROSS_MARGIN = 30%,
# DISCOUNT_RATE = 10%, ACQUISITION_COST = EUR 150, MIN_TENURE_DAYS = 365.
#
# Note what is NOT in this list: the retention rate. It used to be here as a
# guess; section 5 measures it from the data instead.
# ------------------------------------------------------------------------------
# They live in src/cltv_core.py so the teaching app imports the same numbers.


# %% 2. LOAD THE TRANSACTION LOG
# ==============================================================================

print("=" * 78)
print("CUSTOMER LIFETIME VALUE (CLTV)")
print("=" * 78)

transactions, source = prepare()

# One row per customer: first/last purchase, order count, revenue, tenure.
df = customer_table(transactions)
print(f"Aggregated to {len(df):,} customers")


# %% 3. GROSS CONTRIBUTION PER YEAR (GC)
# ==============================================================================


show_source(annual_contribution)

df["GC"] = annual_contribution(df)

print("\n" + "-" * 78)
print("3. GROSS CONTRIBUTION PER YEAR (GC)")
print("-" * 78)
print(f"Assumed gross margin: {GROSS_MARGIN:.0%}")
print(f"Tenure floor: {MIN_TENURE_DAYS} days")
print(df["GC"].describe().round(1).to_string())
print(f"\nTotal annual contribution of the base: EUR {df['GC'].sum():,.0f}")

single_order = df["Orders"] == 1
print(f"\n{single_order.mean():.0%} of customers ordered exactly once, so their observed "
      f"tenure is 0 days.")
print(f"Without a floor their GC would be infinite. With the {MIN_TENURE_DAYS}-day floor they "
      f"average EUR {df.loc[single_order, 'GC'].mean():,.0f}/year,")
print(f"against EUR {df.loc[~single_order, 'GC'].mean():,.0f} for customers who came back — "
      "which is the right way round.")

# Show what a shorter floor would have done, because it is the kind of choice
# that looks harmless and quietly reorders the entire customer base.
short_floor_gc = df["Revenue"] / (np.maximum(df["TenureDays"], 30) / 365.25) * GROSS_MARGIN
print(f"\nHad the floor been 30 days instead, those same one-time buyers would average "
      f"EUR {short_floor_gc[single_order].mean():,.0f}/year")
print(f"— above the EUR {short_floor_gc[~single_order].mean():,.0f} of customers who "
      "actually came back. A single visit would outrank a")
print("two-year relationship, and nothing in the output would look obviously wrong.")


# %% 4. HISTORICAL CLTV — WHAT THEY HAVE ALREADY BEEN WORTH
# ==============================================================================


def historical_clv(
    customers: pd.DataFrame, discount_rate: float = DISCOUNT_RATE
) -> pd.Series:
    """The discounted stream of contributions each customer actually delivered.

    This is the slides' basis form, applied backwards over the history we can
    observe — each customer over *their own* tenure, which is what the real
    transaction dates now let us do::

                     T
            LTV_i = SUM  GC_it * (1 / (1 + d))^t
                    t=1

    Purely retrospective: no retention rate, no assumption about the future.
    Only the discount rate enters, and over a two-year file it barely moves the
    numbers — there is no year 10 here to discount. Discounting earns its keep
    over long horizons, which is exactly where the *predictive* model lives.
    """
    observed_years = np.maximum(customers["TenureDays"], MIN_TENURE_DAYS) / 365.25
    value = pd.Series(0.0, index=customers.index)
    for year in range(1, int(np.ceil(observed_years.max())) + 1):
        # The fraction of this year each customer was actually active (0 to 1).
        active_fraction = (observed_years - (year - 1)).clip(lower=0, upper=1)
        value += customers["GC"] * active_fraction / (1 + discount_rate) ** year
    return value


df["CLV_Historical"] = historical_clv(df)

print("\n" + "-" * 78)
print("4. HISTORICAL CLTV (backward-looking)")
print("-" * 78)
print(f"Discount rate d = {DISCOUNT_RATE:.0%}, each customer over their own observed tenure")
print(df["CLV_Historical"].describe().round(1).to_string())
print(f"\nSanity check — the sum of historical CLTV is EUR {df['CLV_Historical'].sum():,.0f},")
print(f"against EUR {df['Revenue'].sum() * GROSS_MARGIN:,.0f} of actual gross contribution "
      "banked over the period.")
print("Those should be in the same ballpark, and the gap is the discounting plus the")
print("tenure floor. A historical figure that drifts far from money actually received is")
print("a sign the annualisation is wrong.")


# %% 5. MEASURING THE RETENTION RATE FROM COHORTS
# ==============================================================================
# Rr is the number the whole formula turns on, and the one most often invented.
# It cannot be read off a customer table — it needs repeat observation of the
# same people over time. That is what a transaction log gives us.

print("\n" + "-" * 78)
print("5. MEASURING Rr — COHORT RETENTION")
print("-" * 78)

retention_matrix = cohort_retention(transactions)

print("Every customer belongs to the month of their first purchase — their cohort.")
print("For each later month, what share of that cohort bought again?\n")
print("Rows = cohort, columns = months since first purchase, values = % still buying:")
print((retention_matrix.iloc[:8, :10] * 100).round(1).to_string())

print("\nTwo things to read off that table:")
print("  * Column 0 is always 100% — by construction, not as a finding.")
print("  * The bottom-left is empty. A cohort acquired three months before the data")
print("    ends cannot have a twelve-month figure yet. Those are missing observations,")
print("    NOT zeros — treating them as zeros invents a collapse that never happened.")

average_curve = retention_matrix.mean(axis=0)
print("\nAveraged across cohorts, monthly retention runs:")
print("  " + "  ".join(f"m{m}:{v:.0%}" for m, v in average_curve.head(7).items()))
print(f"\nThat looks alarming, and it is not. Monthly retention of ~{average_curve.iloc[1]:.0%} does "
      "not mean 80%\nof customers left — it means most people do not buy gifts every single "
      "month. The\ncurve flattens rather than falling to zero, which is the signature of a "
      "real customer\nbase: an early drop-off, then a loyal core that keeps coming back.")

# The formula works in years, so measure Rr in years.
annual = annual_retention(transactions)
MEASURED_RETENTION = annual["retention_rate"]

print("\n" + "-" * 40)
print("The CLTV formula works in YEARS, so Rr must be an annual number:")
print(f"  customers acquired in year 1 ({annual['year_one'][0]:%Y-%m-%d} to "
      f"{annual['year_one'][1]:%Y-%m-%d}): {annual['cohort_size']:,}")
print(f"  of those, bought again in year 2:                          {annual['returned']:,}")
print(f"\n  Rr = {annual['returned']:,} / {annual['cohort_size']:,} = "
      f"{MEASURED_RETENTION:.1%}")
print("\nMeasured, not assumed. This is the number the previous version of this lesson")
print("had to guess at, because a snapshot of customer attributes cannot produce it.")

fig_cohort = px.imshow(
    (retention_matrix.iloc[:, :13] * 100).round(1),
    text_auto=".0f",
    color_continuous_scale="Blues",
    aspect="auto",
    title="Cohort retention: % of each cohort still buying, by month since first purchase",
    labels=dict(x="Months since first purchase", y="Cohort", color="% retained"),
)
fig_cohort.update_layout(height=620)
show(fig_cohort)

fig_curve = go.Figure()
for cohort in retention_matrix.index[:6]:
    row = retention_matrix.loc[cohort].dropna()
    fig_curve.add_trace(
        go.Scatter(x=row.index[1:], y=row.values[1:] * 100, mode="lines",
                   name=str(cohort), opacity=0.55)
    )
fig_curve.add_trace(
    go.Scatter(x=average_curve.index[1:], y=average_curve.values[1:] * 100, mode="lines+markers",
               name="average", line=dict(color="black", width=3))
)
fig_curve.update_layout(
    title="Retention curves — a steep early drop, then a loyal core that flattens out",
    xaxis_title="Months since first purchase",
    yaxis_title="% of cohort still buying",
    height=460,
)
show(fig_curve)


# %% 6. PREDICTIVE CLTV — THE CLOSED FORM
# ==============================================================================


show_source(clv_multiplier)


def predictive_clv(
    customers: pd.DataFrame,
    retention_rate: float | pd.Series = MEASURED_RETENTION,
    discount_rate: float = DISCOUNT_RATE,
    acquisition_cost: float = ACQUISITION_COST,
) -> pd.Series:
    """The lecture's closed-form CLTV::

            LTV_i = GC_i * (Rr / (1 - Rr + d)) - AC_i

    Forward-looking, and an infinite-horizon model: it assumes the customer keeps
    contributing GC every year, surviving each year with probability Rr, forever.
    Section 5b shows how quickly that infinite sum converges — which is why the
    simplification is defensible.

    ``retention_rate`` takes a single number (the same Rr for everybody, as on
    the slide) or a Series (one Rr per customer, section 5c).
    """
    return customers["GC"] * clv_multiplier(retention_rate, discount_rate) - acquisition_cost


def horizon_clv(
    customers: pd.DataFrame,
    years: int,
    retention_rate: float = MEASURED_RETENTION,
    discount_rate: float = DISCOUNT_RATE,
    acquisition_cost: float = ACQUISITION_COST,
) -> pd.Series:
    """The same model truncated at ``years``::

                     T
            LTV_i = SUM  GC_i * (Rr / (1 + d))^t  -  AC_i
                    t=1

    Useful in its own right — many companies refuse to plan past five years — and
    useful here as a check that the closed form is really the limit of this sum.
    """
    ratio = retention_rate / (1 + discount_rate)
    multiplier = sum(ratio**t for t in range(1, years + 1))
    return customers["GC"] * multiplier - acquisition_cost


df["CLV_Predictive"] = predictive_clv(df)

print("\n" + "-" * 78)
print("6. PREDICTIVE CLTV (forward-looking, closed form)")
print("-" * 78)
print(f"Rr = {MEASURED_RETENTION:.1%} (measured)   d = {DISCOUNT_RATE:.0%} (assumed)   "
      f"AC = EUR {ACQUISITION_COST:,.0f} (assumed)")
print(f"Multiplier Rr / (1 - Rr + d) = {clv_multiplier(MEASURED_RETENTION):.2f}  "
      f"-> each customer is worth {clv_multiplier(MEASURED_RETENTION):.2f} years of contribution")
print(df["CLV_Predictive"].describe().round(1).to_string())

# 6b. The closed form is the limit of the finite-horizon sum.
print("\nThe closed form is the infinite-horizon limit — watch it converge:")
print(f"  {'horizon':>10}  {'multiplier':>11}  {'mean CLTV':>12}")
for years in [1, 2, 3, 5, 10, 20, 50]:
    ratio = MEASURED_RETENTION / (1 + DISCOUNT_RATE)
    multiplier = sum(ratio**t for t in range(1, years + 1))
    mean_value = horizon_clv(df, years).mean()
    print(f"  {years:>8} y  {multiplier:>11.3f}  {mean_value:>12,.0f}")
print(f"  {'infinite':>10}  {clv_multiplier(MEASURED_RETENTION):>11.3f}  {df['CLV_Predictive'].mean():>12,.0f}")
print("The tail terms stop mattering quickly, so the closed form and a medium-term plan")
print("give the same answer — the infinite horizon is a convenience, not a claim that")
print("customers live forever.")


# %% 7. ONE RETENTION RATE FOR EVERYBODY?
# ==============================================================================
# The formula above gives every customer the same Rr. That is certainly wrong:
# somebody who bought last week is not as likely to leave as somebody silent for
# six months. The slides define "predictive CLV" as the variant that uses
# behavioural data — and with a transaction log we can *measure* the behaviour
# rather than asserting a plausible-looking table of numbers.

band_retention = retention_by_recency_band(transactions)

print("\n" + "-" * 78)
print("7. GIVING EACH CUSTOMER THEIR OWN RETENTION RATE")
print("-" * 78)
print("Stand at the one-year mark. Sort everyone who has bought so far by how long")
print("they have been quiet. Then look forward a year and count who came back:\n")
print(band_retention.assign(
    RetentionRate=lambda t: (t["RetentionRate"] * 100).round(1).astype(str) + "%"
).to_string())

print("\nMeasured, not invented — and the spread is large: customers quiet for under a month")
print(f"come back at {band_retention['RetentionRate'].iloc[0]:.0%}, those quiet for "
      f"{band_retention.index[-1]} at {band_retention['RetentionRate'].iloc[-1]:.0%}.")
print("Note the direction of the test: the recency half is measured strictly *before* the")
print("return half, so nothing here peeks at the future it is trying to predict.")

# Map each customer's current recency onto the band rate we just measured. The
# edges are read back off the measured table so the two can never drift apart.
band_edges = [0] + [int(label.split("-")[1].rstrip("d")) + 1 for label in band_retention.index]
band_edges[-1] = max(band_edges[-1], int(df["Recency"].max()) + 1)
df["Rr_Behavioural"] = pd.cut(
    df["Recency"], bins=band_edges, labels=band_retention["RetentionRate"].to_numpy(), right=False
).astype(float)
df["CLV_Behavioural"] = predictive_clv(df, retention_rate=df["Rr_Behavioural"])

print("\nApplying those measured rates per customer:")
print(f"  average Rr across the base: {df['Rr_Behavioural'].mean():.1%} "
      f"(the single measured rate was {MEASURED_RETENTION:.1%})")
print(f"  total equity: EUR {df['CLV_Predictive'].sum():,.0f} -> "
      f"EUR {df['CLV_Behavioural'].sum():,.0f}, a "
      f"{df['CLV_Behavioural'].sum() / df['CLV_Predictive'].sum() - 1:+.0%} change")
print(
    "\nThat total moved more than the average rate did, and the reason is worth knowing.\n"
    f"The multiplier is *convex* in Rr: it climbs from {clv_multiplier(0.4):.2f} to "
    f"{clv_multiplier(0.8):.2f} as Rr goes from 40% to 80%,\n"
    "so the loyal customers gain more than the lapsed ones lose. Splitting one average\n"
    "rate into five therefore changes the total. Averaging the input is not the same as\n"
    "averaging the output — a trap that reaches far beyond CLTV."
)
print("\nIndividual customers move a great deal more than the total:")
biggest_move = (df["CLV_Behavioural"] - df["CLV_Predictive"]).abs().nlargest(3).index
print(df.loc[biggest_move, ["Customer ID", "Recency", "GC", "CLV_Predictive",
                            "CLV_Behavioural"]].round(0).to_string(index=False))


# %% 8. WHAT IF WE HAD GUESSED INSTEAD?
# ==============================================================================
# This is the part to remember. The data contributes GC; the business
# assumptions contribute the multiplier. Change Rr from 0.60 to 0.90 and every
# customer's value quadruples without a single number in the data changing.


def sensitivity_table(
    retention_rates: list[float], discount_rates: list[float]
) -> pd.DataFrame:
    """The multiplier ``Rr / (1 - Rr + d)`` across a grid of both assumptions."""
    return pd.DataFrame(
        {f"d={d:.0%}": [clv_multiplier(rr, d) for rr in retention_rates] for d in discount_rates},
        index=[f"Rr={rr:.0%}" for rr in retention_rates],
    ).round(2)


retention_grid = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
discount_grid = [0.05, 0.10, 0.15, 0.20]
sensitivity = sensitivity_table(retention_grid, discount_grid)

print("\n" + "-" * 78)
print("8. WHAT IF WE HAD GUESSED INSTEAD?")
print("-" * 78)
print("Lifetime multiplier Rr / (1 - Rr + d):")
print(sensitivity.to_string())

low, high = clv_multiplier(0.60, DISCOUNT_RATE), clv_multiplier(0.90, DISCOUNT_RATE)
print(f"\nAt d = {DISCOUNT_RATE:.0%}, moving retention from 60% to 90% takes the multiplier "
      f"from {low:.2f} to {high:.2f}")
print(f"— a {high / low:.1f}x change in every customer's value, and in the acquisition budget")
print("that follows from it. Moving the discount rate across its whole plausible range does")
print("far less. Retention is the lever.")

# The measured value against the round numbers people reach for.
print(f"\nWe measured Rr = {MEASURED_RETENTION:.1%}. Had we reached for a plausible round "
      "number instead:")
measured_equity = df["CLV_Predictive"].sum()
for guess in [0.50, 0.60, 0.70, 0.80]:
    guessed_equity = (df["GC"] * clv_multiplier(guess) - ACQUISITION_COST).sum()
    print(f"  Rr = {guess:.0%}  ->  customer equity EUR {guessed_equity:>12,.0f}   "
          f"({guessed_equity / measured_equity - 1:+.0%} vs measured)")
print(f"  Rr = {MEASURED_RETENTION:.1%}  ->  customer equity EUR {measured_equity:>12,.0f}   "
      "(measured)")
print("\nThat spread is the value of an afternoon spent building the cohort table. None of")
print("those guesses is unreasonable; they are just wrong, and each one would have set a")
print("different acquisition budget for the entire business.")
print("\nThe sensitivity table still matters after measuring — Rr was measured on two years")
print("of one retailer's history, and next year is not obliged to repeat it. Measuring tells")
print("you where you are; the table tells you how wrong you can afford to be.")

fig_sens = px.imshow(
    sensitivity,
    text_auto=True,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="CLTV multiplier: retention rate vs discount rate",
    labels=dict(x="Discount rate", y="Retention rate", color="Years of GC"),
)
fig_sens.update_layout(height=420)
show(fig_sens)


# %% 9. HISTORICAL vs PREDICTIVE
# ==============================================================================

COMPARISON = pd.DataFrame(
    {
        "Historical CLV": [
            "Understand what a customer has already contributed",
            "Past transactions only",
            "Retrospective",
            "Grounded in fact; no assumptions to argue about",
            "Blind to the future; says nothing about a new customer",
            "Reporting, and businesses with little data",
        ],
        "Predictive CLV": [
            "Forecast what a customer will contribute next",
            "Transactions + retention and discount assumptions",
            "Forward-looking",
            "Supports budgets, targeting and acquisition ceilings",
            "Only as good as its assumptions; needs regular re-fitting",
            "Planning long-term retention investment",
        ],
    },
    index=["Objective", "Data used", "Approach", "Advantages", "Limitations", "Best suited for"],
)

print("\n" + "-" * 78)
print("9. HISTORICAL vs PREDICTIVE CLTV")
print("-" * 78)
print(COMPARISON.to_string())

correlations = pd.DataFrame(
    {
        "Pearson": [
            df["CLV_Historical"].corr(df["CLV_Predictive"]),
            df["CLV_Historical"].corr(df["CLV_Behavioural"]),
        ],
        "Spearman (rank)": [
            df["CLV_Historical"].corr(df["CLV_Predictive"], method="spearman"),
            df["CLV_Historical"].corr(df["CLV_Behavioural"], method="spearman"),
        ],
    },
    index=["vs flat predictive", "vs behavioural predictive"],
).round(3)

print("\nHow much does the forward-looking view actually disagree with the backward one?")
print(correlations.to_string())

print(
    "\nRead the two columns against each other, because they tell different stories.\n"
    "\nThe **flat** predictive model barely re-ranks anybody (rank correlation "
    f"{correlations.loc['vs flat predictive', 'Spearman (rank)']:.2f}).\n"
    "That is arithmetic, not a finding: one GC per customer times one constant multiplier\n"
    "is a rescaling of the historical figure. It changes the euros, not the order.\n"
    "\nThe **behavioural** model does re-rank — and notice that Pearson barely moved\n"
    f"({correlations.loc['vs flat predictive', 'Pearson']:.2f} to "
    f"{correlations.loc['vs behavioural predictive', 'Pearson']:.2f}) while the rank "
    f"correlation fell to "
    f"{correlations.loc['vs behavioural predictive', 'Spearman (rank)']:.2f}."
)

shift = (df["CLV_Historical"].rank(ascending=False)
         - df["CLV_Behavioural"].rank(ascending=False))
print(f"\nPearson is being held up by a handful of very large customers, whose position "
      f"never\nchanges. Underneath it, {(shift.abs() > 500).mean():.0%} of customers move "
      f"more than 500 places and the median\ncustomer moves {shift.abs().median():.0f}. "
      "On money — which is always this skewed — a high Pearson\ncorrelation can hide a "
      "complete reshuffle of everyone who is not a whale. Rank\ncorrelation is the honest "
      "check when the question is 'who should we target?'.")

print("\nThe customers the two views disagree about most:")
df["Rank_Historical"] = df["CLV_Historical"].rank(ascending=False)
df["Rank_Behavioural"] = df["CLV_Behavioural"].rank(ascending=False)
df["Rank_Shift"] = df["Rank_Historical"] - df["Rank_Behavioural"]
movers = pd.concat([df.nlargest(3, "Rank_Shift"), df.nsmallest(3, "Rank_Shift")])
print(
    movers[["Customer ID", "Recency", "Orders", "Revenue", "Rr_Behavioural", "Rank_Historical",
            "Rank_Behavioural", "Rank_Shift"]]
    .to_string(index=False, formatters={
        "Rr_Behavioural": "{:.0%}".format,
        "Rank_Historical": "{:.0f}".format,
        "Rank_Behavioural": "{:.0f}".format,
        "Rank_Shift": "{:+.0f}".format,
    })
)
print("Positive shift = the forward-looking view promotes them (they are still active);")
print("negative = it demotes them (they spent well, then went quiet).")

fig_compare = px.scatter(
    df,
    x="CLV_Historical",
    y="CLV_Behavioural",
    color="Recency",
    color_continuous_scale="Viridis_r",
    opacity=0.6,
    title="Historical vs predictive CLTV — the fan opens up because retention varies",
    labels={
        "CLV_Historical": "Historical CLTV (EUR already contributed, discounted)",
        "CLV_Behavioural": "Predictive CLTV (EUR expected from here)",
        "Recency": "Recency (d)",
    },
)
fig_compare.update_layout(height=560)
show(fig_compare)


# %% 10. CUSTOMER EQUITY AND THE ACQUISITION CEILING
# ==============================================================================


def customer_equity(clv: pd.Series) -> float:
    """The sum of all customer lifetime values.

    The slides' "Customer Equity": the value of the entire customer base, and one
    of the inputs to valuing the company itself. It is the number that turns a
    marketing metric into a finance one.
    """
    return float(clv.sum())


def max_acquisition_cost(
    customers: pd.DataFrame,
    retention_rate: float = MEASURED_RETENTION,
    discount_rate: float = DISCOUNT_RATE,
) -> pd.Series:
    """CLTV *before* subtracting acquisition cost — the ceiling on what to pay.

    Spending more than this to win a customer destroys value by construction.
    It is the most directly actionable number CLTV produces.
    """
    return customers["GC"] * clv_multiplier(retention_rate, discount_rate)


equity = customer_equity(df["CLV_Predictive"])
ceiling = max_acquisition_cost(df)

print("\n" + "-" * 78)
print("10. CUSTOMER EQUITY AND THE ACQUISITION CEILING")
print("-" * 78)
print(f"Customer equity (sum of predictive CLTV over {len(df):,} customers): EUR {equity:,.0f}")
print(f"Average CLTV per customer:                                       EUR "
      f"{df['CLV_Predictive'].mean():,.0f}")
print("\nMost you should ever pay to acquire a customer like...")
print(f"  the median one:   EUR {ceiling.median():,.0f}")
print(f"  a top-decile one: EUR {ceiling.quantile(0.9):,.0f}")
print(f"  a bottom-decile one: EUR {ceiling.quantile(0.1):,.0f}")
print(f"\nAt the assumed AC of EUR {ACQUISITION_COST:,.0f}, "
      f"{(ceiling < ACQUISITION_COST).mean():.1%} of customers cost more to acquire")
print("than they will ever return — the segment to stop buying.")


# %% 11. CLTV SEGMENTS, AND WHERE THEY MEET RFM
# ==============================================================================


show_source(clv_segments)

df["CLV_Segment"] = clv_segments(df["CLV_Predictive"])

print("\n" + "-" * 78)
print("11. CLTV SEGMENTS")
print("-" * 78)
print("Tiers are cut on the flat closed-form CLTV — the lecture formula, one Rr for")
print("everybody. Using the recency-adjusted version instead would make the cross-tab")
print("against RFM partly circular, since recency drives both.")
segment_table = (
    df.groupby("CLV_Segment", observed=True)
    .agg(
        Customers=("Customer ID", "count"),
        Mean_CLTV=("CLV_Predictive", "mean"),
        Mean_Revenue=("Revenue", "mean"),
        Mean_Orders=("Orders", "mean"),
        Mean_Recency=("Recency", "mean"),
        Mean_TenureDays=("TenureDays", "mean"),
    )
    .round(1)
)
segment_table["Share_of_Equity"] = (
    df.groupby("CLV_Segment", observed=True)["CLV_Predictive"].sum() / equity * 100
).round(1)
print(segment_table.to_string())

premium_share = segment_table.loc["Premium", "Share_of_Equity"]
print(f"\nThe Premium quarter carries {premium_share:.0f}% of total customer equity.")

# Slide 59: RFM tells you who is valuable *now*, CLTV tells you how much to
# invest going forward. Cross them and you get an action per cell.
rfm_segments = segment_rfm(transactions)
df["RFM_Segment"] = df["Customer ID"].map(
    rfm_segments.set_index("Id")["Segment"]
).to_numpy()

print("\nRFM segment vs CLTV tier — the two views combined:")
combined = pd.crosstab(df["RFM_Segment"], df["CLV_Segment"])
print(combined.to_string())
print(
    "\nRead it as a decision table. 'Can't Lose Them' customers in the Premium tier are\n"
    "the single most urgent group in the business: high value, and walking out.\n"
    "'Lost' customers in the Low tier are the cheapest thing to stop spending on."
)

at_risk_premium = int(combined.loc["Can't Lose Them", "Premium"]) if "Can't Lose Them" in combined.index else 0
print(f"\nHere that urgent group is {at_risk_premium} customers, worth EUR "
      f"{df.loc[(df['RFM_Segment'] == "Can't Lose Them") & (df['CLV_Segment'] == 'Premium'), 'CLV_Predictive'].sum():,.0f} "
      "in expected lifetime value.")

fig_segments = go.Figure(
    go.Bar(
        x=segment_table.index.astype(str),
        y=segment_table["Mean_CLTV"],
        marker_color=["#B0BEC5", "#90A4AE", "#4C78A8", "#2E5B8A"],
        text=segment_table["Mean_CLTV"].round(0),
        texttemplate="EUR %{text:,.0f}",
    )
)
fig_segments.update_layout(
    title="Average predictive CLTV by value tier",
    yaxis_title="CLTV (EUR)",
    height=420,
)
show(fig_segments)

fig_overlap = px.imshow(
    combined,
    text_auto=True,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="RFM segment vs CLTV tier (customers per cell)",
    labels=dict(x="CLTV tier", y="RFM segment", color="Customers"),
)
fig_overlap.update_layout(height=520)
show(fig_overlap)


# %% 12. WHERE CLTV STOPS
# ==============================================================================

print("\n" + "-" * 78)
print("12. WHAT CLTV STILL CANNOT TELL YOU")
print("-" * 78)
print(
    "CLTV put a euro value on every customer and told us how much to invest in each —\n"
    "and, unlike the RFM lesson, it did it with a retention rate measured rather than\n"
    "guessed. Section 7 stretched that as far as a till receipt goes: a rate that varies\n"
    "with recency. But recency is still just another purchase outcome.\n"
    "\nWhat it cannot do is tell you *who these people are*. This transaction log holds\n"
    "an invoice number, a product code, a price and a country — and nothing else. There\n"
    "is no age here, no income, no household, no channel preference. 'Premium tier' is\n"
    "the most specific description this data can produce, and you cannot brief a\n"
    "campaign on it.\n"
    "\nThat is a limitation of the *data*, not of CLTV, and it is why the next lesson\n"
    "changes dataset: clustering needs attributes a receipt never records.\n"
    "\n  -> clustering.py   who are they, using everything we know about them"
)

print("\n" + "=" * 78)
print(f"Done. Customer equity EUR {equity:,.0f} across {len(df):,} customers  (data: {source})")
print("=" * 78)
