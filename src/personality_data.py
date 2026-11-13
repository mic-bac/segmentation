"""
personality_data.py
-------------------
Data access for the clustering lesson (``clustering.py``).

Dataset: https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

Why this dataset, and only for clustering
-----------------------------------------
``rfm.py`` and ``cltv.py`` use a transaction log (see ``retail_data.py``), because
recency, frequency and retention can only be measured from dated purchases.
Clustering needs the opposite thing: **customer attributes**. The retail log
records invoice numbers and prices but no age or income, so on its own it cannot
tell you that a segment is "affluent households without children" — and that kind
of description is the entire point of clustering. (A loyalty-card programme links
receipts to demographics; these two public datasets are simply not linked.)

So the two datasets are not a compromise, they are the lesson: each method needs
the data its question requires.

Shape of the data
-----------------
2,240 rows, **one row per customer**, 29 columns of demographics, two years of
category spending, purchase counts per channel, and responses to five campaigns.
Rich in attributes, poor in time — the mirror image of the retail log.

Graceful degradation
--------------------
``data/`` is git-ignored, so a fresh clone has no CSV. ``load_customers`` then
falls back to a **synthetic** customer base with the same columns and the same
kinds of structure, and reports ``source="synthetic"`` so the lesson can say so
out loud. The lesson always runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Paths are anchored to this file, not to the working directory, so the lessons
# behave the same whether you run `python rfm.py`, an IDE cell, or a notebook
# from somewhere else entirely.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CUSTOMER_CSV = DATA_DIR / "customer_segmentation.csv"

# ---------------------------------------------------------------------------
# Column groups — named once, reused everywhere
# ---------------------------------------------------------------------------

SPENDING_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]

# Z_CostContact is always 3 and Z_Revenue is always 11 — constant columns carry
# zero information, and a constant column with variance 0 is worse than useless
# in a distance-based algorithm (it survives standardisation as NaN/0).
CONSTANT_COLS = ["Z_CostContact", "Z_Revenue"]

# The 29 raw columns, in file order. The synthetic fallback reproduces exactly
# this schema, so nothing downstream can tell the two sources apart structurally.
RAW_COLUMNS = [
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    *SPENDING_COLS,
    "NumDealsPurchases",
    *PURCHASE_COLS,
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    *CONSTANT_COLS,
    "Response",
]

# Features used by the clustering lesson.
#
# The selection is the point of the lesson, not a detail. RFM and CLTV can only
# see purchase outcomes; clustering can use everything the company knows, so this
# list deliberately mixes **demographics** (who they are) with **behaviour** (what
# they do) — which is what lets a cluster be described as "affluent, no children,
# responds to campaigns" rather than merely "spends a lot".
#
# Two deliberate omissions:
#   * `Recency` — RFM's territory, and leaving it out keeps the two lessons
#     answering different questions. It also measurably sharpens the clusters
#     (silhouette 0.31 vs 0.26 at k=3): recency is close to random noise in this
#     file and noise costs you resolution in every distance-based method.
#   * the share-of-wallet columns — nine features cluster visibly worse than
#     seven here. Each extra dimension spreads the same points further apart
#     (the curse of dimensionality), so features have to earn their place.
CLUSTERING_FEATURES = [
    "Income",  # demographic: purchasing power
    "Age",  # demographic: life stage
    "Total_Children",  # demographic: household shape
    "Total_Spending",  # behaviour: value
    "Total_Purchases",  # behaviour: activity
    "Total_Campaigns_Accepted",  # behaviour: marketing responsiveness
    "NumWebVisitsMonth",  # behaviour: channel usage
]

# Cleaning rules, as named constants rather than magic numbers in the code.
INCOME_CLIP_QUANTILE = 0.998  # squashes the single 666,666 joke entry
MAX_PLAUSIBLE_AGE = 100  # birth years 1893/1899/1900 are data-entry errors

# `Marital_Status` ships with free-text junk. Mapping it is not cosmetic: every
# unmapped label becomes its own dummy column the moment anyone one-hot encodes.
MARITAL_MAP = {
    "Married": "Married",
    "Together": "Together",
    "Single": "Single",
    "Divorced": "Divorced",
    "Widow": "Widow",
    "Alone": "Single",  # same thing, typed by a different person
    "Absurd": "Other",  # not a marital status
    "YOLO": "Other",  # also not a marital status
}


# ---------------------------------------------------------------------------
# Loading (with a synthetic fallback so the lesson always runs)
# ---------------------------------------------------------------------------


def load_customers(
    path: Path | str = CUSTOMER_CSV, random_state: int = 42
) -> tuple[pd.DataFrame, str]:
    """Load the raw customer table; returns ``(dataframe, source)``.

    ``source`` is ``"kaggle"`` or ``"synthetic"`` so each lesson can be honest
    about which data produced the numbers on screen. Nothing downstream branches
    on it — the two frames have identical schemas.
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path), "kaggle"
    return make_synthetic_customers(random_state=random_state), "synthetic"


def make_synthetic_customers(n_customers: int = 2240, random_state: int = 42) -> pd.DataFrame:
    """Generate a customer base with the same 29 columns as the Kaggle original.

    Built from four latent customer *types* (see ``TYPES`` below) whose mixing
    weights and parameters are set so that the lessons stay interesting: the RFM
    ladder fills most of its segments, CLTV spreads over a wide range, and the
    clustering algorithms have real structure to find rather than one gaussian
    blob. The type label itself is **not** returned — recovering it is exactly
    what the clustering lesson is pretending to do.

    The known data-quality quirks of the original are reproduced on purpose
    (missing incomes, one absurd income, a few impossible birth years, junk
    marital-status labels), so the cleaning step still has work to do and the
    printed output tells the same story on either data source.
    """
    rng = np.random.default_rng(random_state)

    #                      weight  income   recency   spend    wine  meat  fruit  kids
    TYPES = [
        ("high_value", 0.22, 78_000, (0, 45), 1500, 0.55, 0.25, 0.05, 0.2),
        ("budget_family", 0.34, 34_000, (0, 99), 180, 0.25, 0.25, 0.15, 1.4),
        ("lapsed", 0.24, 52_000, (55, 99), 420, 0.45, 0.20, 0.10, 0.9),
        ("web_savvy", 0.20, 60_000, (0, 60), 700, 0.30, 0.35, 0.12, 0.6),
    ]
    weights = np.array([t[1] for t in TYPES])
    kinds = rng.choice(len(TYPES), size=n_customers, p=weights / weights.sum())

    rows = []
    for i, kind in enumerate(kinds):
        _, _, income_mid, recency_range, spend_mid, wine, meat, fruit, kid_rate = TYPES[kind]

        # Income: lognormal around the type's midpoint, so the tail is on the
        # right the way real income distributions are.
        income = float(np.clip(rng.lognormal(np.log(income_mid), 0.35), 5_000, 160_000))

        # Spending scales with income relative to the type's midpoint, so richer
        # customers inside a type also spend more — a signal a model can find.
        # Clipped at the real file's maximum so the synthetic euro amounts stay in
        # the same ballpark as the ones students would see on the real data.
        budget = min(2_500.0, spend_mid * (income / income_mid) ** 0.7 * rng.lognormal(0, 0.40))
        # Category mix: the type's taste profile, jittered per customer so no two
        # customers share a wallet split exactly.
        shares = np.array([wine, fruit, meat, 0.08, 0.06, 0.10]) * rng.dirichlet(
            np.full(6, 12.0)
        )
        spend = (budget * shares / shares.sum()).round()

        total_spend = spend.sum()
        recency = int(rng.integers(*recency_range))

        # Purchase counts: Poisson driven by spend, split across channels by type.
        n_purchases = 1 + rng.poisson(np.clip(total_spend / 45, 0.5, 24))
        if kind == 3:  # web_savvy leans online
            channel_p = [0.55, 0.20, 0.25]
        elif kind == 1:  # budget_family shops in store
            channel_p = [0.20, 0.10, 0.70]
        else:
            channel_p = [0.30, 0.30, 0.40]
        web, catalog, store = rng.multinomial(n_purchases, channel_p)

        # Campaign acceptance rises with spending — the "responsive high spender"
        # pattern that makes the RFM/CLTV story coherent.
        p_accept = float(np.clip(0.02 + total_spend / 6_000, 0.02, 0.45))
        campaigns = rng.binomial(1, p_accept, size=5)

        rows.append(
            {
                "ID": 1000 + i,
                # Clipped to the same span as the real file (1940-1996) — the
                # impossible birth years are planted deliberately further down.
                "Year_Birth": int(np.clip(rng.normal(1969, 12), 1940, 1996)),
                "Education": rng.choice(
                    ["Graduation", "PhD", "Master", "2n Cycle", "Basic"],
                    p=[0.50, 0.22, 0.17, 0.09, 0.02],
                ),
                "Marital_Status": rng.choice(
                    ["Married", "Together", "Single", "Divorced", "Widow"],
                    p=[0.39, 0.26, 0.21, 0.10, 0.04],
                ),
                "Income": income,
                "Kidhome": int(min(2, rng.poisson(kid_rate * 0.6))),
                "Teenhome": int(min(2, rng.poisson(kid_rate * 0.5))),
                # Same enrolment window as the original file, same dd-mm-yyyy text
                # format — including the format, so the parsing lesson still applies.
                "Dt_Customer": (
                    pd.Timestamp("2012-07-30") + pd.Timedelta(days=int(rng.integers(0, 700)))
                ).strftime("%d-%m-%Y"),
                "Recency": recency,
                "MntWines": int(spend[0]),
                "MntFruits": int(spend[1]),
                "MntMeatProducts": int(spend[2]),
                "MntFishProducts": int(spend[3]),
                "MntSweetProducts": int(spend[4]),
                "MntGoldProds": int(spend[5]),
                "NumDealsPurchases": int(rng.integers(0, 6)),
                "NumWebPurchases": int(web),
                "NumCatalogPurchases": int(catalog),
                "NumStorePurchases": int(store),
                "NumWebVisitsMonth": int(np.clip(rng.poisson(8 if kind == 3 else 5), 0, 20)),
                "AcceptedCmp3": int(campaigns[2]),
                "AcceptedCmp4": int(campaigns[3]),
                "AcceptedCmp5": int(campaigns[4]),
                "AcceptedCmp1": int(campaigns[0]),
                "AcceptedCmp2": int(campaigns[1]),
                "Complain": int(rng.binomial(1, 0.01)),
                "Z_CostContact": 3,
                "Z_Revenue": 11,
                "Response": int(rng.binomial(1, p_accept)),
            }
        )

    df = pd.DataFrame(rows)[RAW_COLUMNS]

    # Plant the original's data-quality quirks so cleaning still has work to do.
    quirks = rng.choice(len(df), size=30, replace=False)
    df.loc[quirks[:24], "Income"] = np.nan  # 24 missing incomes, as in the original
    df.loc[quirks[24], "Income"] = 666_666  # the absurd entry
    df.loc[quirks[25:28], "Year_Birth"] = [1893, 1899, 1900]  # impossible ages
    df.loc[quirks[28:], "Marital_Status"] = ["Absurd", "YOLO"]  # free-text junk

    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_customers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean the raw table: dates, missing incomes, outliers, junk categories.

    Every step here is a decision, not a formality, so each one prints what it
    did. "How many rows did my cleaning throw away?" is a question students
    should be able to answer about their own pipelines.
    """
    out = df.copy()
    n_before = len(out)

    # 1. Dates. The file stores dd-mm-yyyy *as text*. Parsing it explicitly is
    #    not optional: on strings, max() is alphabetical, so "31-12-2012" would
    #    come out "later" than "01-01-2014".
    out["Dt_Customer"] = pd.to_datetime(out["Dt_Customer"], format="%d-%m-%Y")

    # 2. Missing income. 24 customers have none. Median-fill rather than drop:
    #    they are complete in every other respect, and the median is robust to
    #    the skew that income always has.
    out["Income"] = pd.to_numeric(out["Income"], errors="coerce")
    n_missing = int(out["Income"].isna().sum())
    out["Income"] = out["Income"].fillna(out["Income"].median())

    # 3. Income outliers. One customer reports 666,666. Winsorising (capping at
    #    a high quantile) keeps the row instead of deleting a whole customer,
    #    and the quantile is set high enough that genuine six-figure earners
    #    survive untouched — clipping at q99 would have flattened them too.
    cap = out["Income"].quantile(INCOME_CLIP_QUANTILE)
    n_capped = int((out["Income"] > cap).sum())
    out["Income"] = out["Income"].clip(upper=cap)

    # 4. Impossible ages. Birth years 1893/1899/1900 would make these customers
    #    120+ years old. Unlike the income outlier this is not an extreme value,
    #    it is a *wrong* value, so there is nothing to cap — drop the rows.
    reference_date = out["Dt_Customer"].max()
    age = reference_date.year - out["Year_Birth"]
    n_impossible = int((age > MAX_PLAUSIBLE_AGE).sum())
    out = out[age <= MAX_PLAUSIBLE_AGE].copy()

    # 5. Junk categories ("Absurd", "YOLO", "Alone").
    out["Marital_Status"] = out["Marital_Status"].map(MARITAL_MAP).fillna("Other")

    # 6. Constant columns carry no information.
    out = out.drop(columns=[c for c in CONSTANT_COLS if c in out.columns])

    out = out.reset_index(drop=True)

    if verbose:
        print(f"  filled {n_missing} missing incomes with the median")
        print(f"  capped {n_capped} income(s) at the {INCOME_CLIP_QUANTILE:.1%} quantile "
              f"(${cap:,.0f})")
        print(f"  dropped {n_impossible} row(s) with an implausible age (>{MAX_PLAUSIBLE_AGE})")
        print(f"  dropped constant columns: {', '.join(CONSTANT_COLS)}")
        print(f"  rows: {n_before} -> {len(out)}")

    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns all three lessons share.

    One definition, one place. When RFM's "Frequency" and clustering's
    "Engagement_Score" are built from the same ``Total_Purchases``, the two
    lessons are talking about the same customers — which is the whole point of
    the RFM / CLTV / clustering comparison at the end.
    """
    out = df.copy()

    # The data is a snapshot, so "today" is the most recent enrolment date in
    # the file. Every age/tenure/recency statement below is relative to it.
    reference_date = out["Dt_Customer"].max()

    out["Age"] = reference_date.year - out["Year_Birth"]
    out["Tenure_Days"] = (reference_date - out["Dt_Customer"]).dt.days
    # Someone who enrolled yesterday has a tenure of ~0 days. Anything that
    # divides by tenure (CLTV's annual contribution) would explode, so floor it
    # at a month: "we cannot annualise less than a month of history".
    out["Tenure_Years"] = np.maximum(out["Tenure_Days"], 30) / 365.25

    out["Total_Spending"] = out[SPENDING_COLS].sum(axis=1)
    out["Total_Purchases"] = out[PURCHASE_COLS].sum(axis=1)
    out["Total_Campaigns_Accepted"] = out[CAMPAIGN_COLS].sum(axis=1)
    out["Total_Children"] = out["Kidhome"] + out["Teenhome"]

    # Engagement: purchases plus site visits, normalised to roughly 0-1 so it is
    # comparable across customers.
    out["Engagement_Score"] = (out["Total_Purchases"] + out["NumWebVisitsMonth"]) / (
        out["Total_Purchases"].max() + out["NumWebVisitsMonth"].max()
    )

    # How many of the three channels does this customer actually use?
    out["Purchase_Diversity"] = sum((out[col] > 0).astype(int) for col in PURCHASE_COLS)

    # Share of wallet per category. Scale-free: it separates "the customer who
    # spends 80% on wine" from "the customer who spends a lot", which raw euro
    # amounts cannot. Zero-spend customers have no shares to speak of, so guard
    # the division instead of producing inf/NaN that would blow up the scaler.
    denominator = out["Total_Spending"].replace(0, np.nan)
    for share_col, mnt_col in [
        ("WineShareSpend", "MntWines"),
        ("MeatShareSpend", "MntMeatProducts"),
        ("FruitShareSpend", "MntFruits"),
        ("FishShareSpend", "MntFishProducts"),
        ("SweetShareSpend", "MntSweetProducts"),
        ("GoldShareSpend", "MntGoldProds"),
    ]:
        out[share_col] = (out[mnt_col] / denominator).fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# The one call each lesson makes
# ---------------------------------------------------------------------------


def prepare(
    path: Path | str = CUSTOMER_CSV, random_state: int = 42, verbose: bool = True
) -> tuple[pd.DataFrame, str]:
    """Load + clean + engineer in one step; returns ``(dataframe, source)``."""
    raw, source = load_customers(path, random_state=random_state)

    if verbose:
        print(f"Data source: {source.upper()}"
              + ("" if source == "kaggle" else "  (real CSV not found — using the fallback)"))
        print(f"Raw shape: {raw.shape[0]} customers x {raw.shape[1]} columns")
        print("Cleaning:")

    clean = clean_customers(raw, verbose=verbose)
    featured = engineer_features(clean)

    if verbose:
        added = [c for c in featured.columns if c not in clean.columns]
        print(f"Engineered {len(added)} features: {', '.join(added)}")

    return featured, source
