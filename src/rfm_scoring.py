"""
rfm_scoring.py
--------------
The RFM scoring primitives, shared by ``rfm.py`` (which teaches them) and
``cltv.py`` (which needs RFM segments to compare against customer value).

They live here rather than inside ``rfm.py`` for one reason: importing a ``# %%``
lesson script executes the whole lesson. ``cltv.py`` wants the *functions*, not
another 200 lines of RFM output. One definition, two lessons, no drift — and
``rfm.py`` prints the real source of these functions as it runs, so nothing is
hidden from the reader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_rfm(
    transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Aggregate a transaction log into the three RFM dimensions.

    This is the whole of RFM's data work, and it is three lines of ``groupby``::

        Recency    = days from the customer's last invoice to the snapshot date
        Frequency  = how many distinct invoices they placed
        Monetary   = what those invoices came to

    Two choices in here are worth arguing about, because different teams answer
    them differently and the segments move as a result:

    **Frequency counts invoices, not line items.** A customer who bought
    fourteen different products in one order shopped *once*, not fourteen times.
    Counting rows instead of ``nunique`` on the invoice is the most common way
    to get RFM quietly wrong — it turns big baskets into fake loyalty.

    **The snapshot date is "today".** Recency only means anything relative to a
    fixed point. It defaults to the day after the last invoice in the file, so
    the most recent customer scores 1 day rather than 0 — a zero would break any
    ratio or log taken of recency downstream.
    """
    if snapshot_date is None:
        snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = transactions.groupby("Customer ID")
    rfm = pd.DataFrame(
        {
            "Recency": (snapshot_date - grouped["InvoiceDate"].max()).dt.days,
            "Frequency": grouped["Invoice"].nunique(),
            "Monetary": grouped["Revenue"].sum(),
        }
    )
    return rfm.reset_index().rename(columns={"Customer ID": "Id"})


def score_rfm(rfm: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Score R, F and M on a 1-5 quintile scale (1 = poor, 5 = excellent).

    Two details matter more than they look:

    **Recency is reversed.** A *low* recency (bought yesterday) is *good*, so the
    labels run 5..1 instead of 1..5. Getting this backwards is the single most
    common RFM bug, and it silently inverts every conclusion.

    **All three dimensions are rank-transformed before binning.** Purchase counts
    and euro amounts are lumpy — hundreds of customers share the same value — and
    ``qcut`` cannot place a quintile boundary in the middle of a tie. Ranking
    first breaks the ties, so all five bins actually get filled. (Skip this on one
    dimension and it either raises or silently collapses two bins into one.)
    """
    scored = rfm.copy()
    quintile_labels = list(range(1, bins + 1))

    scored["R_Score"] = pd.qcut(
        scored["Recency"].rank(method="first"), bins, labels=quintile_labels[::-1]
    ).astype(int)
    scored["F_Score"] = pd.qcut(
        scored["Frequency"].rank(method="first"), bins, labels=quintile_labels
    ).astype(int)
    scored["M_Score"] = pd.qcut(
        scored["Monetary"].rank(method="first"), bins, labels=quintile_labels
    ).astype(int)

    # The *profile*: three digits that stay distinguishable ("533" is a recent
    # mid-sized buyer, "155" a lapsed loyal one).
    scored["RFM_Cell"] = (
        scored["R_Score"].astype(str)
        + scored["F_Score"].astype(str)
        + scored["M_Score"].astype(str)
    )
    # The *sum*: convenient for ranking, but it throws the profile away.
    scored["RFM_Sum"] = scored[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return scored


# ---------------------------------------------------------------------------
# View 1: the RFM cube — split each dimension in two, get eight named corners
# ---------------------------------------------------------------------------

# Each octant is keyed by (R_high, F_high, M_high). "High R" means *recent*.
CUBE_NAMES = {
    (True, True, True): "High-Value",
    (True, True, False): "Standard",
    (True, False, True): "High Growth Potential (recent)",
    (True, False, False): "Standard Potential",
    (False, True, True): "High Growth Potential (frequent)",
    (False, True, False): "Maintained",
    (False, False, True): "Key Retention",
    (False, False, False): "Retained",
}


def rfm_cube(rfm: pd.DataFrame) -> pd.Series:
    """Assign each customer to one of the eight octants of the RFM cube.

    The split is at the **median of each raw dimension**, so each of R, F and M
    is genuinely 50/50 and the eight octants partition the base. (Splitting on
    the 1-5 scores instead would leave every score-3 customer homeless.)

    A note on the naming: the lecture slide labels two different octants "High
    Growth Potential" — the recent-but-infrequent big spender and the
    frequent-but-lapsed one. Both are growth stories, but they need opposite
    treatments, so they are disambiguated here rather than merged.
    """
    r_high = rfm["Recency"] <= rfm["Recency"].median()  # recent = "up"
    f_high = rfm["Frequency"] >= rfm["Frequency"].median()
    m_high = rfm["Monetary"] >= rfm["Monetary"].median()

    keys = list(zip(r_high, f_high, m_high))
    return pd.Series([CUBE_NAMES[k] for k in keys], index=rfm.index, name="Cube")


# ---------------------------------------------------------------------------
# View 2: the business ladder — named segments from ordered rules
# ---------------------------------------------------------------------------

# (name, predicate) in priority order. Read top to bottom: the first rule a
# customer satisfies is the one they get.
LADDER_RULES = [
    # Recent, frequent, big spenders — the customers to protect.
    ("Champions", lambda r, f, m: (r >= 4) & (f >= 4) & (m >= 4)),
    # Good on all three, just not the very top.
    ("Loyal Customers", lambda r, f, m: (r >= 3) & (f >= 3) & (m >= 3)),
    # Just arrived and bought once. Checked *before* Potential Loyalists, whose
    # rule (f <= 2) also matches these customers. On this data the ordering moves
    # exactly one customer — but on a base with more recent high-spending
    # one-timers it decides the size of both segments. Rule order is a modelling
    # choice, so make it deliberately rather than by accident of typing order.
    ("Recent Customers", lambda r, f, m: (r >= 4) & (f <= 1)),
    # Recent, spends well, has not made a habit of it yet — grow the frequency.
    ("Potential Loyalists", lambda r, f, m: (r >= 4) & (f <= 2) & (m >= 3)),
    # Were valuable, have gone quiet. The highest-urgency segment.
    ("At Risk", lambda r, f, m: (r <= 2) & (f >= 4) & (m >= 4)),
    ("Can't Lose Them", lambda r, f, m: (r <= 2) & (f >= 3)),
    # Quiet, infrequent, low value — cheapest to let go.
    ("Lost", lambda r, f, m: (r <= 2) & (f <= 2) & (m <= 2)),
]
LADDER_DEFAULT = "Hibernating"


def rfm_ladder(scored: pd.DataFrame) -> pd.Series:
    """Apply the named business-rule ladder to the 1-5 scores.

    Vectorised with ``np.select`` rather than ``DataFrame.apply(axis=1)``: same
    first-match-wins semantics, but it evaluates the rules as whole-column
    boolean masks instead of calling a Python function once per customer.
    """
    r, f, m = scored["R_Score"], scored["F_Score"], scored["M_Score"]
    conditions = [rule(r, f, m) for _, rule in LADDER_RULES]
    choices = [name for name, _ in LADDER_RULES]
    return pd.Series(
        np.select(conditions, choices, default=LADDER_DEFAULT), index=scored.index, name="Segment"
    )


def segment_rfm(
    transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Build, score and segment in one call — RFM end to end.

    ``cltv.py`` uses this to get comparable segments without re-running the
    whole RFM lesson.
    """
    scored = score_rfm(build_rfm(transactions, snapshot_date))
    scored["Cube"] = rfm_cube(scored)
    scored["Segment"] = rfm_ladder(scored)
    return scored
