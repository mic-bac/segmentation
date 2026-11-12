"""
cltv_core.py
------------
The CLTV formulas and business assumptions, shared by ``cltv.py`` (which teaches
them) and the teaching app (which drives them from sliders).

They live here rather than inside ``cltv.py`` for the same reason the RFM and
clustering primitives do: importing a ``# %%`` lesson script executes the whole
lesson. ``cltv.py`` prints the real source of these functions as it runs, so
nothing is hidden from the reader.

The retention rate is deliberately *not* a constant here: ``cltv.py`` measures it
from cohorts (section 5), so every function takes it as an argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# The business assumptions — inputs to the model, not facts about the data.
# Every euro the lesson reports depends on them; see cltv.py section 1.
# ------------------------------------------------------------------------------
GROSS_MARGIN = 0.30  # contribution margin: 30 cents of every euro of revenue
DISCOUNT_RATE = 0.10  # d  — a euro next year is worth ~91 cents today
ACQUISITION_COST = 150.0  # AC — marketing spend to win one customer

# A floor on observed tenure, in days. A customer who ordered once has a tenure
# of exactly zero, and dividing by that is a division by zero dressed up as a
# business metric. A full year is the floor, not a token few days: it says "we
# will not annualise from less than a year of history". A customer seen once
# contributed one basket in their first year — not one basket every month for
# ever, which is what a 30-day floor would silently claim. See section 3.
MIN_TENURE_DAYS = 365


def annual_contribution(
    customers: pd.DataFrame,
    gross_margin: float = GROSS_MARGIN,
    min_tenure_days: int = MIN_TENURE_DAYS,
) -> pd.Series:
    """GC_i — the yearly gross contribution of each customer.

    Two steps, and one trap.

    Revenue is not value: selling 100 of anything at a 30% margin contributes 30,
    not 100. So the margin comes off first.

    Then the total has to be annualised, because these customers have been around
    for wildly different lengths of time — from a single day to the full two
    years — and their raw totals are not comparable until they are put on the
    same footing::

        GC_i = (revenue / years observed) * margin

    **The trap** is that "years observed" is zero for anyone who ordered once:
    their first and last purchase are the same day. Divide by that and you get
    infinity, and because a large share of this customer base ordered exactly
    once, that is not an edge case — it is a large share of your answer.

    The floor is where this gets decided, and it is not a formality. Floor at 30
    days and a customer who spent 200 in one visit is recorded as contributing
    2,400 a year, which promotes one-time buyers into the most valuable segment
    you have. Floor at a **year** and the same customer contributes 200 in their
    first year, which is what actually happened. Short floors do not protect you
    from the division by zero so much as hide it behind a plausible number.
    """
    observed_years = np.maximum(customers["TenureDays"], min_tenure_days) / 365.25
    return customers["Revenue"] / observed_years * gross_margin


def clv_multiplier(
    retention_rate: float | pd.Series,
    discount_rate: float = DISCOUNT_RATE,
) -> float | pd.Series:
    """The factor ``Rr / (1 - Rr + d)`` that turns one year of GC into a lifetime.

    Read it as "how many years' worth of contribution a customer is good for, in
    today's money". At Rr = 0.60 and d = 0.10 it is 1.2 years; at Rr = 0.90 it is
    4.5. Retention is the lever, by a wide margin — see section 7.
    """
    return retention_rate / (1 - retention_rate + discount_rate)


def clv_segments(clv: pd.Series) -> pd.Series:
    """Split the base into four value tiers by CLTV quartile.

    Quartiles, not fixed euro thresholds: they adapt to the business and keep the
    tiers usefully sized. The cost is that "Premium" is always exactly 25% of
    customers, however good or bad the year was.
    """
    return pd.qcut(clv, 4, labels=["Low Value", "Medium Value", "High Value", "Premium"])
