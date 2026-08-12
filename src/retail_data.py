"""
retail_data.py
--------------
Shared data access for the two value-based lessons: ``rfm.py`` and ``cltv.py``.

Both work from the same **transaction log** — one row per line item on one
invoice — because that is what RFM and CLTV actually need. Recency is a real
date difference, frequency is a real invoice count, and retention can be
*measured* from cohorts instead of assumed.

Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Run ``uv run python fetch_data.py`` once to download it.

Shape of the data
-----------------
1,067,371 line items from a UK online gift retailer, 2009-12-01 to 2011-12-09 —
just over two years, which is the minimum needed to watch a cohort survive a
full year and is why this dataset is the standard one for teaching CLTV.

    Invoice      order number; a leading "C" marks a cancellation
    StockCode    product code; a handful are not products at all
    Quantity     units; negative on returns
    InvoiceDate  timestamp
    Price        unit price
    Customer ID  missing on ~23% of rows (guest checkout)
    Country      almost entirely United Kingdom

Graceful degradation
--------------------
``data/`` is git-ignored, so a fresh clone has no CSV. ``load_transactions``
then falls back to a **synthetic** transaction log with the same schema and, more
importantly, the same *dynamics* — customers who stop buying — so the cohort
retention curves still decay and the CLTV lesson still has something real to
measure. Every lesson reports which source it used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Paths are anchored to this file, not the working directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RETAIL_CSV = DATA_DIR / "retail" / "online_retail_II.csv"

# Reading with explicit dtypes rather than letting pandas guess. On a million
# rows this is the difference between ~150 MB and well over half a gigabyte:
# `Country` has 43 distinct values and `category` stores it once, while the
# default `object` dtype stores a separate Python string per row.
COLUMN_TYPES = {
    "Invoice": "string",
    "StockCode": "string",
    "Quantity": "int32",
    "Price": "float64",
    "Customer ID": "float64",  # float, not int — it has missing values
    "Country": "category",
}

# Stock codes that are not products. Postage, bank charges and manual
# adjustments are real rows in the ledger, but counting them as customer
# spending would inflate every "Monetary" figure with the retailer's own fees.
# Dropping them barely moves the row count and changes what revenue *means*.
NON_PRODUCT_CODES = {
    "POST",  # postage
    "DOT",  # dotcom postage
    "M",  # manual adjustment
    "m",
    "C2",  # carriage
    "D",  # discount
    "S",  # samples
    "BANK CHARGES",
    "ADJUST",  # stock adjustments
    "ADJUST2",
    "AMAZONFEE",
    "CRUK",  # charity donation
    "TEST001",
    "TEST002",
}
NON_PRODUCT_PREFIXES = ("gift_",)  # gift vouchers: sold, but not a product

# ---------------------------------------------------------------------------
# Telling businesses apart from consumers
#
# This retailer sells to the public *and* to shops that resell its stock, and
# the two behave nothing alike. A wholesaler orders 96 of one item, spends
# thousands per order, and restocks every fortnight; a consumer buys two of
# something for a birthday. Left mixed together, a few hundred businesses
# dominate every euro-denominated statistic and the "customer segments" end up
# describing the retailer's trade counter rather than its shoppers.
#
# There is no flag in the data saying which is which, so it has to be inferred
# from behaviour. Three independent signals, any one of which is enough:
# ---------------------------------------------------------------------------
WHOLESALE_MEDIAN_QUANTITY = 24  # typical line is a case, not an item
WHOLESALE_AVG_ORDER_VALUE = 1_000.0  # orders the size of a business's
WHOLESALE_ORDER_COUNT = 50  # restocking every fortnight for two years


# ---------------------------------------------------------------------------
# Loading (with a synthetic fallback so the lesson always runs)
# ---------------------------------------------------------------------------


def load_transactions(
    path: Path | str = RETAIL_CSV, random_state: int = 42
) -> tuple[pd.DataFrame, str]:
    """Load the raw transaction log; returns ``(dataframe, source)``.

    ``source`` is ``"retail"`` or ``"synthetic"`` so each lesson can be honest
    about which data produced the numbers on screen.
    """
    path = Path(path)
    if path.exists():
        frame = pd.read_csv(path, dtype=COLUMN_TYPES, parse_dates=["InvoiceDate"])
        return frame, "retail"
    return make_synthetic_transactions(random_state=random_state), "synthetic"


def make_synthetic_transactions(
    n_customers: int = 4_000,
    start: str = "2009-12-01",
    end: str = "2011-12-09",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a transaction log with the same schema and dynamics as the original.

    The dynamics are the part that matters. A fallback that gave every customer
    a constant purchase rate would produce flat cohort retention curves, and the
    CLTV lesson would have nothing to measure. So each customer here gets:

    * an **acquisition date**, spread across the window;
    * a **lifetime**, drawn from an exponential distribution — most customers
      buy once or twice and drift away, a few stay for the whole period;
    * a **purchase rate**, so within their lifetime orders arrive as a Poisson
      process.

    That is the simplest generator that produces a genuinely decaying retention
    curve, which is exactly what ``cltv.py`` needs to fit a retention rate to.
    """
    rng = np.random.default_rng(random_state)
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    window_days = (end_date - start_date).days

    # About one customer in eight is a business — a shop reselling the stock.
    # They are generated here so the B2C filter has something real to find; a
    # fallback where the filter removes nobody would hide that step entirely.
    is_business = rng.random(n_customers) < 0.12

    rows = []
    for customer in range(n_customers):
        customer_id = 12_000 + customer
        business = bool(is_business[customer])

        acquired_on = start_date + pd.Timedelta(days=int(rng.integers(0, window_days)))
        # Mean lifetime a bit over a year; the exponential's long tail supplies
        # the minority of loyal customers that carry most of the revenue. The
        # mean is tuned so the measured annual retention lands in the same
        # 50-65% range as the real data — otherwise the fallback would tell
        # students a far gloomier story than the dataset it stands in for.
        lifetime_days = float(rng.exponential(430))
        # Orders per year, lognormal: most customers are occasional, a few
        # frequent. Businesses restock on a schedule, so they order far more often
        # and keep going — which is what makes them distinguishable at all.
        if business:
            orders_per_year = float(rng.lognormal(np.log(7.0), 0.5))
            lifetime_days = max(lifetime_days, float(rng.uniform(400, 730)))
        else:
            orders_per_year = float(rng.lognormal(np.log(4.0), 0.7))

        # Poisson process: exponential gaps between consecutive orders.
        offset = 0.0
        order_offsets = []
        while offset <= lifetime_days:
            order_offsets.append(offset)
            offset += float(rng.exponential(365.25 / orders_per_year))

        # How big this customer's baskets tend to be, and how expensive.
        basket_size = max(1, int(rng.lognormal(np.log(12), 0.5)))
        price_level = float(rng.lognormal(np.log(2.6), 0.5))
        # Consumers buy a couple of something; businesses buy it by the case.
        units_per_line = float(rng.uniform(24, 44)) if business else 6.0

        for order_number, day_offset in enumerate(order_offsets):
            order_date = acquired_on + pd.Timedelta(days=day_offset)
            if order_date > end_date:
                break
            # Shop hours, so the timestamps look like the real thing.
            order_date += pd.Timedelta(hours=int(rng.integers(8, 19)),
                                       minutes=int(rng.integers(0, 60)))
            invoice = f"5{customer:05d}{order_number:02d}"

            for _ in range(rng.poisson(basket_size) + 1):
                rows.append(
                    (
                        invoice,
                        f"{rng.integers(10000, 99999)}",
                        int(rng.poisson(units_per_line) + 1),
                        order_date,
                        round(max(0.25, rng.lognormal(np.log(price_level), 0.6)), 2),
                        float(customer_id),
                        "United Kingdom" if rng.random() < 0.92 else "Germany",
                    )
                )

    frame = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID",
                 "Country"],
    )

    # Reproduce the quirks the cleaning step exists to handle, so it still has
    # work to do and the printed output tells the same story on either source.
    # The proportions mirror the real file — guest checkout is common, returns
    # are uncommon, postage lines are rare — rather than being split evenly,
    # which would make the cleaning report look obviously manufactured.
    quirks = rng.permutation(len(frame))
    n_guests = int(len(frame) * 0.23)
    n_cancels = int(len(frame) * 0.02)
    n_postage = int(len(frame) * 0.004)
    guests = quirks[:n_guests]
    cancels = quirks[n_guests:n_guests + n_cancels]
    postage = quirks[n_guests + n_cancels:n_guests + n_cancels + n_postage]

    frame.loc[guests, "Customer ID"] = np.nan  # guest checkout
    frame.loc[cancels, "Invoice"] = "C" + frame.loc[cancels, "Invoice"]
    frame.loc[cancels, "Quantity"] *= -1  # returns
    frame.loc[postage, "StockCode"] = "POST"

    return frame.astype(COLUMN_TYPES)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_transactions(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Turn the raw ledger into rows that represent customer purchases.

    Four things in this file are not customer purchases, and each has to go for
    a different reason. The counts are printed because "how many rows did my
    cleaning throw away, and why" is a question you should always be able to
    answer about your own pipeline.
    """
    out = df.copy()
    n_raw = len(out)
    removed = {}

    # 1. No customer, no customer analysis. About a quarter of rows are guest
    #    checkouts with no ID — real revenue, but impossible to attribute to
    #    anybody, so they cannot take part in a segmentation. (Note what this
    #    means for the totals below: they describe identified customers only.)
    before = len(out)
    out = out.dropna(subset=["Customer ID"])
    removed["no customer ID (guest checkout)"] = before - len(out)

    # 2. Cancellations. A "C" invoice reverses an earlier order. They could be
    #    netted off the original purchase; here they are dropped, which slightly
    #    overstates what customers kept. The alternative — matching every
    #    cancellation to its original invoice — is a lesson of its own.
    before = len(out)
    out = out[~out["Invoice"].str.startswith("C", na=False)]
    removed["cancellations"] = before - len(out)

    # 3. Not products: postage, bank charges, manual adjustments, vouchers.
    before = len(out)
    is_non_product = out["StockCode"].isin(NON_PRODUCT_CODES) | out["StockCode"].str.startswith(
        NON_PRODUCT_PREFIXES, na=False
    )
    out = out[~is_non_product]
    removed["not a product (postage, fees, adjustments)"] = before - len(out)

    # 4. Whatever is left with a non-positive quantity or price is a data error
    #    rather than a sale — returns without a "C" invoice, free samples,
    #    write-offs. They would subtract from customers' spending.
    before = len(out)
    out = out[(out["Quantity"] > 0) & (out["Price"] > 0)]
    removed["non-positive quantity or price"] = before - len(out)

    out = out.copy()
    out["Customer ID"] = out["Customer ID"].astype("int64")
    # Line revenue, computed once here so no lesson has to remember the formula.
    out["Revenue"] = out["Quantity"] * out["Price"]
    out = out.reset_index(drop=True)

    if verbose:
        for reason, count in removed.items():
            print(f"  removed {count:>7,} rows — {reason}")
        print(f"  rows: {n_raw:,} -> {len(out):,} "
              f"({len(out) / n_raw:.1%} kept, {out['Customer ID'].nunique():,} customers)")
        # The counts must account for every dropped row; if they ever stop
        # doing so, a cleaning step is dropping rows it did not report.
        assert sum(removed.values()) == n_raw - len(out), "cleaning counts do not add up"

    return out


# ---------------------------------------------------------------------------
# B2C filter
# ---------------------------------------------------------------------------


def wholesale_profile(transactions: pd.DataFrame) -> pd.DataFrame:
    """Per-customer measurements of the three business-buying signals."""
    grouped = transactions.groupby("Customer ID")
    orders = grouped["Invoice"].nunique()
    return pd.DataFrame(
        {
            "MedianLineQuantity": grouped["Quantity"].median(),
            "AverageOrderValue": grouped["Revenue"].sum() / orders,
            "Orders": orders,
        }
    )


def is_wholesale(transactions: pd.DataFrame) -> pd.Series:
    """Flag customers who buy like a business rather than a consumer.

    Returns a boolean Series indexed by ``Customer ID``. A customer is treated
    as wholesale if **any** of the three signals fires:

    * **median line quantity >= 24** — every purchase is a case, not an item.
      The median (not the mean) matters: a consumer might buy 50 party favours
      once, but only a reseller does it on every line of every order.
    * **average order value >= 1000** — orders the size a household does not place.
    * **>= 50 orders** in the two-year window — restocking every fortnight, which
      is a supply chain rather than shopping.

    Any single threshold could be argued with, and that is the honest position:
    this is a judgement encoded as a rule, not a fact recovered from the data.
    What it must not be is invisible — hence ``filter_b2c`` reporting exactly
    what it removed, so the call can be inspected and overruled.
    """
    profile = wholesale_profile(transactions)
    return (
        (profile["MedianLineQuantity"] >= WHOLESALE_MEDIAN_QUANTITY)
        | (profile["AverageOrderValue"] >= WHOLESALE_AVG_ORDER_VALUE)
        | (profile["Orders"] >= WHOLESALE_ORDER_COUNT)
    )


def filter_b2c(transactions: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only consumer transactions, and report what that cost.

    The revenue share removed is deliberately printed: it is large, and a lesson
    that quietly deleted nearly half a business's turnover without saying so
    would be teaching the wrong habit.
    """
    wholesale = is_wholesale(transactions)
    business_ids = set(wholesale.index[wholesale])
    consumer = transactions[~transactions["Customer ID"].isin(business_ids)].reset_index(drop=True)

    if verbose:
        profile = wholesale_profile(transactions)
        revenue = transactions.groupby("Customer ID")["Revenue"].sum()
        by_signal = {
            f"median line quantity >= {WHOLESALE_MEDIAN_QUANTITY}":
                profile["MedianLineQuantity"] >= WHOLESALE_MEDIAN_QUANTITY,
            f"average order value >= {WHOLESALE_AVG_ORDER_VALUE:,.0f}":
                profile["AverageOrderValue"] >= WHOLESALE_AVG_ORDER_VALUE,
            f"{WHOLESALE_ORDER_COUNT}+ orders in the window":
                profile["Orders"] >= WHOLESALE_ORDER_COUNT,
        }
        print("  business buyers identified by:")
        for label, mask in by_signal.items():
            print(f"    {int(mask.sum()):>5,} customers — {label}")
        print(f"  removed {int(wholesale.sum()):,} of {len(wholesale):,} customers "
              f"({wholesale.mean():.1%}), carrying "
              f"{revenue[wholesale].sum() / revenue.sum():.1%} of revenue")
        print(f"  B2C log: {len(consumer):,} line items, "
              f"{consumer['Customer ID'].nunique():,} customers")

    return consumer


# ---------------------------------------------------------------------------
# Aggregation: transactions -> customers
# ---------------------------------------------------------------------------


def customer_table(
    transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Collapse the transaction log to one row per customer.

    ``snapshot_date`` is the day we pretend to be standing on — everything
    "since the last purchase" is measured from it. It defaults to the day after
    the last invoice in the file, because using the last invoice date itself
    would give the single most recent customer a recency of exactly zero and
    quietly break any log or ratio taken of it later.
    """
    if snapshot_date is None:
        snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = transactions.groupby("Customer ID")
    customers = pd.DataFrame(
        {
            "FirstPurchase": grouped["InvoiceDate"].min(),
            "LastPurchase": grouped["InvoiceDate"].max(),
            "Orders": grouped["Invoice"].nunique(),
            "Items": grouped["Quantity"].sum(),
            "Revenue": grouped["Revenue"].sum(),
        }
    )

    customers["Recency"] = (snapshot_date - customers["LastPurchase"]).dt.days
    # Tenure is *observed* time as a customer: first purchase to last. A customer
    # who ordered once has a tenure of zero — see cltv.py, where that matters.
    customers["TenureDays"] = (customers["LastPurchase"] - customers["FirstPurchase"]).dt.days
    customers["AverageOrderValue"] = customers["Revenue"] / customers["Orders"]
    return customers.reset_index()


# ---------------------------------------------------------------------------
# Cohorts: the only honest way to measure retention
# ---------------------------------------------------------------------------


def cohort_retention(transactions: pd.DataFrame) -> pd.DataFrame:
    """Monthly acquisition cohort x months-since-acquisition retention matrix.

    Every customer is assigned to the month of their **first** purchase — their
    cohort — and then we ask, for each later month, what share of that cohort
    bought again. Rows are cohorts, columns are months since acquisition, values
    are retention rates. Column 0 is always 100% by construction.

    This is the only way to measure retention from a transaction log, and it is
    what the previous version of this lesson could not do: a single snapshot of
    customer attributes has no repeat observation of anybody, so a retention
    rate could only be assumed. Here it is measured.

    The lower-left of the matrix is empty on purpose: a cohort acquired three
    months before the data ends cannot have a twelve-month retention figure yet.
    Reading those NaNs as zeros is a classic way to invent a collapse in
    retention that never happened.
    """
    frame = transactions[["Customer ID", "InvoiceDate"]].copy()
    order_month = frame["InvoiceDate"].dt.to_period("M")
    cohort_month = frame.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period("M")

    frame["Cohort"] = cohort_month
    frame["PeriodsSince"] = (order_month - cohort_month).apply(lambda offset: offset.n)

    active = frame.groupby(["Cohort", "PeriodsSince"])["Customer ID"].nunique().unstack()
    cohort_sizes = active[0]
    return active.div(cohort_sizes, axis=0)


def retention_by_recency_band(
    transactions: pd.DataFrame, bands: tuple[int, ...] = (30, 90, 180)
) -> pd.DataFrame:
    """Measure the annual retention rate *separately for each recency band*.

    One retention rate for the whole base is obviously too crude: somebody who
    bought last week is not as likely to be gone as somebody silent for six
    months. This measures how much that matters, rather than asserting it.

    The design is the same look-ahead as ``annual_retention``, split by
    behaviour: stand at the one-year mark, sort everyone who has bought so far
    by how long they have been quiet, then look forward and see who came back.
    Because the "who came back" half comes strictly *after* the "how quiet are
    they" half, nothing here can peek at the future it is trying to predict —
    the discipline any predictive feature has to survive.

    Returns one row per band: customers, how many returned, and the rate.
    """
    start = transactions["InvoiceDate"].min()
    split = start + pd.DateOffset(years=1)

    history = transactions[transactions["InvoiceDate"] < split]
    future = transactions[transactions["InvoiceDate"] >= split]

    last_seen = history.groupby("Customer ID")["InvoiceDate"].max()
    recency = (split - last_seen).dt.days
    returned = last_seen.index.isin(future["Customer ID"].unique())

    # Recency here is measured at the one-year mark, so it cannot exceed ~365
    # days. Only keep band edges that actually fall inside the observed range,
    # then close the last band above the maximum — otherwise a band edge can
    # coincide with the final edge and `pd.cut` rejects the duplicate.
    usable = [edge for edge in bands if edge <= recency.max()]
    edges = [0, *usable, int(recency.max()) + 1]
    labels = [f"{edges[i]}-{edges[i + 1] - 1}d" for i in range(len(edges) - 1)]
    band = pd.cut(recency, bins=edges, labels=labels, right=False)

    # Keep `band` as the ordered categorical `pd.cut` produced. Converting it to
    # a plain array here would drop that ordering, and the groupby below would
    # come back sorted as text — "180-364d" before "30-89d".
    frame = pd.DataFrame({"Band": band, "Returned": returned}, index=recency.index)
    summary = frame.groupby("Band", observed=True).agg(
        Customers=("Returned", "size"), Returned=("Returned", "sum")
    )
    summary["RetentionRate"] = summary["Returned"] / summary["Customers"]
    return summary


def annual_retention(transactions: pd.DataFrame) -> dict:
    """Measure the annual retention rate Rr — the CLTV formula's key input.

    The CLTV formula works in **yearly** periods, so it needs a yearly number:
    of the customers acquired in the first twelve months, what share bought
    again during the following twelve months?

    That is a different question from the monthly matrix above, and a much more
    stable one. Monthly retention in online retail looks alarming (single-digit
    percentages are normal) because most customers simply do not shop every
    month — it does not mean they have gone. Annual repeat purchase is the right
    resolution for an annual model.

    Returns the rate plus the counts behind it, so the lesson can show its work.
    """
    first_purchase = transactions.groupby("Customer ID")["InvoiceDate"].min()
    start = transactions["InvoiceDate"].min()
    year_one_end = start + pd.DateOffset(years=1)

    cohort = first_purchase[first_purchase < year_one_end].index
    # Who from that cohort came back in the second year?
    second_year = transactions[transactions["InvoiceDate"] >= year_one_end]
    returned = set(second_year["Customer ID"].unique()) & set(cohort)

    return {
        "retention_rate": len(returned) / len(cohort),
        "cohort_size": len(cohort),
        "returned": len(returned),
        "year_one": (start, year_one_end),
    }


# ---------------------------------------------------------------------------
# The one call each lesson makes
# ---------------------------------------------------------------------------


def prepare(
    path: Path | str = RETAIL_CSV,
    random_state: int = 42,
    b2c_only: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load + clean the transaction log; returns ``(transactions, source)``.

    ``b2c_only`` (on by default) drops the retailer's business customers, so the
    lessons describe consumers rather than the trade counter. Set it to ``False``
    to see how much the wholesalers were distorting things.
    """
    raw, source = load_transactions(path, random_state=random_state)

    if verbose:
        label = "ONLINE RETAIL II" if source == "retail" else "SYNTHETIC"
        hint = "" if source == "retail" else "  (no CSV found — run: uv run python fetch_data.py)"
        print(f"Data source: {label}{hint}")
        print(f"Raw log: {len(raw):,} line items, "
              f"{raw['InvoiceDate'].min():%Y-%m-%d} to {raw['InvoiceDate'].max():%Y-%m-%d}")
        print("Cleaning:")

    clean = clean_transactions(raw, verbose=verbose)

    if b2c_only:
        if verbose:
            print("Restricting to consumers (B2C):")
        clean = filter_b2c(clean, verbose=verbose)

    return clean, source
