# Customer Segmentation

Three runnable lessons on segmenting a customer base, built for students: each script is a
readable walkthrough of one method family, with the reasoning written into the code rather
than hidden behind it. They progress from a segmentation you can do in a spreadsheet to one
that needs machine learning, and each lesson ends by naming what it *cannot* do — which is
what the next one picks up.

## 🎯 The three lessons

| Script | Question it answers | Method | Data |
|---|---|---|---|
| `rfm.py` | Who are our best customers *right now*? | Recency / Frequency / Monetary scoring | transactions |
| `cltv.py` | What is a customer worth, and how much may we spend to win one? | Customer Lifetime Value | transactions |
| `clustering.py` | Who *are* these people, and how do we describe them? | K-Means, Hierarchical, DBSCAN, MeanShift | attributes |

```
segmentation/
├── fetch_data.py              one-time download of the transaction dataset
├── src/retail_data.py         transaction log: load, clean, aggregate, cohorts
├── src/rfm_scoring.py         RFM scoring (rfm.py teaches it, cltv.py reuses it)
├── src/personality_data.py    customer attributes for the clustering lesson
├── rfm.py                     lesson 1 — Recency, Frequency, Monetary
├── cltv.py                    lesson 2 — Customer Lifetime Value
├── clustering.py              lesson 3 — unsupervised segmentation
└── data/                      git-ignored, see below
```

## 🚀 Getting started

Requires Python 3.13 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python fetch_data.py     # once — downloads and converts the transaction data
uv run python rfm.py
uv run python cltv.py
uv run python clustering.py
```

Every lesson prints its results and opens interactive Plotly figures in a browser. To get
the numbers without the browser tabs — useful when presenting from a terminal, or in CI:

```bash
SHOW_FIGURES=0 uv run python cltv.py
```

The scripts are `# %%` cell notebooks, so they also run cell by cell in VS Code or Jupyter
(`uv sync --group dev` for the kernel). Paths are anchored to the script, not the working
directory, so they behave the same however you launch them.

Runtimes on a laptop: `rfm.py` and `cltv.py` about 5 seconds; `clustering.py` about 12
seconds, dominated by fitting four algorithms and resampling each five times for the
stability check. Peak memory stays under 500 MB.

## 📊 Two datasets, and why

Segmentation methods do not all want the same data, and pretending otherwise is how you end
up teaching a method on data that cannot support it.

**Lessons 1–2 use a transaction log** —
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), 1,067,371
invoice lines from a UK online gift retailer, Dec 2009 – Dec 2011, ~5,900 identified
customers. RFM needs dated purchases to compute recency and count orders, and CLTV needs
repeat observation of the same people over time to **measure** a retention rate rather than
assume one. `fetch_data.py` downloads it (no login required) and converts it to CSV.

This retailer sells both to the public and to shops that resell its stock, and the two
behave nothing alike — a reseller orders 96 of one item every fortnight. The lessons
therefore run **B2C only**: `is_wholesale()` flags business buyers on three behavioural
signals (median line quantity ≥ 24, average order value ≥ 1,000, or 50+ orders in the
window), and `prepare(b2c_only=True)` — the default — drops them. That removes ~12% of
customers carrying ~46% of revenue, and the filter prints exactly that, because a lesson
that quietly deleted half a business's turnover would be teaching the wrong habit. Pass
`b2c_only=False` to see how much the wholesalers were distorting things; it is a
worthwhile five minutes, since removing them visibly sharpens the RFM signal (recency
against frequency goes from −0.26 to −0.39).

**Lesson 3 uses customer attributes** — Kaggle
[Customer Personality Analysis](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering),
2,240 customers with age, income, household, channel behaviour and campaign responses. A
till receipt records an invoice number and a price, never an age; so no amount of
transaction data will tell you a segment is "affluent households without children", and
that kind of description is the entire point of clustering. Place the CSV at
`data/personality/customer_segmentation.csv` (or `data/customer_segmentation.csv`).

`data/` is git-ignored, so a fresh clone has neither file. **The lessons still run**: each
falls back to synthetic data with the same schema and — importantly for the cohort work —
the same dynamics, and every script reports which source it used.

## 📘 Key concepts

### RFM (`rfm.py`)

Three numbers per customer, aggregated from the transaction log, each scored 1 (poor) to 5
(excellent):

- **Recency** — days since the last invoice (lower is better, so the scale is reversed)
- **Frequency** — number of distinct invoices, *not* line items
- **Monetary** — total revenue

The three scores form a *profile* (`533`), not a sum: `533` and `155` both total 11 but are
opposite customers needing opposite treatment. The lesson builds two segmentations from
those scores — the eight-octant **RFM cube** (*RFM-Würfel*) and a named business-rule ladder
(Champions, At Risk, Lost, …) — and shows where two defensible rule sets disagree.

Strengths: trivial to implement, and a good predictor of who responds next. Limits: it
measures **outcomes only** (*misst nur Endergebnisse*), it says nothing about the future,
and its scores are relative — a customer can change segment without changing behaviour.

### CLTV (`cltv.py`)

The total value of a customer relationship. Basis form, the discounted stream of
contributions:

$$LTV_i = \sum_{t=1}^{T} GC_{it} \left(\frac{1}{1+d}\right)^t$$

With a constant contribution and retention rate, that infinite sum collapses to the closed
form the lesson centres on:

$$LTV_i = GC_i \left(\frac{Rr}{1 - Rr + d}\right) - AC_i$$

- $GC_i$ — gross contribution per year (*Jahresbeitrag*: revenue minus attributable cost)
- $Rr$ — retention rate (*Retentionsrate*), **measured from cohorts**, not assumed
- $d$ — discount rate (*Diskontierungszins*)
- $AC_i$ — acquisition cost (*Akquisekosten*)

The lesson builds a **cohort retention matrix** — every customer assigned to the month of
their first purchase, then tracked forward — and derives $Rr$ from it, including a
recency-banded version measured with a strict look-ahead so it cannot peek at the future it
predicts. It then covers **historical vs predictive** CLTV, the sum of all CLTVs as
**customer equity** (an input to valuing the company), and CLTV as the **ceiling on
acquisition cost**.

Two things it insists on: the annualisation floor is a modelling decision that can silently
promote one-time buyers above two-year relationships, and moving $Rr$ from 60% to 90%
changes every customer's value nearly fourfold — which is why measuring it is worth an
afternoon.

### Clustering (`clustering.py`)

Unsupervised learning — *es gibt keine "richtigen" Lösungen*. Group customers by similarity
across everything known about them at once, maximising similarity within clusters and
difference between them.

| Family | A cluster is… | Algorithms |
|---|---|---|
| Centroid-based | proximity to a centre | K-Means, K-Medoids |
| Connectivity-based | proximity between clusters | Hierarchical (agglomerative / divisive) |
| Density-based | dense regions separated by sparse ones | DBSCAN, OPTICS, HDBSCAN |
| Distribution-based | likelihood under a fitted distribution | Gaussian Mixture Models |

Because there are no labels, evaluation is *subjektiv und abhängig vom Anwendungsfall*. The
lesson uses all four standard checks: the **silhouette score**, **Davies-Bouldin** and
**Calinski-Harabasz** indices, **visual inspection** via a 2D PCA projection, and a
**stability analysis** (*Stabilitätsanalyse*) that refits on resampled data and measures
agreement — the check most reporting skips, and the one that fails most often here.

It also refuses to let a metric make a judgement call: every index prefers k=2, the lesson
explains why it uses k=4 anyway, and says plainly that the metrics narrow the field while a
human picks.

## 📝 License

MIT — see [LICENSE](LICENSE).
