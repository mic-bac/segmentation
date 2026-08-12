"""
==============================================================================
CLUSTERING — LETTING THE DATA FIND THE SEGMENTS
==============================================================================
Course: Big Data and Machine Learning
Topic: Unsupervised customer segmentation with Python

Business question
-----------------
"How do we segment customers holistically, by need rather than by till receipt?"

RFM (``rfm.py``) and CLTV (``cltv.py``) both rank customers on purchase outcomes.
Neither can tell you *who* a segment is. Clustering can: it groups customers by
similarity across **everything you know about them at once** — income, age,
household, channel behaviour, campaign response — and hands back groups you can
describe in a sentence and target with a campaign.

The catch: this is **unsupervised** learning. There is no correct answer to
check against, no accuracy to report. A clustering is good if it is stable,
interpretable, and useful for the decision at hand — three things no single
number measures. This lesson therefore spends as much effort on *evaluating*
clusterings as on producing them.

The four algorithms
-------------------
    K-Means         centroid-based    fast, needs k in advance
    Hierarchical    connectivity      builds a full tree, very interpretable
    DBSCAN          density-based     finds odd shapes, isolates outliers
    MeanShift       density modes     finds k itself, but slow

Learning objectives
-------------------
1. Understand why standardisation is mandatory before any distance-based method
2. Choose k with the elbow and silhouette methods — and see where they fail
3. Run four algorithm families and read what each assumes about a "cluster"
4. Evaluate without labels: silhouette, Calinski-Harabasz, Davies-Bouldin
5. Inspect clusters visually via a 2D projection (PCA)
6. Test whether clusters are *stable* under resampling — most reporting skips this
7. Profile and name the segments, then feed them back to RFM and CLTV

Run it
------
    uv sync
    uv run python clustering.py

    SHOW_FIGURES=0 uv run python clustering.py    # numbers only, no browser tabs

Dataset: https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering
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
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples

# Make `src/` importable no matter where this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.clustering_core import (  # noqa: E402
    FAMILIES,
    build_algorithms,
    choose_eps,
    choose_k,
    cluster_fingerprints,
    estimate_meanshift_bandwidth,
    name_segment,
    pca_projection,
    profile_clusters,
    scale_features,
    score_clustering,
    stability,
    sweep_k,
)
from src.personality_data import CLUSTERING_FEATURES, prepare  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RANDOM_STATE = 42
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "1") != "0"

# Per-customer silhouettes for the plot in section 9 stay local: they cannot be
# subsampled the way the summary score can (see section 9).
SILHOUETTE_PLOT_SAMPLE = 800


def show(fig) -> None:
    """Display a figure unless SHOW_FIGURES=0."""
    if SHOW_FIGURES:
        fig.show()


def show_source(fn) -> None:
    """Print the real source of a function, so the lesson can't drift from it.

    The clustering primitives live in ``src/clustering_core.py`` — importing this
    script would re-run the whole lesson, so anything that wants the functions
    (the teaching app, a notebook) needs them somewhere importable. Printing them
    here means the code you read is the code that just ran.
    """
    print(f"\n--- {fn.__module__}.{fn.__name__} " + "-" * (48 - len(fn.__name__)))
    print(inspect.getsource(fn))


# %% 2. LOAD THE CUSTOMER BASE
# ==============================================================================

print("=" * 78)
print("CLUSTERING — UNSUPERVISED CUSTOMER SEGMENTATION")
print("=" * 78)

df, source = prepare()


# %% 3. THE FEATURE MATRIX, AND WHY IT MUST BE STANDARDISED
# ==============================================================================


show_source(scale_features)

X, scaler = scale_features(df, CLUSTERING_FEATURES)

print("\n" + "-" * 78)
print("3. FEATURES AND STANDARDISATION")
print("-" * 78)
print(f"Clustering on {len(CLUSTERING_FEATURES)} features x {len(df):,} customers")

spread = pd.DataFrame(
    {
        "raw_std": df[CLUSTERING_FEATURES].std(),
        "scaled_std": pd.Series(X.std(axis=0), index=CLUSTERING_FEATURES),
    }
).round(2)
print(spread.to_string())
dominant = spread["raw_std"].idxmax()
print(f"\nBefore scaling, '{dominant}' has a spread "
      f"{spread['raw_std'].max() / spread['raw_std'].drop(dominant).max():,.0f}x larger than "
      "every other feature:")
print("it would have decided the clusters single-handedly. After scaling, all features "
      "get an equal vote.")


# %% 4. THE ALGORITHM FAMILIES
# ==============================================================================
# "Clustering" is not one method — it is a set of incompatible answers to the
# question "what counts as a cluster?". Each family draws different shapes.

print("\n" + "-" * 78)
print("4. THE ALGORITHM FAMILIES")
print("-" * 78)
print(FAMILIES.to_string(index=False))
print("\nThis lesson runs one algorithm from each of the first three families, plus")
print("MeanShift — a mode-seeking method that, like DBSCAN, works out k for itself.")


# %% 5. HOW MANY CLUSTERS? THE ELBOW AND SILHOUETTE METHODS
# ==============================================================================


show_source(sweep_k)
sweep = sweep_k(X, range(2, 11))
metric_choice = choose_k(sweep)

print("\n" + "-" * 78)
print("5. CHOOSING k")
print("-" * 78)
print(sweep.round(3).to_string(index=False))
print(f"\nAll three quality metrics agree on their favourite: k = {metric_choice}.")

# ...and we are going to overrule them. Deliberately, and with a reason.
#
# k=2 splits the base into "spends a lot" and "spends little". That is true, and
# useless: it is roughly what RFM's monetary score already told us, and you
# cannot run a marketing department on two segments. Look at the silhouette
# column again — between k=4, 5 and 6 it varies by less than 0.01, which is well
# inside the noise. The metrics are not distinguishing those options; they only
# rule out the *bad* ones. So the choice among them is ours to make on
# interpretability, which is exactly what the evaluation literature says to do.
K = 4

print(f"We will use k = {K} instead, and it is worth being explicit about why:")
print("  * k=2 scores best but splits customers into 'spends a lot' / 'spends little' —")
print("    a distinction RFM already gave us for free.")
sil = sweep.set_index("k")["silhouette"]
print(f"  * between k=4, 5 and 6 the silhouette varies by only "
      f"{sil.loc[[4, 5, 6]].max() - sil.loc[[4, 5, 6]].min():.3f}. The metric cannot separate")
print("    them, so it is not the thing making this decision.")
print(f"  * k={K} gives four segments a marketing team can actually name and act on"
      f" (section 12).")
print("\nThis is the normal state of affairs in unsupervised learning: the metrics narrow")
print("the field, then a human picks. Pretending the number chose itself is the mistake.")

fig_k = make_subplots(
    rows=1, cols=2, subplot_titles=("Elbow: inertia (always falls)", "Silhouette (can peak)")
)
fig_k.add_trace(
    go.Scatter(x=sweep["k"], y=sweep["inertia"], mode="lines+markers", name="Inertia",
               line=dict(color="#4C78A8")),
    row=1, col=1,
)
fig_k.add_trace(
    go.Scatter(x=sweep["k"], y=sweep["silhouette"], mode="lines+markers", name="Silhouette",
               line=dict(color="#F58518")),
    row=1, col=2,
)
for col in (1, 2):
    fig_k.add_vline(x=K, line_dash="dash", line_color="grey",
                    annotation_text=f"chosen k={K}", row=1, col=col)
fig_k.update_xaxes(title_text="Number of clusters (k)")
fig_k.update_layout(height=420, title_text="Choosing k", showlegend=False)
show(fig_k)


# %% 6. DBSCAN NEEDS A RADIUS, NOT A k
# ==============================================================================


show_source(choose_eps)

MIN_SAMPLES = 5
EPS, kdist_curve = choose_eps(X, MIN_SAMPLES)

print("\n" + "-" * 78)
print("6. CHOOSING DBSCAN'S RADIUS (eps)")
print("-" * 78)
print(f"k-distance knee at eps = {EPS:.2f}  (min_samples = {MIN_SAMPLES})")
print("\nHow sensitive is that? Clusters found at nearby values of eps:")
for candidate in [EPS * 0.7, EPS * 0.85, EPS, EPS * 1.3, EPS * 1.6]:
    labels = DBSCAN(eps=candidate, min_samples=MIN_SAMPLES).fit_predict(X)
    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  eps={candidate:4.2f}  ->  {n_found:2d} cluster(s), "
          f"{(labels == -1).mean():5.1%} of customers marked as noise")
print("A parameter this twitchy is the price DBSCAN charges for not needing k.")

fig_eps = go.Figure(
    go.Scatter(x=np.arange(len(kdist_curve)), y=kdist_curve, mode="lines",
               line=dict(color="#4C78A8"), name="k-distance")
)
fig_eps.add_hline(y=EPS, line_dash="dash", line_color="red",
                  annotation_text=f"knee: eps = {EPS:.2f}")
fig_eps.update_layout(
    title=f"k-distance curve (distance to the {MIN_SAMPLES}th nearest neighbour, sorted)",
    xaxis_title="Customers, sorted by distance",
    yaxis_title=f"Distance to {MIN_SAMPLES}th neighbour",
    height=420,
)
show(fig_eps)


# %% 7. FITTING THE FOUR ALGORITHMS
# ==============================================================================

# MeanShift needs a bandwidth — the radius of the window it slides towards dense
# regions. Estimated once here, from a 500-customer subsample, and reused: it is
# a property of the data, not of any single fit, and re-deriving it inside every
# fit (including the five stability refits) would repeat the most expensive step
# in the lesson six times over for the same answer.
BANDWIDTH = estimate_meanshift_bandwidth(X)

# One spec table instead of four copy-pasted blocks. Each `build` takes no
# arguments so the stability analysis in section 10 can refit an algorithm on a
# resample without knowing anything about it.
ALGORITHMS = build_algorithms(K, EPS, BANDWIDTH, MIN_SAMPLES)

print("\n" + "-" * 78)
print("7. FITTING FOUR ALGORITHMS")
print("-" * 78)

labels = {}
for name, spec in ALGORITHMS.items():
    labels[name] = spec["build"]().fit_predict(X)
    found = len(set(labels[name])) - (1 if -1 in labels[name] else 0)
    print(f"  {name:<14} {spec['family']:<20} -> {found} clusters   ({spec['note']})")

print("\nNote which algorithms were *told* how many clusters to find and which worked it")
print("out themselves — and that the two who decided for themselves disagree.")


# %% 8. EVALUATION WITHOUT LABELS
# ==============================================================================


show_source(score_clustering)

scores = pd.DataFrame({name: score_clustering(X, lab) for name, lab in labels.items()}).T

print("\n" + "-" * 78)
print("8. EVALUATION (no labels exist, so these are all internal measures)")
print("-" * 78)
print(scores.round(3).to_string())
print("\n  silhouette         higher is better (-1 to 1)")
print("  calinski_harabasz  higher is better")
print("  davies_bouldin     lower is better")
print("  noise              share of customers the algorithm refused to place")
print("\nRead these as a group, not a leaderboard. A method that dumps 90% of customers")
print("into one cluster and calls the other 10% noise can score beautifully while being")
print("useless — check `clusters`, `noise` and `smallest_cluster` before believing any")
print("of the three scores.")

# The density-based methods deserve a specific verdict rather than a shrug.
tiny = {name: int(row["smallest_cluster"]) for name, row in scores.iterrows()}
dbscan_clusters = int(scores.loc["DBSCAN", "clusters"])
print(f"\nDBSCAN is the instructive failure here. It split the base into {dbscan_clusters} clusters,")
print(f"the smallest holding {tiny['DBSCAN']} customers, and scored the worst silhouette of the four.")
print("That is not a bug — it is what density-based clustering does to data with no dense")
print("regions. DBSCAN looks for islands separated by empty space; standardised customer")
print("data is one diffuse cloud, so it finds no islands and shatters the edges into")
print("fragments instead. Density methods earn their keep on data with genuine voids and")
print("odd shapes (spatial data, anomaly detection) — not on a demographic blob.")
print("\nMeanShift, the other method that chose its own k, landed on "
      f"{int(scores.loc['MeanShift', 'clusters'])} — and K-Means was told to")
print("find 4. Three algorithms, three different answers, none of them wrong: they are")
print("answering three different questions about what a cluster is.")


# %% 9. VISUAL INSPECTION — 2D PROJECTION AND SILHOUETTE PLOTS
# ==============================================================================
# Numbers alone will not tell you whether a clustering makes sense. The standard
# check is to squash the feature space down to two dimensions and look at it.


show_source(pca_projection)

coords, variance = pca_projection(X)

print("\n" + "-" * 78)
print("9. VISUAL INSPECTION")
print("-" * 78)
print(f"The 2D projection keeps {variance.sum():.0%} of the variance "
      f"(PC1 {variance[0]:.0%}, PC2 {variance[1]:.0%}) — enough to trust the broad picture,")
print("not enough to judge fine boundaries.")

fig_pca = make_subplots(rows=2, cols=2, subplot_titles=list(labels))
for position, (name, lab) in enumerate(labels.items()):
    row, col = divmod(position, 2)
    fig_pca.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(size=3, color=lab, colorscale="Viridis", opacity=0.7),
            text=[f"{name}: {'noise' if v == -1 else f'cluster {v}'}" for v in lab],
            hoverinfo="text",
            showlegend=False,
        ),
        row=row + 1, col=col + 1,
    )
fig_pca.update_layout(
    height=760,
    title_text=f"The same customers, four algorithms (PCA projection, "
               f"{variance.sum():.0%} of variance)",
)
fig_pca.update_xaxes(title_text="PC1")
fig_pca.update_yaxes(title_text="PC2")
show(fig_pca)

# The silhouette plot: one horizontal bar per customer, grouped by cluster and
# sorted. Wide, even blocks that all clear the average line mean a healthy
# clustering; a cluster sitting mostly left of the line is one that should
# probably not exist.
# `silhouette_samples` needs a value per customer, so it cannot be subsampled the
# way the summary score can — it is O(n^2) by construction. Drawing it for four
# values of k on the full base means four 5-million-entry distance matrices for a
# picture whose shape is identical on a fraction of the points, so the plot (and
# only the plot) works from a fixed random sample.
plot_index = np.random.default_rng(RANDOM_STATE).choice(
    len(X), size=min(SILHOUETTE_PLOT_SAMPLE, len(X)), replace=False
)
X_plot = X[plot_index]

fig_sil = make_subplots(rows=2, cols=2, subplot_titles=[f"k = {k}" for k in (2, 3, 4, 5)])
for position, k in enumerate((2, 3, 4, 5)):
    row, col = divmod(position, 2)
    k_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_plot)
    per_customer = silhouette_samples(X_plot, k_labels)
    y_offset = 0
    for cluster in range(k):
        values = np.sort(per_customer[k_labels == cluster])
        fig_sil.add_trace(
            go.Scatter(
                x=values,
                y=np.arange(y_offset, y_offset + len(values)),
                mode="lines",
                fill="tozerox",
                line=dict(width=0.5),
                showlegend=False,
                name=f"cluster {cluster}",
            ),
            row=row + 1, col=col + 1,
        )
        y_offset += len(values) + 20
    fig_sil.add_vline(x=per_customer.mean(), line_dash="dash", line_color="red",
                      row=row + 1, col=col + 1)
fig_sil.update_layout(height=760, title_text="Silhouette analysis (red line = average)")
fig_sil.update_xaxes(title_text="Silhouette coefficient")
fig_sil.update_yaxes(showticklabels=False, title_text="Customers, grouped by cluster")
show(fig_sil)


# %% 10. STABILITY — THE CHECK ALMOST NOBODY RUNS
# ==============================================================================


show_source(stability)

print("\n" + "-" * 78)
print("10. STABILITY UNDER RESAMPLING (adjusted Rand index, 5 runs at 80%)")
print("-" * 78)

for name, spec in ALGORITHMS.items():
    mean_ari, std_ari = stability(spec["build"], X, labels[name])
    scores.loc[name, "stability"] = mean_ari
    verdict = "solid" if mean_ari > 0.75 else "shaky" if mean_ari > 0.5 else "unreliable"
    print(f"  {name:<14} ARI = {mean_ari:.3f} +/- {std_ari:.3f}   {verdict}")

print("\n1.0 = the same customers grouped together every time; 0.0 = chance.")
print("A clustering you intend to build campaigns on should be well above 0.75 — the")
print("segments have to survive next quarter's data, not just this quarter's.")


# %% 11. THE HIERARCHY, DRAWN
# ==============================================================================
# Hierarchical clustering's selling point is that it does not commit to a k. It
# builds the whole tree, and you choose where to cut it afterwards.

DENDROGRAM_SAMPLE = 60  # 2,237 leaves would be an unreadable smear

rng = np.random.default_rng(RANDOM_STATE)
sample_index = rng.choice(len(X), size=DENDROGRAM_SAMPLE, replace=False)
linkage_matrix = linkage(X[sample_index], method="ward")
tree = dendrogram(linkage_matrix, no_plot=True)

# Where to cut for K clusters. Merge heights are the last column of the linkage
# matrix, in ascending order: the final merge (-1) takes the tree from 2 clusters
# to 1, so the last (K-1) merges are the ones that must NOT happen. Cutting
# halfway between the last merge we keep and the first we reject is unambiguous.
merge_heights = linkage_matrix[:, 2]
cut_height = (merge_heights[-K] + merge_heights[-(K - 1)]) / 2

fig_tree = go.Figure()
for x_coords, y_coords in zip(tree["icoord"], tree["dcoord"]):
    fig_tree.add_trace(
        go.Scatter(x=x_coords, y=y_coords, mode="lines", line=dict(color="#4C78A8", width=1),
                   hoverinfo="y", showlegend=False)
    )
fig_tree.add_hline(y=cut_height, line_dash="dash", line_color="red",
                   annotation_text=f"cut here for {K} clusters", annotation_position="right")
fig_tree.update_layout(
    title=f"Hierarchical clustering dendrogram (sample of {DENDROGRAM_SAMPLE} customers)",
    xaxis_title="Customers",
    yaxis_title="Merge distance",
    height=560,
    xaxis=dict(showticklabels=False),
)
show(fig_tree)

print("\n" + "-" * 78)
print("11. THE DENDROGRAM")
print("-" * 78)
print(f"Drawn on a sample of {DENDROGRAM_SAMPLE} customers — all {len(X):,} leaves would be an")
print("unreadable smear, and the shape of the tree is what matters.")
print(f"Cut line at height {cut_height:.2f} yields {K} clusters. Slide it up and you merge")
print("segments; slide it down and you split them. Reading that trade-off straight off a")
print("picture is something no other algorithm here offers.")


# %% 12. WHO ARE THESE PEOPLE?
# ==============================================================================
# The payoff. A cluster number is worthless; a described segment is a campaign.


show_source(profile_clusters)

df["Cluster"] = labels["K-Means"]
profiles = profile_clusters(df, labels["K-Means"], CLUSTERING_FEATURES)

print("\n" + "-" * 78)
print("12. SEGMENT PROFILES (K-Means)")
print("-" * 78)
print(profiles.to_string())

# Which feature makes each cluster unusual? Compare each cluster's mean against
# the overall mean, in standard deviations — the same z-score the algorithm saw.
z_scores = cluster_fingerprints(df, profiles, CLUSTERING_FEATURES)

print("\nHow far each segment sits from the average customer (in standard deviations):")
print(z_scores.to_string())

print("\nReading those rows left to right gives each segment a name:")
for cluster in sorted(z_scores.index):
    traits = name_segment(z_scores.loc[cluster])
    print(f"  Cluster {cluster} ({profiles.loc[cluster, 'Customers']:>4} customers): {traits}")

fig_profiles = px.imshow(
    z_scores.T,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-2,
    zmax=2,
    aspect="auto",
    title="Segment fingerprints (standard deviations from the average customer)",
    labels=dict(x="Cluster", y="Feature", color="z-score"),
)
fig_profiles.update_layout(height=460)
show(fig_profiles)


# %% 13. BACK TO THE BUSINESS QUESTION
# ==============================================================================
# Clustering found groups without being told what to look for. Are those groups
# worth different amounts — and did we need clustering to find that out?

print("\n" + "-" * 78)
print("13. ARE THE DATA-DRIVEN SEGMENTS WORTH DIFFERENT AMOUNTS?")
print("-" * 78)

value_check = df.groupby("Cluster").agg(
    Customers=("ID", "count"),
    Mean_Spending=("Total_Spending", "mean"),
    Mean_Income=("Income", "mean"),
    Mean_Recency=("Recency", "mean"),
    Campaign_Response=("Total_Campaigns_Accepted", "mean"),
).round(1)
value_check["Share_of_Revenue"] = (
    df.groupby("Cluster")["Total_Spending"].sum() / df["Total_Spending"].sum() * 100
).round(1)
print(value_check.to_string())

# A value-only segmentation, for comparison: split the same customers into four
# tiers on spending alone — roughly what RFM's monetary score would give you.
df["SpendQuartile"] = pd.qcut(df["Total_Spending"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
agreement = pd.crosstab(df["Cluster"], df["SpendQuartile"])
print("\nAgainst a value-only split of the same customers (spending quartiles):")
print(agreement.to_string())

print(
    "\nThe clusters clearly differ in value — but so would any segmentation built on\n"
    "spending, and the cross-tab shows the two are far from independent. Value is not\n"
    "what clustering contributed.\n"
    "\nWhat it contributed is the *description*. A spending quartile tells you a customer\n"
    "is in the top 25%; the cluster tells you they are affluent, have no children at\n"
    "home, respond to campaigns and rarely browse the website. The first is a ranking.\n"
    "The second is a brief you can hand to a copywriter — and it came from the\n"
    "demographic columns that RFM and CLTV never had access to."
)

print("\n" + "=" * 78)
print(f"Done. {K} segments over {len(df):,} customers, "
      f"{len(CLUSTERING_FEATURES)} features  (data: {source})")
print("=" * 78)
