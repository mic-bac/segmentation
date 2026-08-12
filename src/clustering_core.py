"""
clustering_core.py
------------------
The clustering primitives, shared by ``clustering.py`` (which teaches them) and
the teaching app (which drives them interactively).

They live here rather than inside ``clustering.py`` for the same reason the RFM
scoring does: importing a ``# %%`` lesson script executes the whole lesson —
loading data, fitting four algorithms, printing 200 lines. Anything that wants
the *functions* needs them somewhere importable. ``clustering.py`` prints the
real source of these as it runs, so nothing is hidden from the reader.

Everything here is UI-free and returns data. No printing, no plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# The silhouette score compares every point against every other one, so it costs
# O(n^2) time and memory — on a few thousand customers that is a multi-million
# entry distance matrix, rebuilt on every call, and a lesson calls it dozens of
# times. scikit-learn can estimate it from a random subsample instead, which is
# accurate to about two decimal places here and turns minutes into seconds. The
# same cap is what keeps the teaching app responsive.
METRIC_SAMPLE = 1_000


# ---------------------------------------------------------------------------
# The algorithm families (lecture taxonomy)
# ---------------------------------------------------------------------------

FAMILIES = pd.DataFrame(
    [
        ("Centroid-based", "Distance to a cluster centre", "K-Means, K-Medoids", "yes"),
        ("Connectivity-based", "Distance between clusters", "Hierarchical (agglomerative)",
         "at the cut"),
        ("Density-based", "Dense regions separated by sparse ones", "DBSCAN, OPTICS, HDBSCAN",
         "no"),
        ("Distribution-based", "Likelihood under a fitted distribution",
         "Gaussian Mixture Models", "yes"),
        ("Graph-based", "Distance along a graph", "Spectral, Affinity Propagation", "varies"),
        ("Compression-based", "Reduce dimensions, then cluster", "BIRCH", "yes"),
    ],
    columns=["Family", "A cluster is...", "Algorithms", "Needs k?"],
)


# ---------------------------------------------------------------------------
# Preparing the feature matrix
# ---------------------------------------------------------------------------


def scale_features(
    customers: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the feature matrix to mean 0, standard deviation 1.

    Not a formality — the single most important line in the lesson. Every
    algorithm below measures the distance between two customers, and distance is
    dominated by whichever column has the largest numbers. Income runs to six
    figures while ``Total_Children`` runs 0-3, so on raw data the "distance"
    between two customers is, to four decimal places, the difference in their
    incomes. Everything else would be along for the ride.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(customers[features]), scaler


# ---------------------------------------------------------------------------
# Choosing the parameters
# ---------------------------------------------------------------------------


def silhouette(X: np.ndarray, labels: np.ndarray, random_state: int = RANDOM_STATE) -> float:
    """Silhouette score, estimated from a fixed random subsample.

    Deterministic (fixed ``random_state``), so repeated calls on the same
    clustering give the same number — a subsampled metric that jittered between
    runs would be worse than useless for comparing algorithms.
    """
    return float(
        silhouette_score(X, labels, sample_size=min(METRIC_SAMPLE, len(X)),
                         random_state=random_state)
    )


def sweep_k(X: np.ndarray, k_values, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Fit K-Means for each k and score the result four ways.

    ``inertia`` (within-cluster sum of squares) always falls as k rises — with
    one cluster per customer it reaches zero — so it can never pick k on its own.
    You look for the "elbow" where the improvement flattens out.

    The other three are genuine quality scores that can go up *or* down::

        silhouette          -1 to 1, higher is better. How much closer a point is
                            to its own cluster than to the nearest other one.
        Calinski-Harabasz   higher is better. Between-cluster spread over
                            within-cluster spread.
        Davies-Bouldin      lower is better, 0 is perfect. Average similarity
                            between each cluster and its most similar neighbour.
    """
    rows = []
    for k in k_values:
        fitted = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": fitted.inertia_,
                "silhouette": silhouette(X, fitted.labels_, random_state),
                "calinski_harabasz": calinski_harabasz_score(X, fitted.labels_),
                "davies_bouldin": davies_bouldin_score(X, fitted.labels_),
            }
        )
    return pd.DataFrame(rows)


def choose_k(sweep: pd.DataFrame) -> int:
    """Pick the k with the best silhouette score."""
    return int(sweep.loc[sweep["silhouette"].idxmax(), "k"])


def knee_index(curve: np.ndarray) -> int:
    """Index of the point furthest from the chord joining the curve's endpoints.

    The standard trick for finding an "elbow" without eyeballing it: draw a
    straight line from the first point to the last, and take the point that
    bulges furthest away from it.
    """
    n = len(curve)
    coords = np.column_stack([np.arange(n), curve])
    start, end = coords[0], coords[-1]
    line = end - start
    line = line / np.linalg.norm(line)
    offsets = coords - start
    # Distance to the line = length of the component perpendicular to it.
    projections = np.outer(offsets @ line, line)
    return int(np.argmax(np.linalg.norm(offsets - projections, axis=1)))


def choose_eps(X: np.ndarray, min_samples: int = 5) -> tuple[float, np.ndarray]:
    """Pick DBSCAN's ``eps`` from the k-distance curve; returns ``(eps, curve)``.

    Sort every point by how far away its ``min_samples``-th nearest neighbour is.
    Points inside a dense cluster have small distances; outliers have large ones.
    The knee of that sorted curve is where "still in a cluster" turns into "out
    in the sparse tail", which is precisely what ``eps`` is supposed to mean.

    This replaces the usual "we picked 1.5 because it seemed reasonable", and it
    matters: DBSCAN's results swing wildly with ``eps``.
    """
    neighbours = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = neighbours.kneighbors(X)
    curve = np.sort(distances[:, -1])
    return float(curve[knee_index(curve)]), curve


# ---------------------------------------------------------------------------
# The algorithms
# ---------------------------------------------------------------------------


def build_algorithms(
    k: int, eps: float, bandwidth: float, min_samples: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict]:
    """One spec table instead of four copy-pasted fitting blocks.

    Each ``build`` takes no arguments, so the stability analysis can refit an
    algorithm on a resample without knowing anything about it. Returned as a
    function rather than a module constant because the app varies ``k`` live.
    """
    return {
        "K-Means": {
            "family": "Centroid-based",
            "build": lambda: KMeans(n_clusters=k, random_state=random_state, n_init=10),
            "note": f"k={k} supplied by us",
        },
        "Hierarchical": {
            "family": "Connectivity-based",
            "build": lambda: AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "note": f"ward linkage, tree cut at k={k}",
        },
        "DBSCAN": {
            "family": "Density-based",
            "build": lambda: DBSCAN(eps=eps, min_samples=min_samples),
            "note": f"eps={eps:.2f} from the k-distance knee",
        },
        "MeanShift": {
            "family": "Density modes",
            # `bin_seeding` starts the search from a coarse grid rather than from
            # every point — same answer here, several times faster.
            "build": lambda: MeanShift(bandwidth=bandwidth, bin_seeding=True),
            "note": f"bandwidth {bandwidth:.2f}, estimated from the data",
        },
    }


def estimate_meanshift_bandwidth(
    X: np.ndarray, quantile: float = 0.2, n_samples: int = 500,
    random_state: int = RANDOM_STATE,
) -> float:
    """MeanShift's window radius, estimated once from a subsample.

    It is a property of the data, not of any single fit, so re-deriving it inside
    every fit (including the stability refits) would repeat the most expensive
    step in the lesson for the same answer.
    """
    return float(
        estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, random_state=random_state)
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def score_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Score a clustering with the three internal metrics.

    Noise points (DBSCAN's label ``-1``) are **excluded** before scoring. They
    are not a cluster — they are the points the algorithm refused to place — and
    letting them in as if they were one silently penalises every density-based
    method against K-Means, which is not allowed to have any. The share of noise
    is reported separately instead, because it is a result in its own right.

    Metrics need at least two clusters to exist; below that they are undefined
    rather than bad, so they come back as NaN.
    """
    signal = labels != -1
    n_clusters = len(set(labels[signal]))
    result = {
        "clusters": n_clusters,
        "noise": float((~signal).mean()),
        "smallest_cluster": (
            int(pd.Series(labels[signal]).value_counts().min()) if n_clusters else 0
        ),
    }
    if n_clusters < 2:
        return {**result, "silhouette": np.nan, "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan}
    return {
        **result,
        "silhouette": silhouette(X[signal], labels[signal]),
        "calinski_harabasz": calinski_harabasz_score(X[signal], labels[signal]),
        "davies_bouldin": davies_bouldin_score(X[signal], labels[signal]),
    }


def stability(
    build,
    X: np.ndarray,
    reference: np.ndarray,
    n_runs: int = 5,
    sample_fraction: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Refit on random subsamples and measure agreement with the full-data fit.

    Returns ``(mean, std)`` of the **adjusted Rand index** — 1.0 means the
    resampled clustering put exactly the same customers together, 0.0 means it
    agrees no better than chance.

    Why bother: every algorithm returns *something* on any input, and none of
    them warn you when the structure they found is an accident of this particular
    sample. If dropping a random fifth of your customers reshuffles the segments,
    the segments were never there — and you would never learn that from a
    silhouette score.
    """
    rng = np.random.default_rng(random_state)
    n_sample = int(len(X) * sample_fraction)
    agreements = []
    for _ in range(n_runs):
        index = rng.choice(len(X), size=n_sample, replace=False)
        resampled = build().fit_predict(X[index])
        agreements.append(adjusted_rand_score(reference[index], resampled))
    return float(np.mean(agreements)), float(np.std(agreements))


def pca_projection(
    X: np.ndarray, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the first two principal components; returns ``(coords, variance)``.

    PCA finds the two directions along which customers differ most and flattens
    everything onto that plane. Some information is always lost — check the
    reported variance share before trusting what you see. Two clusters that
    overlap in the projection may be perfectly separate in the full space.
    """
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X), pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Describing the result
# ---------------------------------------------------------------------------


def profile_clusters(
    customers: pd.DataFrame, labels: np.ndarray, features: list[str]
) -> pd.DataFrame:
    """Average each feature within each cluster — the segment profiles.

    Returns a DataFrame rather than printing, so the same function serves a
    notebook, a report and a Streamlit page.
    """
    profile = customers[features].copy()
    profile["Cluster"] = labels
    summary = profile.groupby("Cluster").mean().round(1)
    summary.insert(0, "Customers", profile.groupby("Cluster").size())
    return summary


def cluster_fingerprints(
    customers: pd.DataFrame, profiles: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """How far each cluster sits from the average customer, in standard deviations.

    The same z-scores the algorithm itself saw, which is what makes a cluster
    describable: "+2 on campaign response, -1 on children" is a segment you can
    name, where a table of raw means is just a table.
    """
    overall = customers[features].mean()
    spread = customers[features].std()
    return ((profiles[features] - overall) / spread).round(2)


def name_segment(fingerprint: pd.Series, threshold: float = 0.6) -> str:
    """Turn one cluster's z-scores into a readable description."""
    high = ", ".join(f"high {f}" for f in fingerprint[fingerprint > threshold].index)
    low = ", ".join(f"low {f}" for f in fingerprint[fingerprint < -threshold].index)
    return " / ".join(part for part in (high, low) if part) or "close to average on everything"
