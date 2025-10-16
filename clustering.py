# %% 
# ==============================================================================
# CUSTOMER SEGMENTATION USING CLUSTERING ALGORITHMS
# ==============================================================================
# This notebook demonstrates customer segmentation using three popular clustering
# algorithms: K-Means, Hierarchical Clustering, and DBSCAN.
# Students will learn how to prepare data, apply different clustering techniques,
# and compare their results using various visualization and evaluation metrics.
# ==============================================================================

# %% [markdown]
# # 1. SETUP AND IMPORTS

# Import libraries for data manipulation, visualization, and clustering
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# # 2. LOAD AND EXPLORE DATA
#https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

# Load the customer segmentation dataset
df = pd.read_csv('data/customer_segmentation.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# %% [markdown]
# # 3. DATA PREPARATION

# STEP 1: Clean the data
# Handle missing values in Income column (if any)
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
df['Income'].fillna(df['Income'].median(), inplace=True)

plt_hist = px.histogram(df, x="Income")
plt_hist.show()
# Cut-off potential outliers
df['Income'] = [i if i < df['Income'].quantile(.98) else df['Income'].quantile(.99) for i in df["Income"]]

df['Year_Birth'] = [i if i < df['Year_Birth'].quantile(.98) else df['Year_Birth'].quantile(.99) for i in df["Year_Birth"]]

# Remove rows with remaining missing values
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")

# %% 
# STEP 2: Feature Engineering
# Create new features for better customer understanding

# Calculate total spending across all product categories
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[spending_cols].sum(axis=1)

# Calculate total children (kids + teens)
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Calculate total marketing campaign response
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                 'AcceptedCmp4', 'AcceptedCmp5']
df['Total_Campaigns_Accepted'] = df[campaign_cols].sum(axis=1)

# Calculate total purchases
purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

# Calculate customer engagement score (0-1)
df['Engagement_Score'] = (df['Total_Purchases'] + df['NumWebVisitsMonth']) / \
                         (df['Total_Purchases'].max() + df['NumWebVisitsMonth'].max())

# Calculate purchasing diversity (number of different channels used)
df['Purchase_Diversity'] = (df['NumWebPurchases'] > 0).astype(int) + \
                           (df['NumCatalogPurchases'] > 0).astype(int) + \
                           (df['NumStorePurchases'] > 0).astype(int)

# Estimating age
df["Age"] = int(df.Dt_Customer.max()[-4:])-df['Year_Birth']

# Meat Share
df["MeatShareSpend"] = df["MntMeatProducts"]/df["Total_Spending"]
df["WineShareSpend"] = df["MntWines"]/df["Total_Spending"]
df["FruitShareSpend"] = df["MntFruits"]/df["Total_Spending"]

print("New features created:")
print(df[['Total_Spending', 'Total_Children', 'Total_Campaigns_Accepted', 
          'Total_Purchases', 'Engagement_Score', 'Purchase_Diversity']].head())


# %% 
# STEP 3: Select features for clustering
# We focus on behavioral and spending features that best represent customer segments

clustering_features = [
    'Income',                    # Customer income (purchasing power)
    #'Total_Spending',            # Total amount spent------------
    'Recency',                   # Days since last purchase
    #'Total_Purchases',           # Frequency of purchases--------
    'Total_Campaigns_Accepted',  # Campaign response (engagement)
    'Total_Children',            # Family size
    'Engagement_Score',          # Calculated engagement metric
    'Purchase_Diversity',        # How many channels customer uses
    'MeatShareSpend',
    "WineShareSpend",
    "FruitShareSpend"
]

# Create the feature matrix for clustering
X = df[clustering_features].copy()
corr = X.select_dtypes(include="number").corr().round(2)
plt_corr = px.imshow(corr, text_auto=True)
plt_corr.show()

print(f"\nClustering features selected: {len(clustering_features)}")
print(f"Feature matrix shape: {X.shape}")
print("\nFeature statistics before scaling:")
print(X.describe())

# %% 
# STEP 4: Standardize features
# Standardization is crucial because clustering algorithms use distance metrics,
# and features with large ranges can dominate the clustering process

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=clustering_features)

print("\nFeature statistics after scaling (standardization):")
print(X_scaled.describe())
print("\nNote: All features now have mean ≈ 0 and std ≈ 1")

# %% [markdown]
# # 5. DETERMINING OPTIMAL NUMBER OF CLUSTERS

# For K-Means, we need to determine the optimal number of clusters
# We'll use the Elbow Method and Silhouette Analysis

# %% 
# Elbow Method: Try different numbers of clusters and plot inertia
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("Computing K-Means for different values of k...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")

# %% 
# Visualize the Elbow Method and Silhouette Scores
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Elbow Method (Inertia)", "Silhouette Score by K")
)

fig.add_trace(
    go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
               name='Inertia', marker=dict(size=8, color='#1f77b4')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
               name='Silhouette Score', marker=dict(size=8, color='#ff7f0e')),
    row=1, col=2
)

fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
fig.update_yaxes(title_text="Inertia", row=1, col=1)
fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

fig.update_layout(height=400, title_text="Optimal Cluster Selection")
fig.show()

# %%
# Based on the elbow method and silhouette score, we select optimal k
optimal_k = 5
print(f"\nOptimal number of clusters selected: {optimal_k}")

# %% [markdown]
# # 6. CLUSTERING ALGORITHM 1: K-MEANS CLUSTERING

# K-Means is one of the most popular clustering algorithms in practice.
# It partitions data into k clusters by minimizing within-cluster variance.
# 
# Advantages:
# - Fast and scalable
# - Easy to understand and implement
# - Works well for convex, roughly spherical clusters
#
# Disadvantages:
# - Requires specifying number of clusters beforehand
# - Sensitive to outliers
# - Assumes similar cluster sizes

print("=" * 70)
print("CLUSTERING ALGORITHM 1: K-MEANS")
print("=" * 70)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

print(f"\nK-Means Results (k={optimal_k}):")
print(f"Cluster distribution:\n{pd.Series(kmeans_labels).value_counts().sort_index()}")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, kmeans_labels):.2f}") # The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion. Higher is better
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, kmeans_labels):.3f}") # It measures the average similarity between each cluster and its most similar neighboring cluster. Lower is better (close to 0)

# %% [markdown]
# # 7. CLUSTERING ALGORITHM 2: HIERARCHICAL CLUSTERING (AGGLOMERATIVE)

# Hierarchical Agglomerative Clustering builds a hierarchy of clusters
# by recursively merging the closest clusters.
#
# Advantages:
# - Provides hierarchical structure (dendrogram)
# - More flexible than K-Means
# - Can use different linkage criteria (Ward, complete, average, single)
#
# Disadvantages:
# - Computationally more expensive
# - Cannot undo previous merges (greedy approach)
# - Results can be sensitive to noise and outliers

print("\n" + "=" * 70)
print("CLUSTERING ALGORITHM 2: HIERARCHICAL CLUSTERING (AGGLOMERATIVE)")
print("=" * 70)

hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Hierarchical_Cluster'] = hierarchical_labels

print(f"\nHierarchical Clustering Results (linkage='ward', n_clusters={optimal_k}):")
print(f"Cluster distribution:\n{pd.Series(hierarchical_labels).value_counts().sort_index()}")
print(f"Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, hierarchical_labels):.2f}") # The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion. Higher is better
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, hierarchical_labels):.3f}") # It measures the average similarity between each cluster and its most similar neighboring cluster. Lower is better (close to 0)

# %% [markdown]
# # 8. CLUSTERING ALGORITHM 3: DBSCAN (Density-Based Clustering)

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# identifies clusters as areas of high density separated by low-density regions.
#
# Advantages:
# - Discovers arbitrary cluster shapes (not just spherical)
# - Robust to outliers (assigns them as noise, label=-1)
# - Does NOT require specifying number of clusters
# - Good for spatial data
#
# Disadvantages:
# - Sensitive to eps and min_samples parameters
# - Struggles with varying cluster densities
# - May not work well with very high-dimensional data

print("\n" + "=" * 70)
print("CLUSTERING ALGORITHM 3: DBSCAN (DENSITY-BASED)")
print("=" * 70)

# For DBSCAN, we need to determine eps (neighborhood radius)
# A common heuristic is to plot the k-distance graph and look for the "elbow"
# For simplicity, we'll use a reasonable eps value

eps = 1.5
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['DBSCAN_Cluster'] = dbscan_labels

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results (eps={eps}, min_samples={min_samples}):")
print(f"Number of clusters: {n_clusters_dbscan}")
print(f"Number of noise points (outliers): {n_noise}")
print(f"Cluster distribution:")
for cluster_id in sorted(set(dbscan_labels)):
    count = sum(1 for x in dbscan_labels if x == cluster_id)
    label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"  {label}: {count} points ({count/len(dbscan_labels)*100:.1f}%)")

if n_clusters_dbscan > 1:
    print(f"Silhouette Score: {silhouette_score(X_scaled, dbscan_labels):.3f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, dbscan_labels):.2f}") # The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion. Higher is better
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, dbscan_labels):.3f}") # It measures the average similarity between each cluster and its most similar neighboring cluster. Lower is better (close to 0)
else:
    print("Note: DBSCAN found too few clusters for meaningful metrics.")

# %% [markdown]
# # 9. VISUALIZATION OF CLUSTERING RESULTS

# Create 3D scatter plots using real features (Income, Total_Spending, Recency)
# This gives us intuitive, interpretable visualizations of the clusters

# %% 
# Visualization 1: K-Means Clusters in 3D Feature Space
fig_kmeans = go.Figure(data=[go.Scatter3d(
    x=df['Income'],
    y=df['Total_Spending'],
    z=df['Recency'],
    mode='markers',
    marker=dict(
        size=4,
        color=kmeans_labels,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Cluster"),
        opacity=0.8
    ),
    text=[f"Income: ${x:,.0f}<br>Spending: ${y:,.0f}<br>Recency: {z:.0f} days<br>Cluster: {c}" 
          for x, y, z, c in zip(df['Income'], df['Total_Spending'], df['Recency'], kmeans_labels)],
    hoverinfo='text'
)])

fig_kmeans.update_layout(
    title='K-Means Clustering (3D Feature Space)',
    scene=dict(
        xaxis_title='Income ($)',
        yaxis_title='Total Spending ($)',
        zaxis_title='Recency (days)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    height=700,
    width=900
)
fig_kmeans.show()

# %% 
# Visualization 2a: Hierarchical Clustering Dendrogram
# A dendrogram is a tree diagram that shows how clusters are hierarchically merged
# together. It's very useful for understanding the hierarchical structure and
# deciding on the optimal number of clusters by looking at the "elbow"

print("\nGenerating Hierarchical Clustering Dendrogram...")

from scipy.cluster.hierarchy import dendrogram, linkage

# Compute the linkage matrix (contains info about how clusters are merged)
# Using Ward's method (same as we used in AgglomerativeClustering)
linkage_matrix = linkage(X_scaled, method='ward')

# Create dendrogram using plotly (custom implementation)
# Note: scipy's dendrogram is in matplotlib, but we'll create an interactive version

# For better visualization, we'll use a sample of data if dataset is very large
sample_size = min(50, len(X_scaled))  # Limit to 50 points for clarity
if len(X_scaled) > sample_size:
    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled.iloc[sample_indices]
    linkage_matrix = linkage(X_sample, method='ward')
    dendro_title = f'Hierarchical Clustering Dendrogram (Sample of {sample_size} customers)'
else:
    dendro_title = 'Hierarchical Clustering Dendrogram (All customers)'

# Calculate dendrogram data manually for plotly visualization
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

dendro_data = scipy_dendrogram(linkage_matrix, no_plot=True)

# Create dendrogram plot using plotly
fig_dendro = go.Figure()

# Draw the dendrogram lines
def plot_dendrogram(dendro_data):
    """Helper function to plot dendrogram from scipy data"""
    icoord = np.array(dendro_data['icoord'])
    dcoord = np.array(dendro_data['dcoord'])
    
    for i in range(len(icoord)):
        fig_dendro.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            hoverinfo='y',
            showlegend=False
        ))

plot_dendrogram(dendro_data)

# Add horizontal line at cut height (optimal number of clusters)
cut_height = linkage_matrix[-optimal_k+1, 2]  # Height where we cut for k clusters
fig_dendro.add_hline(
    y=cut_height,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Cut line for {optimal_k} clusters",
    annotation_position="right"
)

fig_dendro.update_layout(
    title=dendro_title,
    xaxis_title='Sample Index (Customer)',
    yaxis_title='Distance',
    height=600,
    width=1000,
    showlegend=False,
    hovermode='closest'
)

fig_dendro.show()

print(f"Dendrogram shows hierarchical merging of clusters using Ward linkage.")
print(f"Red dashed line indicates where we cut the tree to get {optimal_k} clusters.")
print(f"Higher distances indicate more distant clusters being merged.")

# %% 
# Visualization 2b: Hierarchical Clustering in 3D Feature Space
fig_hierarchical = go.Figure(data=[go.Scatter3d(
    x=df['Income'],
    y=df['Total_Spending'],
    z=df['Recency'],
    mode='markers',
    marker=dict(
        size=4,
        color=hierarchical_labels,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title="Cluster"),
        opacity=0.8
    ),
    text=[f"Income: ${x:,.0f}<br>Spending: ${y:,.0f}<br>Recency: {z:.0f} days<br>Cluster: {c}" 
          for x, y, z, c in zip(df['Income'], df['Total_Spending'], df['Recency'], hierarchical_labels)],
    hoverinfo='text'
)])

fig_hierarchical.update_layout(
    title='Hierarchical Clustering (3D Feature Space)',
    scene=dict(
        xaxis_title='Income ($)',
        yaxis_title='Total Spending ($)',
        zaxis_title='Recency (days)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    height=700,
    width=900
)
fig_hierarchical.show()

# %% 
# Visualization 3: DBSCAN Clustering in 3D Feature Space
# Color noise points (label -1) differently
colors = ['red' if label == -1 else label for label in dbscan_labels]

fig_dbscan = go.Figure(data=[go.Scatter3d(
    x=df['Income'],
    y=df['Total_Spending'],
    z=df['Recency'],
    mode='markers',
    marker=dict(
        size=5,
        color=dbscan_labels,
        colorscale='Inferno',
        showscale=True,
        colorbar=dict(title="Cluster/-1=Noise"),
        opacity=0.8
    ),
    text=[f"Income: ${x:,.0f}<br>Spending: ${y:,.0f}<br>Recency: {z:.0f} days<br>Label: {'Noise' if c == -1 else 'Cluster ' + str(c)}" 
          for x, y, z, c in zip(df['Income'], df['Total_Spending'], df['Recency'], dbscan_labels)],
    hoverinfo='text'
)])

fig_dbscan.update_layout(
    title='DBSCAN Clustering (3D Feature Space)',
    scene=dict(
        xaxis_title='Income ($)',
        yaxis_title='Total Spending ($)',
        zaxis_title='Recency (days)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    height=700,
    width=900
)
fig_dbscan.show()

# %% [markdown]
# # 10. ALGORITHM COMPARISON

# Create a comparison of the three algorithms using different metrics

print("\n" + "=" * 70)
print("CLUSTERING ALGORITHMS COMPARISON")
print("=" * 70)

comparison_data = {
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Number of Clusters': [optimal_k, optimal_k, n_clusters_dbscan],
    'Silhouette Score': [
        silhouette_score(X_scaled, kmeans_labels),
        silhouette_score(X_scaled, hierarchical_labels),
        silhouette_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else np.nan
    ],
    'Calinski-Harabasz Index': [
        calinski_harabasz_score(X_scaled, kmeans_labels),
        calinski_harabasz_score(X_scaled, hierarchical_labels),
        calinski_harabasz_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else np.nan
    ],
    'Davies-Bouldin Index': [
        davies_bouldin_score(X_scaled, kmeans_labels),
        davies_bouldin_score(X_scaled, hierarchical_labels),
        davies_bouldin_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else np.nan
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Visualize comparison
fig_comparison = go.Figure(data=[
    go.Bar(name='Silhouette Score', x=comparison_df['Algorithm'], 
           y=comparison_df['Silhouette Score']),
    go.Bar(name='Calinski-Harabasz (scaled)', x=comparison_df['Algorithm'],
           y=comparison_df['Calinski-Harabasz Index'] / 100),
])

fig_comparison.update_layout(
    title='Algorithm Performance Comparison',
    barmode='group',
    height=400,
    yaxis_title='Score',
    xaxis_title='Algorithm'
)
fig_comparison.show()

# %% [markdown]
# # 11. CUSTOMER SEGMENT ANALYSIS (Using K-Means Results)

# Analyze the characteristics of each cluster to understand customer segments

print("\n" + "=" * 70)
print("CUSTOMER SEGMENT PROFILES (K-Means Clustering)")
print("=" * 70)

for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster}: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"{'='*70}")
    
    print(f"\nSpending Profile:")
    print(f"  Average Income: ${cluster_data['Income'].mean():,.0f}")
    print(f"  Average Total Spending: ${cluster_data['Total_Spending'].mean():,.0f}")
    print(f"  Average Spending by Category:")
    for col in spending_cols:
        print(f"    - {col}: ${cluster_data[col].mean():,.0f}")
    
    print(f"\nPurchase Behavior:")
    print(f"  Average Total Purchases: {cluster_data['Total_Purchases'].mean():.1f}")
    print(f"  Average Web Purchases: {cluster_data['NumWebPurchases'].mean():.1f}")
    print(f"  Average Catalog Purchases: {cluster_data['NumCatalogPurchases'].mean():.1f}")
    print(f"  Average Store Purchases: {cluster_data['NumStorePurchases'].mean():.1f}")
    
    print(f"\nEngagement:")
    print(f"  Average Campaign Response: {cluster_data['Total_Campaigns_Accepted'].mean():.1f}")
    print(f"  Average Days Since Last Purchase (Recency): {cluster_data['Recency'].mean():.0f} days")
    print(f"  Average Engagement Score: {cluster_data['Engagement_Score'].mean():.2f}")
    
    print(f"\nDemographics:")
    print(f"  Average Number of Children: {cluster_data['Total_Children'].mean():.1f}")

# %% 
# Create visualizations comparing cluster characteristics

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Average Income", "Average Total Spending", 
                    "Average Recency", "Average Campaign Response")
)

clusters = sorted(df['KMeans_Cluster'].unique())
cluster_labels_str = [f"Cluster {c}" for c in clusters]

fig.add_trace(
    go.Bar(name='Income', x=cluster_labels_str,
           y=[df[df['KMeans_Cluster']==c]['Income'].mean() for c in clusters]),
    row=1, col=1
)

fig.add_trace(
    go.Bar(name='Spending', x=cluster_labels_str,
           y=[df[df['KMeans_Cluster']==c]['Total_Spending'].mean() for c in clusters]),
    row=1, col=2
)

fig.add_trace(
    go.Bar(name='Recency', x=cluster_labels_str,
           y=[df[df['KMeans_Cluster']==c]['Recency'].mean() for c in clusters]),
    row=2, col=1
)

fig.add_trace(
    go.Bar(name='Campaigns', x=cluster_labels_str,
           y=[df[df['KMeans_Cluster']==c]['Total_Campaigns_Accepted'].mean() for c in clusters]),
    row=2, col=2
)

fig.update_yaxes(title_text="Income ($)", row=1, col=1)
fig.update_yaxes(title_text="Spending ($)", row=1, col=2)
fig.update_yaxes(title_text="Days", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)

fig.update_layout(height=600, title_text="Customer Segment Characteristics")
fig.show()
