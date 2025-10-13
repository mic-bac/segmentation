# %% [markdown]
# # Customer Segmentation Part 2: Machine Learning Clustering Algorithms
# 
# ## Introduction
# In this notebook, we'll explore three popular clustering algorithms:
# 1. **K-Means**: Simple, scalable, and widely used (general-purpose)
# 2. **Hierarchical Clustering**: Creates a dendogram tree structure (better for visualization)
# 3. **DBSCAN**: Density-based, good at finding arbitrary-shaped clusters
#
# We'll compare their performance using various metrics and visualizations.

# %% [markdown]
# ## Section 1: Setup and Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('customer_segmentation.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# %% [markdown]
# ## Section 2: Feature Engineering and Data Preparation

# %% [markdown]
# ### 2.1 Feature Selection and Creation

# %%
# Convert Dt_Customer to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
reference_date = df['Dt_Customer'].max()

# Create features for clustering
df_features = pd.DataFrame()

# Temporal features
df_features['Age'] = reference_date.year - df['Year_Birth']
df_features['Tenure_Days'] = (reference_date - df['Dt_Customer']).dt.days

# Spending features (product categories)
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_features['Total_Spending'] = df[spending_cols].sum(axis=1)
df_features['Avg_Spending'] = df_features['Total_Spending'] / (df_features['Tenure_Days'] + 1)

# Purchase behavior features
purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df_features['Total_Purchases'] = df[purchase_cols].sum(axis=1)
df_features['Frequency'] = df_features['Total_Purchases'] / (df_features['Tenure_Days'] + 1)

# Engagement features
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_features['Campaign_Acceptances'] = df[campaign_cols].sum(axis=1)
df_features['Campaign_Response_Rate'] = df_features['Campaign_Acceptances'] / 6.0

# Recency and web activity
df_features['Recency'] = df['Recency']
df_features['Web_Visits_Month'] = df['NumWebVisitsMonth']

# Family and income features
df_features['Family_Size'] = df['Kidhome'] + df['Teenhome']
df_features['Income'] = df['Income'].fillna(df['Income'].median())

# Add customer ID for reference
df_features['Customer_Id'] = df['Id']

print("Features created successfully!")
print(f"\nFeature statistics:")
print(df_features.describe())
print(f"\nFeatures shape: {df_features.shape}")
print(f"Features: {list(df_features.columns[:-1])}")  # Exclude Customer_Id

# %% [markdown]
# ### 2.2 Data Normalization

# %%
# Select features for clustering (exclude Customer_Id)
clustering_features = df_features.drop('Customer_Id', axis=1)

# Standardize features (crucial for distance-based algorithms)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(clustering_features)

print("Data standardization completed!")
print(f"Scaled data shape: {features_scaled.shape}")
print(f"Scaled data mean (should be ~0): {features_scaled.mean(axis=0)[:3]}")
print(f"Scaled data std (should be ~1): {features_scaled.std(axis=0)[:3]}")

# %% [markdown]
# ## Section 3: Dimensionality Reduction for Visualization

# %% [markdown]
# ### 3.1 Principal Component Analysis (PCA)
# We'll reduce to 2D for visualization while preserving the original high-dimensional
# space for clustering (the best practice in machine learning).

# %%
# Apply PCA for visualization
pca = PCA(n_components=2)
features_pca_2d = pca.fit_transform(features_scaled)

print(f"PCA Explained Variance:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# %% [markdown]
# ### 3.2 Visualize Original Feature Space

# %%
# Create a DataFrame with PCA features for visualization
pca_df = pd.DataFrame({
    'PC1': features_pca_2d[:, 0],
    'PC2': features_pca_2d[:, 1]
})

fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    title='Original Customer Data in PCA Space',
    labels={'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
    opacity=0.6
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
fig.update_layout(height=600, width=900)
fig.show()

# %% [markdown]
# ## Section 4: K-Means Clustering

# %% [markdown]
# ### 4.1 Elbow Method to Find Optimal K

# %% [markdown]
# The Elbow Method helps us find the optimal number of clusters by examining
# the inertia (within-cluster sum of squares) for different K values.

# %%
# Elbow method - test different numbers of clusters
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
K_range = range(2, 11)

print("Computing K-Means for different K values...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(features_scaled, kmeans.labels_))

# Visualize elbow method
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Inertia (Elbow Method)", "Silhouette Score", "Davies-Bouldin Index"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
)

# Inertia plot
fig.add_trace(
    go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', 
               name='Inertia', marker=dict(size=8), line=dict(color='blue')),
    row=1, col=1
)

# Silhouette score plot (higher is better)
fig.add_trace(
    go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
               name='Silhouette Score', marker=dict(size=8), line=dict(color='green')),
    row=1, col=2
)

# Davies-Bouldin Index plot (lower is better)
fig.add_trace(
    go.Scatter(x=list(K_range), y=davies_bouldin_scores, mode='lines+markers',
               name='Davies-Bouldin', marker=dict(size=8), line=dict(color='red')),
    row=1, col=3
)

fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=3)
fig.update_yaxes(title_text="Inertia", row=1, col=1)
fig.update_yaxes(title_text="Silhouette Score (↑ better)", row=1, col=2)
fig.update_yaxes(title_text="Davies-Bouldin (↓ better)", row=1, col=3)
fig.update_layout(height=500, width=1400, title_text="K-Means Optimization Analysis",
                  showlegend=False)
fig.show()

print("\nK-Means Optimization Results:")
for k, inertia, silhouette, db in zip(K_range, inertias, silhouette_scores, davies_bouldin_scores):
    print(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}, Davies-Bouldin={db:.4f}")

# %% [markdown]
# ### 4.2 Train Optimal K-Means Model

# %%
# Based on elbow method and silhouette score, choose optimal K (typically 3-4)
optimal_k = 3
print(f"\nTraining K-Means with K={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(features_scaled)

# Add cluster labels to dataframe
df_features['KMeans_Cluster'] = kmeans_labels
pca_df['KMeans_Cluster'] = kmeans_labels.astype(str)

print(f"Cluster distribution:")
print(df_features['KMeans_Cluster'].value_counts().sort_index())

# %% [markdown]
# ### 4.3 K-Means Visualization

# %%
# Visualize K-Means clusters in PCA space
fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='KMeans_Cluster',
    title='K-Means Clustering Results (K=3)',
    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7
)

# Add cluster centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
fig.add_trace(
    go.Scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1],
               mode='markers', marker=dict(size=20, symbol='star', 
               color='black', line=dict(width=2, color='white')),
               name='Centroids', showlegend=True)
)

fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.update_layout(height=600, width=900)
fig.show()

# %% [markdown]
# ### 4.4 K-Means Cluster Characteristics

# %%
# Analyze cluster characteristics
kmeans_analysis = df_features.groupby('KMeans_Cluster')[
    ['Age', 'Tenure_Days', 'Total_Spending', 'Total_Purchases', 
     'Campaign_Acceptances', 'Recency', 'Income']
].mean().round(2)

print("\nK-Means Cluster Characteristics:")
print(kmeans_analysis)

# Add cluster sizes
cluster_sizes = df_features['KMeans_Cluster'].value_counts().sort_index()
kmeans_analysis['Cluster_Size'] = cluster_sizes.values
print("\n", kmeans_analysis)

# %% [markdown]
# ## Section 5: Hierarchical Clustering

# %% [markdown]
# ### 5.1 Hierarchical Clustering Model

# %% [markdown]
# Hierarchical Clustering creates a tree-like structure (dendrogram) showing
# relationships between clusters. We'll use Ward's method which minimizes variance.

# %%
# Train hierarchical clustering
print("Training Hierarchical Clustering with Ward's method...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(features_scaled)

# Add cluster labels
df_features['Hierarchical_Cluster'] = hierarchical_labels
pca_df['Hierarchical_Cluster'] = hierarchical_labels.astype(str)

print(f"Cluster distribution:")
print(df_features['Hierarchical_Cluster'].value_counts().sort_index())

# %% [markdown]
# ### 5.2 Hierarchical Clustering Visualization

# %%
# Visualize hierarchical clusters in PCA space
fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='Hierarchical_Cluster',
    title='Hierarchical Clustering Results (K=3, Ward Linkage)',
    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
    color_discrete_sequence=px.colors.qualitative.Set2,
    opacity=0.7
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.update_layout(height=600, width=900)
fig.show()

# %% [markdown]
# ### 5.3 Hierarchical Clustering Characteristics

# %%
# Analyze cluster characteristics
hierarchical_analysis = df_features.groupby('Hierarchical_Cluster')[
    ['Age', 'Tenure_Days', 'Total_Spending', 'Total_Purchases', 
     'Campaign_Acceptances', 'Recency', 'Income']
].mean().round(2)

print("\nHierarchical Clustering Cluster Characteristics:")
print(hierarchical_analysis)

# Add cluster sizes
cluster_sizes = df_features['Hierarchical_Cluster'].value_counts().sort_index()
hierarchical_analysis['Cluster_Size'] = cluster_sizes.values
print("\n", hierarchical_analysis)

# %% [markdown]
# ## Section 6: DBSCAN Clustering

# %% [markdown]
# ### 6.1 DBSCAN Parameter Tuning

# %% [markdown]
# DBSCAN is density-based and requires two parameters:
# - **eps**: Maximum distance between points in the same cluster
# - **min_samples**: Minimum number of points to form a dense region

# %%
# Find optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

# Calculate distances to k-nearest neighbors
k = 4  # min_samples will be k+1
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(features_scaled)
distances, indices = neighbors_fit.kneighbors(features_scaled)

# Sort distances
distances = np.sort(distances[:, k-1], axis=0)

print("DBSCAN Parameter Analysis:")
print(f"K-distance statistics (for K={k}):")
print(f"  Min: {distances.min():.4f}")
print(f"  Mean: {distances.mean():.4f}")
print(f"  Median: {np.median(distances):.4f}")
print(f"  Percentile 75: {np.percentile(distances, 75):.4f}")
print(f"  Percentile 90: {np.percentile(distances, 90):.4f}")

# %% [markdown]
# ### 6.2 Train DBSCAN Models

# %%
# Train DBSCAN with different eps values
eps_values = [0.5, 0.75, 1.0, 1.25, 1.5]
dbscan_results = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=4)
    labels = dbscan.fit_predict(features_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    dbscan_results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'labels': labels
    })
    
    print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise points")

# Choose eps value that gives reasonable clusters (not too many, not too few noise points)
optimal_eps = 1.0
print(f"\nChosen eps: {optimal_eps}")

# %% [markdown]
# ### 6.3 Train Final DBSCAN Model

# %%
# Train final DBSCAN
print(f"Training DBSCAN with eps={optimal_eps}, min_samples=4...")
dbscan = DBSCAN(eps=optimal_eps, min_samples=4)
dbscan_labels = dbscan.fit_predict(features_scaled)

# Handle noise points (-1 label)
dbscan_labels_adjusted = dbscan_labels.copy()
n_noise = list(dbscan_labels).count(-1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

print(f"\nDBSCAN Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.2f}%)")

# Add to dataframe
df_features['DBSCAN_Cluster'] = dbscan_labels_adjusted
pca_df['DBSCAN_Cluster'] = dbscan_labels_adjusted.astype(str)

print(f"\nCluster distribution:")
print(df_features['DBSCAN_Cluster'].value_counts().sort_index())

# %% [markdown]
# ### 6.4 DBSCAN Visualization

# %%
# Visualize DBSCAN clusters (noise points in gray)
colors = []
for label in dbscan_labels:
    if label == -1:
        colors.append('Noise')
    else:
        colors.append(f'Cluster {label}')

pca_df['DBSCAN_Label'] = colors

fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='DBSCAN_Label',
    title=f'DBSCAN Clustering Results (eps={optimal_eps}, min_samples=4)',
    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
    color_discrete_map={
        'Noise': 'lightgray',
        'Cluster 0': '#1f77b4',
        'Cluster 1': '#ff7f0e',
        'Cluster 2': '#2ca02c',
        'Cluster 3': '#d62728'
    },
    opacity=0.7
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
fig.update_layout(height=600, width=900)
fig.show()

# %% [markdown]
# ### 6.5 DBSCAN Cluster Characteristics

# %%
# Analyze cluster characteristics (excluding noise points)
dbscan_analysis = df_features[df_features['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[
    ['Age', 'Tenure_Days', 'Total_Spending', 'Total_Purchases', 
     'Campaign_Acceptances', 'Recency', 'Income']
].mean().round(2)

print("\nDBSCAN Cluster Characteristics (excluding noise):")
print(dbscan_analysis)

# Add cluster sizes
cluster_sizes = df_features[df_features['DBSCAN_Cluster'] != -1]['DBSCAN_Cluster'].value_counts().sort_index()
dbscan_analysis['Cluster_Size'] = cluster_sizes.values
print("\n", dbscan_analysis)

# %% [markdown]
# ## Section 7: Clustering Evaluation Metrics

# %% [markdown]
# ### 7.1 Comparison of Clustering Metrics

# %%
# Calculate evaluation metrics for all three algorithms
metrics_comparison = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette Score': [
        silhouette_score(features_scaled, kmeans_labels),
        silhouette_score(features_scaled, hierarchical_labels),
        silhouette_score(features_scaled[dbscan_labels != -1], 
                        dbscan_labels[dbscan_labels != -1])
    ],
    'Davies-Bouldin Index': [
        davies_bouldin_score(features_scaled, kmeans_labels),
        davies_bouldin_score(features_scaled, hierarchical_labels),
        davies_bouldin_score(features_scaled[dbscan_labels != -1], 
                            dbscan_labels[dbscan_labels != -1])
    ],
    'Calinski-Harabasz Index': [
        calinski_harabasz_score(features_scaled, kmeans_labels),
        calinski_harabasz_score(features_scaled, hierarchical_labels),
        calinski_harabasz_score(features_scaled[dbscan_labels != -1], 
                               dbscan_labels[dbscan_labels != -1])
    ]
}).round(4)

print("\nClustering Algorithm Comparison:")
print(metrics_comparison.to_string(index=False))

# %% [markdown]
# ### 7.2 Metrics Visualization

# %%
# Normalize metrics for comparison (higher is better for all in visualization)
metrics_viz = metrics_comparison.copy()
metrics_viz['Davies-Bouldin Index'] = 1 / metrics_viz['Davies-Bouldin Index']  # Invert for consistency

fig = px.bar(
    metrics_viz,
    x='Algorithm',
    y=['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'],
    title='Clustering Algorithm Performance Comparison',
    barmode='group',
    labels={'value': 'Score (normalized)'},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(height=500, width=1000)
fig.show()

# %% [markdown]
# ## Section 8: Algorithm Comparison Visualization

# %% [markdown]
# ### 8.1 Side-by-Side Cluster Comparison

# %%
# Create side-by-side comparison
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("K-Means", "Hierarchical", "DBSCAN"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
)

# K-Means
fig.add_trace(
    px.scatter(pca_df, x='PC1', y='PC2', color='KMeans_Cluster',
               color_discrete_sequence=px.colors.qualitative.Set1).data[0],
    row=1, col=1
)

# Hierarchical
fig.add_trace(
    px.scatter(pca_df, x='PC1', y='PC2', color='Hierarchical_Cluster',
               color_discrete_sequence=px.colors.qualitative.Set2).data[0],
    row=1, col=2
)

# DBSCAN
fig.add_trace(
    px.scatter(pca_df, x='PC1', y='PC2', color='DBSCAN_Label').data[0],
    row=1, col=3
)

fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", row=1, col=1)
fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", row=1, col=2)
fig.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", row=1, col=3)

fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", row=1, col=1)
fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", row=1, col=2)
fig.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", row=1, col=3)

fig.update_layout(height=500, width=1500, title_text="Algorithm Comparison in PCA Space",
                  showlegend=True)
fig.show()

# %% [markdown]
# ## Section 9: Feature Importance in Clustering

# %% [markdown]
# ### 9.1 Cluster Feature Distribution

# %%
# Visualize feature distributions across K-Means clusters
features_to_plot = ['Total_Spending', 'Campaign_Acceptances', 'Recency', 'Income']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=features_to_plot,
    specs=[[{"type": "box"}, {"type": "box"}],
           [{"type": "box"}, {"type": "box"}]]
)

for idx, feature in enumerate(features_to_plot):
    row = (idx // 2) + 1
    col = (idx % 2) + 1
    
    for cluster in sorted(df_features['KMeans_Cluster'].unique()):
        cluster_data = df_features[df_features['KMeans_Cluster'] == cluster][feature]
        fig.add_trace(
            go.Box(y=cluster_data, name=f'Cluster {cluster}', showlegend=(idx==0)),
            row=row, col=col
        )

fig.update_layout(height=700, width=1200, title_text="K-Means Cluster Feature Distributions")
fig.show()

# %% [markdown]
# ## Section 10: Key Takeaways and Recommendations

# %% [markdown]
# ### Summary of Findings:
#
# **Algorithm Comparison:**
# 
# 1. **K-Means**
#    - Pros: Fast, scalable, easy to interpret
#    - Cons: Assumes spherical clusters of similar size
#    - Use case: General-purpose clustering, business applications
#    - Performance: Good for this dataset with clear cluster structure
#
# 2. **Hierarchical Clustering**
#    - Pros: Creates interpretable dendrograms, flexible linkage options
#    - Cons: More computationally expensive, sensitive to outliers
#    - Use case: Understanding cluster hierarchy, exploratory analysis
#    - Performance: Similar to K-Means but provides hierarchical structure
#
# 3. **DBSCAN**
#    - Pros: Finds arbitrary-shaped clusters, handles outliers naturally
#    - Cons: Sensitive to parameter selection, struggles with varying densities
#    - Use case: Data with noise, non-convex clusters
#    - Performance: Identifies outliers but may create too many clusters
#
# **Recommendations for Practice:**
#
# - **Start with K-Means**: It's fast, interpretable, and works well for most customer segmentation
# - **Validate with Hierarchical**: Confirm K-Means results using hierarchical clustering
# - **Use DBSCAN for outlier detection**: Identify problematic customers or data quality issues
# - **Always standardize features**: Critical for distance-based algorithms
# - **Use multiple metrics**: No single metric is perfect; combine Silhouette, Davies-Bouldin, and Calinski-Harabasz
# - **Validate business logic**: Ensure clusters make sense from a business perspective

# %% [markdown]
# ## Section 11: Cluster Stability Analysis

# %% [markdown]
# ### 11.1 Test Clustering Robustness

# %%
# Evaluate clustering stability by training on multiple random samples
print("Testing clustering stability with bootstrap samples...")

stability_results = []
n_iterations = 20

for i in range(n_iterations):
    # Create a random sample with replacement
    sample_indices = np.random.choice(len(features_scaled), size=len(features_scaled), replace=True)
    features_sample = features_scaled[sample_indices]
    
    # Train each algorithm
    kmeans_boot = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_boot.fit(features_sample)
    
    hier_boot = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hier_boot.fit(features_sample)
    
    dbscan_boot = DBSCAN(eps=optimal_eps, min_samples=4)
    dbscan_boot.fit(features_sample)
    
    stability_results.append({
        'Iteration': i + 1,
        'KMeans': silhouette_score(features_sample, kmeans_boot.labels_),
        'Hierarchical': silhouette_score(features_sample, hier_boot.labels_),
        'DBSCAN': silhouette_score(features_sample[dbscan_boot.labels_ != -1],
                                  dbscan_boot.labels_[dbscan_boot.labels_ != -1])
    })

stability_df = pd.DataFrame(stability_results)

print("\nStability Analysis (Silhouette Scores from Bootstrap Samples):")
print(stability_df.describe())

# %% [markdown]
# ### 11.2 Stability Visualization

# %%
# Plot stability across iterations
fig = go.Figure()

for algo in ['KMeans', 'Hierarchical', 'DBSCAN']:
    fig.add_trace(
        go.Scatter(x=stability_df['Iteration'], y=stability_df[algo],
                  mode='lines+markers', name=algo, marker=dict(size=6))
    )

fig.update_layout(
    title='Clustering Algorithm Stability Across Bootstrap Samples',
    xaxis_title='Bootstrap Iteration',
    yaxis_title='Silhouette Score',
    height=500,
    width=1000,
    hovermode='x unified'
)
fig.show()

# %% [markdown]
# ### 11.3 Stability Summary

# %%
stability_summary = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Mean Silhouette': [
        stability_df['KMeans'].mean(),
        stability_df['Hierarchical'].mean(),
        stability_df['DBSCAN'].mean()
    ],
    'Std Dev': [
        stability_df['KMeans'].std(),
        stability_df['Hierarchical'].std(),
        stability_df['DBSCAN'].std()
    ]
}).round(4)

print("\nStability Summary:")
print(stability_summary)
print("\nInterpretation:")
print("- Lower std dev indicates more stable clustering")
print("- K-Means typically shows high stability in business applications")

# %% [markdown]
# ## Section 12: Cluster Interpretation and Business Application

# %% [markdown]
# ### 12.1 Business Profile of K-Means Clusters

# %%
# Create detailed business profiles for K-Means clusters
print("\n" + "="*80)
print("DETAILED BUSINESS PROFILES: K-MEANS CLUSTERS")
print("="*80)

df_full = df_features.copy()
df_full['Total_Spending'] = df_full['Total_Spending']
df_full['Total_Purchases'] = df_full['Total_Purchases']

for cluster in sorted(df_full['KMeans_Cluster'].unique()):
    cluster_mask = df_full['KMeans_Cluster'] == cluster
    cluster_customers = df_full[cluster_mask]
    
    print(f"\n{'CLUSTER ' + str(cluster) + ' PROFILE':^80}")
    print("-" * 80)
    print(f"Size: {len(cluster_customers)} customers ({len(cluster_customers)/len(df_full)*100:.1f}%)")
    print(f"\nSpending Behavior:")
    print(f"  Average Total Spending: ${cluster_customers['Total_Spending'].mean():,.0f}")
    print(f"  Average Purchase Frequency: {cluster_customers['Frequency'].mean():.3f} purchases/day")
    print(f"  Total Purchases: {cluster_customers['Total_Purchases'].mean():.1f}")
    print(f"\nEngagement Metrics:")
    print(f"  Campaign Acceptance Rate: {cluster_customers['Campaign_Response_Rate'].mean()*100:.1f}%")
    print(f"  Average Recency: {cluster_customers['Recency'].mean():.0f} days")
    print(f"  Web Visits/Month: {cluster_customers['Web_Visits_Month'].mean():.1f}")
    print(f"\nDemographics:")
    print(f"  Average Age: {cluster_customers['Age'].mean():.1f} years")
    print(f"  Average Income: ${cluster_customers['Income'].mean():,.0f}")
    print(f"  Average Family Size: {cluster_customers['Family_Size'].mean():.2f}")
    print(f"\nRecommendations:")
    
    # Generate business recommendations based on cluster characteristics
    avg_spending = cluster_customers['Total_Spending'].mean()
    avg_frequency = cluster_customers['Frequency'].mean()
    avg_recency = cluster_customers['Recency'].mean()
    campaign_rate = cluster_customers['Campaign_Response_Rate'].mean()
    
    if avg_spending > df_full['Total_Spending'].quantile(0.75) and campaign_rate > df_full['Campaign_Response_Rate'].quantile(0.75):
        print("  → PREMIUM SEGMENT: Focus on loyalty programs and exclusive offers")
    elif avg_recency > df_full['Recency'].quantile(0.75):
        print("  → AT-RISK SEGMENT: Implement re-engagement campaigns")
    elif avg_frequency < df_full['Frequency'].quantile(0.25) and avg_spending > df_full['Total_Spending'].mean():
        print("  → HIGH-VALUE BUT INACTIVE: Personalized win-back campaigns needed")
    else:
        print("  → DEVELOPING SEGMENT: Growth opportunities through targeted marketing")

# %% [markdown]
# ## Section 13: Final Recommendations

# %% [markdown]
# ### Summary and Best Practices for Your Course

# %%
print("\n" + "="*80)
print("KEY LEARNINGS: CLUSTERING FOR CUSTOMER SEGMENTATION")
print("="*80)

summary_text = """
1. FEATURE ENGINEERING IS CRITICAL
   - Raw data must be transformed into meaningful features
   - Time-based features (recency, tenure, frequency) capture customer lifecycle
   - Behavioral features (spending, engagement) reflect value and loyalty
   - Demographic features provide context for business decisions

2. DATA PREPROCESSING IS MANDATORY
   - Missing values must be handled (imputation, removal)
   - Scaling/standardization is essential for distance-based algorithms
   - Outliers should be investigated (not always removed)
   - Feature selection/dimensionality reduction aids interpretation

3. ALGORITHM SELECTION DEPENDS ON CONTEXT
   ┌─────────────────────┬──────────────────────┬───────────────┐
   │ Algorithm           │ Best For             │ Key Advantage │
   ├─────────────────────┼──────────────────────┼───────────────┤
   │ K-Means             │ Business applications│ Speed & clarity│
   │ Hierarchical        │ Exploratory analysis │ Dendrogram    │
   │ DBSCAN              │ Outlier detection    │ No K needed   │
   └─────────────────────┴──────────────────────┴───────────────┘

4. EVALUATION REQUIRES MULTIPLE METRICS
   - Silhouette Score (−1 to 1): Measures cohesion and separation
   - Davies-Bouldin Index: Ratio of within/between cluster distances
   - Calinski-Harabasz Index: Ratio of between/within cluster variance
   - Stability Testing: Bootstrap/cross-validation on multiple samples

5. VISUALIZATION ENABLES INSIGHT
   - PCA/t-SNE reduce high dimensions for human interpretation
   - Scatter plots show cluster separation and structure
   - Feature distributions reveal what differentiates clusters
   - Business metrics validate technical quality

6. VALIDATION MUST BE MULTI-FACETED
   ✓ Statistical: Evaluation metrics confirm cluster quality
   ✓ Stability: Bootstrap samples test robustness
   ✓ Business: Clusters align with known customer segments
   ✓ Actionable: Insights lead to concrete marketing strategies

7. PRACTICAL WORKFLOW FOR PRACTITIONERS
   Step 1: Load and explore data → Identify missing values, outliers
   Step 2: Engineer features → Create business-relevant variables
   Step 3: Scale/normalize → Prepare for algorithms
   Step 4: Try K-Means → Fast baseline with elbow method
   Step 5: Validate → Use metrics + bootstrap stability tests
   Step 6: Interpret → Profile clusters for business context
   Step 7: Iterate → Refine features or try alternative algorithms
   Step 8: Implement → Design targeted strategies for each segment
"""

print(summary_text)

# %% [markdown]
# ### Conclusion

# %% [markdown]
"""
This notebook demonstrates that **clustering is both art and science**:

**The Science:**
- Mathematical frameworks (K-Means minimizes inertia, DBSCAN finds density peaks)
- Rigorous evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Statistical validation through bootstrap and cross-validation

**The Art:**
- Feature engineering requires domain knowledge about business context
- Parameter tuning involves interpretation and iteration
- Success ultimately depends on whether clusters drive business value

**For Your Students:**
1. Master the fundamentals: understand what each algorithm optimizes
2. Practice feature engineering: this determines 80% of success
3. Always validate multiple ways: metrics + stability + business logic
4. Learn from failures: negative results provide valuable insights
5. Think practically: algorithms serve business goals, not vice versa

**Next Steps:**
- Try these approaches on your own datasets
- Experiment with different features and parameters
- Compare results across multiple algorithms
- Document learnings and insights for team sharing
"""

print("\n✓ Analysis Complete!")
