# %% [markdown]
# # Customer Segmentation Part 1: RFM and Customer Lifetime Value Analysis
# 
# ## Introduction
# In this notebook, we'll explore two traditional customer segmentation approaches:
# 1. **RFM Segmentation**: Based on Recency, Frequency, and Monetary value
# 2. **CLV Segmentation**: Based on Customer Lifetime Value
#
# These methods provide intuitive business insights and don't require machine learning
# but are foundational to understanding customer segments.

# %% [markdown]
# ## Section 1: Data Loading and Exploration

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('customer_segmentation.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# %% [markdown]
# ## Section 2: Data Preparation

# %%
# Create a copy for processing
df_clean = df.copy()

# Convert Dt_Customer to datetime
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'])

# Calculate customer age from birth year
reference_date = df_clean['Dt_Customer'].max()
df_clean['Age'] = (reference_date.year - df_clean['Year_Birth']).astype(int)

# Calculate total spending across all product categories
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_clean['Total_Spending'] = df_clean[spending_cols].sum(axis=1)

# Calculate customer tenure (days since they became a customer)
df_clean['Tenure_Days'] = (reference_date - df_clean['Dt_Customer']).dt.days

# Calculate total number of purchases
purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df_clean['Total_Purchases'] = df_clean[purchase_cols].sum(axis=1)

# Calculate campaign acceptance rate
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_clean['Campaign_Acceptances'] = df_clean[campaign_cols].sum(axis=1)

# Handle missing income values
df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())

print("Data preparation completed!")
print(f"\nNew features created:")
print(f"- Age: {df_clean['Age'].min()} to {df_clean['Age'].max()}")
print(f"- Tenure (days): {df_clean['Tenure_Days'].min()} to {df_clean['Tenure_Days'].max()}")
print(f"- Total Spending: ${df_clean['Total_Spending'].min():.0f} to ${df_clean['Total_Spending'].max():.0f}")
print(f"- Total Purchases: {df_clean['Total_Purchases'].min()} to {df_clean['Total_Purchases'].max()}")

# %% [markdown]
# ## Section 3: RFM (Recency, Frequency, Monetary) Segmentation

# %% [markdown]
# ### 3.1 Understanding RFM
# - **Recency (R)**: Days since last purchase (lower is better - more recent)
# - **Frequency (F)**: Number of purchases made (higher is better)
# - **Monetary (M)**: Total amount spent (higher is better)
#
# We'll score each dimension and combine them to create segments.

# %%
# Create RFM dataframe
rfm_df = pd.DataFrame({
    'Id': df_clean['Id'],
    'Recency': df_clean['Recency'],
    'Frequency': df_clean['Total_Purchases'],
    'Monetary': df_clean['Total_Spending']
})

# Calculate RFM scores (1-5 scale for each dimension)
# For Recency: lower values get higher scores (more recent is better)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
# For Frequency and Monetary: higher values get higher scores
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, 
                             labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, 
                             labels=[1, 2, 3, 4, 5], duplicates='drop')

# Convert scores to numeric
rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)

# Calculate RFM Score (combine all three)
rfm_df['RFM_Score'] = rfm_df['R_Score'] * 100 + rfm_df['F_Score'] * 10 + rfm_df['M_Score']

# Create RFM Segments based on business logic
def rfm_segment(r, f, m):
    """
    Classify customers into segments based on RFM scores
    """
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2 and m >= 3:
        return 'Potential Loyalists'
    elif r >= 4 and f <= 1:
        return 'Recent Customers'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'At Risk'
    elif r <= 2 and f >= 3:
        return 'Cant Lose Them'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Lost'
    else:
        return 'Hibernating'

rfm_df['Segment'] = rfm_df.apply(
    lambda x: rfm_segment(x['R_Score'], x['F_Score'], x['M_Score']), 
    axis=1
)

print("RFM Segmentation Summary:")
print(rfm_df['Segment'].value_counts())
print("\nRFM Score Examples:")
print(rfm_df[['Id', 'Recency', 'Frequency', 'Monetary', 
              'R_Score', 'F_Score', 'M_Score', 'Segment']].head(10))

# %% [markdown]
# ### 3.2 RFM Segment Characteristics

# %%
# Analyze each segment
segment_analysis = rfm_df.groupby('Segment').agg({
    'Recency': ['mean', 'min', 'max'],
    'Frequency': ['mean', 'min', 'max'],
    'Monetary': ['mean', 'min', 'max'],
    'Id': 'count'  # Count of customers in each segment
}).round(2)

segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns.values]
segment_analysis = segment_analysis.rename(columns={'Id_count': 'Count'})
print("\nDetailed Segment Analysis:")
print(segment_analysis)

# %% [markdown]
# ### 3.3 RFM Visualization

# %%
# Create 3D scatter plot of RFM
fig = go.Figure(data=[go.Scatter3d(
    x=rfm_df['Recency'],
    y=rfm_df['Frequency'],
    z=rfm_df['Monetary'],
    mode='markers',
    marker=dict(
        size=5,
        color=rfm_df['R_Score'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Recency Score')
    ),
    text=rfm_df['Segment'],
    hovertemplate='<b>%{text}</b><br>Recency: %{x}<br>Frequency: %{y}<br>Monetary: $%{z}<extra></extra>'
)])

fig.update_layout(
    title='3D RFM Segmentation',
    scene=dict(
        xaxis_title='Recency (days)',
        yaxis_title='Frequency (# purchases)',
        zaxis_title='Monetary ($)',
    ),
    height=700,
    width=1000
)
fig.show()

# %%
# Segment distribution
segment_counts = rfm_df['Segment'].value_counts().sort_values(ascending=True)
fig = px.barh(
    x=segment_counts.values,
    y=segment_counts.index,
    labels={'x': 'Number of Customers', 'y': 'Segment'},
    title='Customer Distribution Across RFM Segments',
    color=segment_counts.values,
    color_continuous_scale='Blues'
)
fig.update_layout(height=500, width=800)
fig.show()

# %%
# Heatmap of average metrics by segment
segment_metrics = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
fig = px.imshow(
    segment_metrics.T,
    labels=dict(x="Segment", y="Metric", color="Average Value"),
    title="Average RFM Metrics by Segment",
    color_continuous_scale='RdYlGn_r',
    aspect='auto'
)
fig.update_layout(height=400, width=900)
fig.show()

# %%
# Box plots comparing segments
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Recency by Segment", "Frequency by Segment", "Monetary by Segment"),
    specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]]
)

for col, col_name in enumerate(['Recency', 'Frequency', 'Monetary']):
    fig.add_trace(
        go.Box(x=rfm_df['Segment'], y=rfm_df[col_name], name=col_name, 
               showlegend=False),
        row=1, col=col+1
    )

fig.update_layout(height=500, width=1200, title_text="RFM Metrics Distribution Across Segments")
fig.show()

# %% [markdown]
# ## Section 4: Customer Lifetime Value (CLV) Segmentation

# %% [markdown]
# ### 4.1 Understanding CLV
# Customer Lifetime Value represents the total revenue a customer generates over their
# relationship with the company. We'll calculate CLV and segment customers based on it.

# %%
# Calculate Customer Lifetime Value (CLV)
# CLV = (Total Spending) * (Campaign Acceptance Rate) / (1 + Churn Risk)
# where Churn Risk is estimated from recency

# Calculate churn risk (higher recency = higher churn risk)
df_clean['Churn_Risk'] = df_clean['Recency'] / df_clean['Recency'].max()

# Calculate campaign response rate
df_clean['Campaign_Response_Rate'] = df_clean['Campaign_Acceptances'] / 6.0

# Calculate CLV
# Formula: Total Spending × Campaign Response Rate × (1 - Churn Risk) × Loyalty Factor
df_clean['CLV'] = (df_clean['Total_Spending'] * 
                   (1 + df_clean['Campaign_Response_Rate']) * 
                   (1 - df_clean['Churn_Risk']))

print("CLV Statistics:")
print(df_clean['CLV'].describe())
print(f"\nCLV Range: ${df_clean['CLV'].min():.2f} to ${df_clean['CLV'].max():.2f}")

# %% [markdown]
# ### 4.2 Creating CLV Segments

# %%
# Create CLV-based segments using quantiles
clv_df = pd.DataFrame({
    'Id': df_clean['Id'],
    'CLV': df_clean['CLV'],
    'Total_Spending': df_clean['Total_Spending'],
    'Campaign_Acceptances': df_clean['Campaign_Acceptances'],
    'Recency': df_clean['Recency'],
    'Total_Purchases': df_clean['Total_Purchases'],
    'Income': df_clean['Income']
})

# Segment based on CLV quartiles
clv_df['CLV_Segment'] = pd.qcut(clv_df['CLV'], 4, 
                                 labels=['Low Value', 'Medium Value', 'High Value', 'Premium'],
                                 duplicates='drop')

print("CLV Segment Distribution:")
print(clv_df['CLV_Segment'].value_counts().sort_index())
print("\nCLV Segment Boundaries:")
quartiles = pd.qcut(clv_df['CLV'], 4, retbins=True, duplicates='drop')
for i, (start, end) in enumerate(zip(quartiles[1][:-1], quartiles[1][1:])):
    print(f"Segment {i+1}: ${start:.2f} - ${end:.2f}")

# %% [markdown]
# ### 4.3 CLV Segment Analysis

# %%
# Detailed analysis of CLV segments
clv_analysis = clv_df.groupby('CLV_Segment').agg({
    'CLV': ['mean', 'median', 'std', 'min', 'max'],
    'Total_Spending': ['mean', 'min', 'max'],
    'Campaign_Acceptances': 'mean',
    'Recency': 'mean',
    'Total_Purchases': 'mean',
    'Income': 'mean',
    'Id': 'count'
}).round(2)

clv_analysis.columns = ['_'.join(col).strip() for col in clv_analysis.columns.values]
clv_analysis = clv_analysis.rename(columns={'Id_count': 'Customer_Count'})
print("\nDetailed CLV Segment Analysis:")
print(clv_analysis)

# %% [markdown]
# ### 4.4 CLV Visualization

# %%
# Distribution of CLV across segments
fig = px.box(
    clv_df,
    x='CLV_Segment',
    y='CLV',
    color='CLV_Segment',
    title='Customer Lifetime Value Distribution by Segment',
    labels={'CLV': 'CLV ($)', 'CLV_Segment': 'Segment'},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(height=500, width=900, showlegend=False)
fig.show()

# %%
# Scatter plot: CLV vs Total Spending colored by segment
fig = px.scatter(
    clv_df,
    x='Total_Spending',
    y='CLV',
    color='CLV_Segment',
    size='Campaign_Acceptances',
    hover_data=['Recency', 'Total_Purchases'],
    title='CLV vs Total Spending (bubble size = Campaign Acceptances)',
    labels={'Total_Spending': 'Total Spending ($)', 'CLV': 'Customer Lifetime Value ($)'},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(height=600, width=1000)
fig.show()

# %%
# Radar chart comparing segment characteristics
avg_metrics = clv_df.groupby('CLV_Segment')[
    ['Total_Spending', 'Campaign_Acceptances', 'Total_Purchases', 'Income']
].mean()

# Normalize metrics for better visualization (0-100 scale)
avg_metrics_normalized = avg_metrics.div(avg_metrics.max()) * 100

fig = go.Figure()

for segment in avg_metrics_normalized.index:
    fig.add_trace(go.Scatterpolar(
        r=avg_metrics_normalized.loc[segment].values,
        theta=['Total Spending', 'Campaign Acceptances', 'Purchases', 'Income'],
        fill='toself',
        name=segment
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=True,
    title='CLV Segment Characteristics (Normalized)',
    height=600,
    width=800
)
fig.show()

# %%
# Segment size pie chart
segment_sizes = clv_df['CLV_Segment'].value_counts()
fig = px.pie(
    values=segment_sizes.values,
    names=segment_sizes.index,
    title='Customer Distribution Across CLV Segments',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(height=500, width=800)
fig.show()

# %%
# Comparison of multiple metrics across CLV segments
metrics_comparison = clv_df.groupby('CLV_Segment')[
    ['CLV', 'Total_Spending', 'Campaign_Acceptances', 'Recency']
].mean()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=tuple(metrics_comparison.columns),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

for idx, col in enumerate(metrics_comparison.columns):
    row = (idx // 2) + 1
    col_num = (idx % 2) + 1
    
    fig.add_trace(
        go.Bar(x=metrics_comparison.index, y=metrics_comparison[col], 
               name=col, showlegend=False),
        row=row, col=col_num
    )

fig.update_layout(height=700, width=1000, title_text="Key Metrics Comparison Across CLV Segments")
fig.show()

# %% [markdown]
# ## Section 5: Comparison and Business Insights

# %%
# Merge RFM and CLV results for comparison
comparison_df = rfm_df[['Id', 'Segment']].merge(
    clv_df[['Id', 'CLV_Segment']], on='Id'
)

# Cross-tabulation
cross_tab = pd.crosstab(comparison_df['Segment'], comparison_df['CLV_Segment'])
print("Cross-tabulation: RFM Segments vs CLV Segments")
print(cross_tab)

# %%
# Visualization of segment overlap
fig = px.density_heatmap(
    comparison_df,
    x='Segment',
    y='CLV_Segment',
    nbinsx=20,
    nbinsy=20,
    title='Overlap between RFM and CLV Segmentation',
    labels={'Segment': 'RFM Segment', 'CLV_Segment': 'CLV Segment'},
    color_continuous_scale='YlOrRd'
)
fig.update_layout(height=600, width=1000)
fig.show()

# %% [markdown]
# ## Section 6: Key Takeaways

# %% [markdown]
# ### Summary of Findings:
# 
# 1. **RFM Segmentation** provides a simple, interpretable way to segment customers
#    based on their transaction behavior.
#
# 2. **CLV Segmentation** considers the long-term value of customers including
#    their engagement and loyalty potential.
#
# 3. **Champions and Premium Customers** should receive personalized attention
#    and loyalty programs.
#
# 4. **At-Risk and Low-Value Customers** may benefit from re-engagement campaigns
#    or targeted retention efforts.
#
# 5. **These methods are complementary** - using both provides a more complete
#    understanding of your customer base for targeted marketing strategies.
