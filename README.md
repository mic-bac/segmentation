# Customer Segmentation Examples

This repository demonstrates different approaches to customer segmentation using both traditional business rules and machine learning techniques. It serves as a practical introduction to customer analytics for data science students.

## 🎯 Overview

The project implements two main approaches to customer segmentation:

1. **Traditional Business Methods** ([rfm_clv.py](rfm_clv.py)):
   - RFM (Recency, Frequency, Monetary) Analysis
   - Customer Lifetime Value (CLV) Calculation

2. **Machine Learning Approaches** ([clustering.py](clustering.py)):
   - K-Means Clustering
   - Hierarchical Clustering
   - DBSCAN (Density-Based Spatial Clustering)

## 📊 Features

- Comprehensive data preprocessing and feature engineering
- Interactive visualizations using Plotly
- Performance comparison of different clustering algorithms
- Detailed customer segment analysis and profiling
- Standardized evaluation metrics

## 🚀 Getting Started

### Prerequisites

Create a conda environment using the provided environment file:

```bash
conda env create -f conda_env.yaml
conda activate segmentation
```

### Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib

### Dataset

Place your customer data in `data/customer_segmentation.csv` with the following key features:
- Customer demographics
- Purchase history
- Campaign responses
- Product preferences

Source: [Kaggle Customer Segmentation Dataset](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering?select=customer_segmentation.csv)

## 📘 Usage

### 1. Traditional Segmentation

Run RFM and CLV analysis:
```bash
python rfm_clv.py
```

This script demonstrates:
- RFM scoring and segmentation
- Customer Lifetime Value calculation
- Segment visualization and analysis

### 2. Clustering Analysis

Run machine learning based clustering:
```bash
python clustering.py
```

This script covers:
- Data preprocessing and feature selection
- Multiple clustering algorithms
- Algorithm comparison and evaluation
- Interactive 3D visualizations of clusters

## 📊 Key Concepts

### RFM Analysis
- **Recency**: Time since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending

### CLV (Customer Lifetime Value)
- Predicts future revenue from customers
- Combines purchase history and engagement metrics
- Helps prioritize customer relationships

### Clustering
- Unsupervised learning approach
- Groups similar customers together
- Reveals natural segments in customer base

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
