# Breast Cancer Classification

## Overview
This project applies machine learning techniques to classify breast cancer tumors as malignant or benign based on features from digitized images of breast mass. Using the Wisconsin Breast Cancer dataset, this project implements multiple classification algorithms and compares their performance.

## Dataset
The Wisconsin Breast Cancer dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image. Each record represents follow-up data for one breast cancer case.

Features include:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension

Each feature is computed for the mean, standard error, and "worst" or largest values, resulting in 30 features.

## Methodology
The project follows these key steps:
1. **Data Exploration and Visualization** - Analyzing feature distributions and correlations
2. **Dimensionality Reduction** - Using Principal Component Analysis (PCA) 
3. **Model Training** - Implementing multiple classification algorithms
4. **Hyperparameter Tuning** - Using grid search and randomized search
5. **Model Evaluation** - Comparing models based on accuracy scores

## Results

### Model Performance
| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 99.30%   |
| Linear SVM            | 99.30%   |
| Random Forest         | 99.30%   |
| Gradient Boosting     | 98.60%   |
| K-Nearest Neighbors   | 97.20%   |
| Decision Tree         | 96.50%   |

### PCA Analysis
PCA was able to reduce the dimensionality from 30 features to 5 principal components while preserving 84.35% of the variance.

| Principal Component | Variance Explained |
|---------------------|-------------------|
| PC1                 | 44.90%            |
| PC2                 | 18.47%            |
| PC3                 | 9.18%             |
| PC4                 | 6.45%             |
| PC5                 | 5.35%             |
| **Total**           | **84.35%**        |

### Hyperparameter Tuning Results

#### Logistic Regression
- Optimal parameters: `C=1, max_iter=100, penalty='l2'`
- Validation score: 0.98

#### K-Nearest Neighbors
- Optimal parameters: `leaf_size=10, n_neighbors=3`
- Validation score: 0.98

#### SVM
- Optimal parameters: `C=2, kernel='rbf'`
- Validation score: 0.99

#### Random Forest
- Optimal parameters: `max_depth=7, max_features=2, max_leaf_nodes=52`
- Validation score: 0.97

## Requirements
The project requires the following Python libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- plotly

See `requirements.txt` for specific versions.

## Usage
To run this project:
1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook Breast_Cancer_Classification.ipynb`

## Visualizations
The project includes various visualizations:
- Distribution of tumor features
- Correlation analysis
- PCA visualization
- Model performance comparison

## Future Work
- Implement neural network approaches
- Explore feature importance in more detail
- Apply more advanced ensemble methods
- Test on external datasets
