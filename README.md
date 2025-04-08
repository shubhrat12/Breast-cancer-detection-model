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
3. **Model Training** - Implementing multiple classification algorithms:
   - Logistic Regression
   - Support Vector Machines
   - K-Nearest Neighbors
   - Random Forest
   - Decision Tree
   - Gradient Boosting
4. **Hyperparameter Tuning** - Using grid search and randomized search
5. **Model Evaluation** - Comparing models based on accuracy scores

## Results
The best performing models were Logistic Regression and Linear SVM, both achieving 99.3% accuracy on the validation set. The dimensionality reduction with PCA preserved approximately 84% of the variance while reducing to just 5 principal components.

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
