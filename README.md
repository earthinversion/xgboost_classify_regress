# XGBoost Prediction and Visualization
This repository provides utilities for training, saving, and loading XGBoost models for classification and regression tasks, including data visualization using SHAP (SHapley Additive exPlanations) values. The project includes Jupyter notebooks demonstrating classification on CSV data and the Iris dataset, as well as regression on the California Housing dataset.


## Installation
```bash
conda env create -f environment.yml
conda activate xgboost_training
```

```bash
conda activate xgboost_training
```

## Dependencies
- Key libraries used in this project:
1. XGBoost
1. SHAP (for explainability)
1. Matplotlib (for plotting)
1. Scikit-Learn (for datasets and metrics)

## for osx-arm64
conda install numpy pandas scipy scikit-learn matplotlib joblib xgboost

## for osx-64
conda install numpy=1.19.2 pandas=1.0.1 scipy=1.6.1 scikit-learn=0.24.0 matplotlib=3.1.3 joblib=0.15.1 xgboost=1.3.1
