# AML Project 4

This repository contains our Course Project 4 work on:
- non-linear classification models
- tree-based methods
- SVM with different kernels
- a Streamlit demo for presenting the results

## Main Files

- `AML_Project4.ipynb`: main notebook with the project workflow and results
- `gui2.py`: Streamlit app for demonstrating outputs interactively
- `outputs/`: exported CSV result tables
- `report/`: report materials
- `slides/`: presentation materials

## Project Summary

We study the Bank Marketing dataset and predict whether a client subscribes to a term deposit (`deposit`: yes/no).  
The project compares:
- baseline logistic regression
- non-linear models: polynomial, step-function, spline, GAM-style, and local classification
- tree-based classifiers: bagging, random forest, and boosting
- SVM models with linear, polynomial, RBF, and sigmoid kernels

## Run the Notebook

Open and run:

```bash
jupyter notebook AML_Project4.ipynb
```

or in JupyterLab:

```bash
jupyter lab AML_Project4.ipynb
```

## Run the Streamlit App

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then start the app:

```bash
streamlit run gui2.py
```


## Example Outputs

The project generates:
- model comparison tables
- feature importance results
- exported CSV summaries in `outputs/`
- presentation-ready plots such as:
  - `model_comparison.png`
  - `feature_importance.png`
  - `target_distribution.png`
  - `numeric_boxplots.png`
  - `correlation_heatmap.png`

