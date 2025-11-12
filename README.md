# ML Statistical Models - Non-Linear Bases, Bias & Variance, Regularization

This repository contains an experiment performed with Linear and Logistic Regression models to explore the use of Non-Linear basis functions to learn non-linear patterns data. The use of Gaussian bases different in values and quantities is explored, with bias and variance evaluations and visualizations. This experiment also visualizes the effects of different regularization techniques, specifically Lasso (L1) and Ridge (L2) regression.

The models and relevant classes are included in `Models.py`, with various tunable hyperparameters passed through initializer arguments. 

The experiments, their results, and their visualizations are contained in `Code.ipynb`.

To run this code and experiment with different values yourself, follow the below steps:
1. Create a python virtual environment with `python3 -m venv .venv`
2. Activate your virtual environment: `source .venv/bin/activate`
3. Install the dependencies: `pip install -r requirements.txt`
