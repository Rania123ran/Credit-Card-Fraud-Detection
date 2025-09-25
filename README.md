# Credit-Card-Fraud-Detection

Detect fraudulent card transactions with **Python / scikit-learn** using two models:
**Logistic Regression** (with feature scaling) and **Random Forest** (no scaling).

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-4B8BBE" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest-228B22" />
  <img src="https://img.shields.io/badge/Data-Imbalanced-critical" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white" />

</p>



## Goals
- Explore fraud patterns (atypical amounts, new merchants, online purchases).
- Train and compare **Logistic Regression** and **Random Forest**.
- Produce probabilities and pick a practical decision **threshold**.
- Keep the pipeline simple, reproducible, and easy to adapt.

## Dataset
LINK : https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

Main columns:
- `distance_from_home` — distance from cardholder’s home to the transaction  
- `distance_from_last_transaction` — distance since previous transaction  
- `ratio_to_median_purchase_price` — amount / customer’s typical median amount  
- `repeat_retailer` — 1 if the merchant was used before by this customer  
- `used_chip` — 1 if EMV chip used  
- `used_pin_number` — 1 if PIN used  
- `online_order` — 1 if e-commerce transaction  
- `fraud` — **target** (1 = fraud, 0 = legitimate)

> Features are pre-engineered per customer; the dataset is anonymized (no customer ID).


## Quickstart

**Requirements**
- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` (optional), `jupyter`

**Setup**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
