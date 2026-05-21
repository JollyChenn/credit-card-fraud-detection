# Credit Card Fraud Detection

I'm a first year Banking and Finance student at APU Malaysia, teaching myself data science on the side. This is my second fintech project. I wanted to build something that actually matters in the real world, and fraud detection is one of the most important problems in banking right now.

The dataset has 284,807 real European card transactions from 2013. Only 492 of them are fraud. That 0.17% rate is the whole challenge.

**Results:**
- ROC-AUC: 0.972
- Recall: 88.8%  caught 87 out of 98 frauds in the test set
- Only 11 slipped through

---

## Why this is harder than it looks

My first instinct was just train a model and done. But a model that predicts "legitimate" for everything still gets 99.8% accuracy. Completely useless.

The real problem is the imbalance. The model barely sees fraud during training so it never learns what fraud actually looks like. I fixed this with SMOTE  it creates synthetic fraud examples so the model has enough to learn from.

The other thing I got wrong early on: accuracy is the wrong metric. In fraud detection you care about recall. How many real frauds did you catch? Missing a fraud costs a bank around €5,000. A false alarm costs €10 to review. Missing fraud is 500x worse. The model has to reflect that.

---

## What I built

Trained and compared two models Logistic Regression as a baseline and Random Forest with 200 trees. Random Forest won with ROC-AUC 0.972.

Features I engineered:
- `Amount_log` — log-transformed amount because fraud amounts are weirdly distributed
- `Hour_sin` / `Hour_cos` — cyclical time encoding so the model understands 11pm and midnight are close together
- `Amount_scaled` — standardised amount, fit on training data only to avoid data leakage

For the threshold I swept from 0.05 to 0.95 and picked the one that maximised F2 score, which weights recall twice as heavily as precision. The default 0.5 cutoff is arbitrary and not right for fraud detection.

Added SHAP explainability because regulators require banks to justify why a transaction was flagged. You can't just say the model said so. SHAP shows exactly which features pushed each prediction toward fraud or legitimate.

---

## Files

```
fraud_detection_model.py   full training pipeline
fraud_scorer.py            terminal scorer — score any transaction live
dashboard.html             interactive results dashboard
requirements.txt
```

Generated after you run the model:
```
evaluation_plots.png
shap_summary.png
threshold_analysis.png
fraud_report.png
```

---

## How to run it

**1. Get the data**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in this folder.

**2. Install packages**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python fraud_detection_model.py
```
Takes around 2-3 minutes.

**4. View the dashboard**
```bash
python -m http.server 8000
```
Then open `http://localhost:8000/dashboard.html` in your browser.

**5. Score transactions live**
```bash
python fraud_scorer.py
```

---

## What I learned

Accuracy means nothing when your classes are imbalanced. I would have been happy with 99% before this project. Now I know that number is meaningless if the model never actually catches the minority class.

Data leakage is subtle and easy to miss. I almost fit the StandardScaler on the full dataset before splitting. That leaks test information into training and makes results look better than they are.

SHAP changed how I think about ML. A model that can't explain itself isn't deployable in a regulated industry. Banking has legal obligations around decision explainability and SHAP makes that possible.

---

## Stack

Python · scikit-learn · pandas · NumPy · SHAP · imbalanced-learn · matplotlib · Chart.js

---

Dataset from [Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Project #2 in my fintech portfolio. Project #1 was a Credit Risk PD model with Basel II expected loss calculations.
