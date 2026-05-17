# Credit Card Fraud Detection

My second fintech project. I wanted to build something that actually matters — fraud costs banks billions every year, and the ML side of it is genuinely interesting because the problem is harder than it looks.

The dataset is 284,807 real European card transactions from 2013. Only 492 of them are fraud. That 0.17% rate is the whole challenge.

**Results I got:**
- ROC-AUC: **0.972**
- Recall: **88.8%** — caught 87 out of 98 frauds in the test set
- Only 11 frauds slipped through

---

## Why this problem is harder than it sounds

My first instinct was: just train a model, done. But if you do that naively, the model learns to predict "legitimate" for *everything* and still gets 99.8% accuracy. That's useless.

The real challenge is the imbalance. The model barely sees any fraud examples during training, so it never really learns what fraud looks like. I fixed this with SMOTE — it creates synthetic fraud examples by interpolating between real ones, so the model gets enough fraud data to actually learn from.

The other thing I got wrong at first: accuracy is the wrong metric entirely. In fraud detection you care about **recall** — of all the real frauds, how many did you catch? Missing a fraud costs a bank around €5,000. Flagging a legitimate transaction costs about €10 to review. So missing fraud is 500x worse than a false alarm. The model should reflect that.

---

## What I built

**Two models trained and compared:**
- Logistic Regression (fast, interpretable, decent baseline)
- Random Forest with 200 trees (winner — ROC-AUC 0.972)

**Features I engineered:**
- `Amount_log` — log-transformed transaction amount. Fraud amounts are weirdly distributed so log scale helps
- `Hour_sin` / `Hour_cos` — cyclical encoding of the hour. Without this, the model thinks 11pm and midnight are far apart
- `Amount_scaled` — standardized amount, fit on training data only (fitting on the full dataset would be data leakage)

**Threshold optimization:**
The model outputs a probability. By default everyone uses 0.5 as the cutoff but that's arbitrary. I swept all thresholds from 0.05 to 0.95 and picked the one that maximized F2 score — which weights recall twice as heavily as precision.

**SHAP explainability:**
Regulators in most countries require banks to explain why a transaction was flagged. You can't just say "the model said so." SHAP calculates how much each feature pushed a prediction toward fraud or legitimate — so you can tell a customer exactly why their transaction was blocked.

---

## Files

```
fraud_detection_model.py   main training pipeline
fraud_scorer.py            interactive terminal scorer — type in a transaction and score it live
generate_linkedin_image.py generates the project poster image
dashboard.html             web dashboard showing all results
requirements.txt
```

Generated after you run the model:
```
fraud_model.pkl
evaluation_plots.png
shap_summary.png
threshold_analysis.png
fraud_report.png
```

---

## How to run it

**1. Get the data**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (free account needed) and put it in this folder.

**2. Install packages**

```bash
pip install -r requirements.txt
```

**3. Train**

```bash
python fraud_detection_model.py
```

Takes about 2-3 minutes. Trains both models, picks the best one, runs SHAP, saves everything.

**4. View the dashboard**

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/dashboard.html` in your browser. (You need the local server because browsers block local file fetching for security reasons.)

**5. Score your own transactions**

```bash
python fraud_scorer.py
```

Four modes: score a custom transaction, score random real ones from the dataset, batch scoring with cost analysis, or a SHAP deep-dive on any single transaction.

---

## Things I learned

The accuracy paradox is real. Before this project I would have been happy with 99% accuracy. Now I know that means nothing if your model predicts the majority class for everything.

Data leakage is subtle. I almost fit my StandardScaler on the full dataset before splitting. That leaks future information into training and makes your results look better than they are. Scaler gets fit on training data only, then applied to test.

SHAP changed how I think about ML. A model that can't explain itself isn't useful in a regulated industry. Banking especially — you have a legal obligation to explain decisions to customers. SHAP makes that possible.

---

## Stack

Python · scikit-learn · pandas · NumPy · SHAP · imbalanced-learn · matplotlib · Chart.js

---

Dataset from [Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This is project #2 in my fintech portfolio — project #1 was a Credit Risk PD model with Basel II expected loss calculations.
