"""
╔══════════════════════════════════════════════════════════════════╗
║      CREDIT CARD FRAUD DETECTION MODEL — Fintech Project #2     ║
║      Production-grade ML pipeline with SHAP explainability       ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES (plain English):
──────────────────────────────────────
Banks process millions of credit card transactions daily. About
1 in every 1,000 is fraudulent. Our job: build a model that
catches fraud WITHOUT blocking legitimate customers.

This script:
  1. Loads 284,807 real credit card transactions
  2. Explores and understands the data
  3. Engineers smart features (time, amount transforms)
  4. Handles class imbalance with SMOTE
  5. Trains and compares Logistic Regression vs Random Forest
  6. Optimizes the decision threshold for business outcomes
  7. Explains every prediction with SHAP
  8. Saves model, plots, and dashboard data

HOW TO RUN:
──────────
  1. Download "creditcard.csv" from Kaggle:
     https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Place it in the same folder as this script
  3. Run: python fraud_detection_model.py
  4. Wait ~2-3 minutes for training to complete
"""

# ══════════════════════════════════════════════════════════════════
# STEP 0: IMPORT LIBRARIES
# ══════════════════════════════════════════════════════════════════

import pandas as pd              # Data tables (like Excel)
import numpy as np               # Math operations on arrays
import matplotlib                # Charts and plots
matplotlib.use("Agg")            # Non-interactive backend (no popups)
import matplotlib.pyplot as plt
import json                      # Save structured results
import joblib                    # Save/load trained models
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# scikit-learn: main ML library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, average_precision_score
)

# SMOTE: creates synthetic fraud examples to fix class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠ imblearn not installed. Install with: pip install imbalanced-learn")
    print("  Will fall back to class_weight='balanced' (still works well)\n")

# SHAP: tells us WHY each prediction was made
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed. Install with: pip install shap")
    print("  Will use built-in feature importance instead\n")


def extract_shap_values(shap_output):
    """
    Extract fraud-class SHAP values robustly across SHAP versions.
    
    Why this exists: SHAP changed its return format over versions.
    - Older SHAP: returns a list [class_0_values, class_1_values]
    - Newer SHAP (>=0.43): returns a 3D array (samples, features, classes)
    - LinearExplainer: returns a 2D array directly
    
    Without this function, the code would silently use the wrong
    values on newer SHAP versions.
    """
    if isinstance(shap_output, list):
        return shap_output[1]   # Old format: take fraud class
    if hasattr(shap_output, "ndim") and shap_output.ndim == 3:
        return shap_output[:, :, 1]  # New 3D format: take fraud class
    return shap_output           # 2D array (linear explainer)


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD THE DATA
# ══════════════════════════════════════════════════════════════════
"""
THE DATASET:
────────────
Real European credit card transactions from September 2013.
For privacy, the original features were transformed using PCA
(scrambles values but preserves patterns).

Columns:
  Time:   Seconds elapsed from the first transaction
  V1–V28: PCA-transformed features (anonymized)
  Amount: Transaction amount in euros
  Class:  0 = legitimate, 1 = fraud
"""

print("=" * 65)
print("  CREDIT CARD FRAUD DETECTION MODEL")
print("  Fintech Portfolio Project #2")
print("=" * 65)

DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    print(f"\n✗ File not found: {DATA_PATH}")
    print("\n  To get the data:")
    print("  1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Download (free Kaggle account required)")
    print("  3. Extract creditcard.csv into this folder")
    print(f"  4. Expected location: {os.path.abspath(DATA_PATH)}")
    sys.exit(1)

print("\n[1/8] Loading dataset...")
df = pd.read_csv(DATA_PATH)

total = len(df)
frauds = int(df["Class"].sum())
legit = total - frauds
fraud_pct = (frauds / total) * 100

print(f"  ✓ Loaded {total:,} transactions")
print(f"  ✓ Legitimate: {legit:,} ({100 - fraud_pct:.3f}%)")
print(f"  ✓ Fraudulent: {frauds:,} ({fraud_pct:.3f}%)")
print(f"  ✓ Imbalance: 1 fraud per {int(legit / max(frauds, 1))} legitimate")


# ══════════════════════════════════════════════════════════════════
# STEP 2: EXPLORE THE DATA
# ══════════════════════════════════════════════════════════════════
"""
WHY EXPLORE FIRST:
──────────────────
Before building a model, understand the data. Like a doctor
examining a patient before treatment.
"""

print("\n[2/8] Exploring data...")
missing = df.isnull().sum().sum()
print(f"  ✓ Missing values: {missing}")

fraud_amt = df[df["Class"] == 1]["Amount"]
legit_amt = df[df["Class"] == 0]["Amount"]
print(f"  ✓ Average legit amount: €{legit_amt.mean():.2f}")
print(f"  ✓ Average fraud amount: €{fraud_amt.mean():.2f}")
print(f"  ✓ Max amount in data:   €{df['Amount'].max():,.2f}")


# ══════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
"""
WHAT IS FEATURE ENGINEERING?
────────────────────────────
Features are the inputs to our model — the "clues" it uses to
decide if a transaction is fraud. Better features = better model.

NEW FEATURES WE CREATE:
  1. Amount_log:  Log-transformed amount (compresses extremes)
  2. Hour_sin/cos: Cyclical hour-of-day encoding
     Why sin/cos? So that 23:00 and 00:00 are mathematically
     close together — they're 1 hour apart, not 23!
  3. Amount_scaled: Standardized amount (added AFTER train/test
     split to prevent data leakage)
"""

print("\n[3/8] Engineering features...")
data = df.copy()

# Cyclical time encoding
data["Hour"] = (data["Time"] / 3600) % 24
data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24)

# Log-transform of Amount (np.log1p = log(1+x), safe for zeros)
data["Amount_log"] = np.log1p(data["Amount"])

# Build feature matrix (Amount_scaled added after split)
X = data.drop(columns=["Class", "Time", "Hour"])
y = data["Class"]

print(f"  ✓ Features ready: V1-V28, Amount, Amount_log, Hour_sin, Hour_cos")


# ══════════════════════════════════════════════════════════════════
# STEP 4: TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════
"""
WHY WE SPLIT THE DATA:
──────────────────────
Imagine studying for an exam by memorizing the answer key.
You'd score 100% on THAT test, but fail a real exam because
you never learned the material.

Same idea here:
  - Training set (80%): Model learns from this
  - Test set (20%):     We evaluate on data the model NEVER saw

stratify=y ensures both sets have the same fraud ratio.

CRITICAL — DATA LEAKAGE PREVENTION:
  We fit the scaler ONLY on training data. Fitting on the full
  dataset would let test information "leak" into training, giving
  us misleadingly good results. This is a common beginner mistake.
"""

print("\n[4/8] Splitting data (80% train / 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Now fit the scaler on training data ONLY
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train["Amount_scaled"] = scaler.fit_transform(X_train[["Amount"]]).ravel()
X_test["Amount_scaled"] = scaler.transform(X_test[["Amount"]]).ravel()

# Drop raw Amount (we have scaled and log versions)
X_train = X_train.drop(columns=["Amount"])
X_test = X_test.drop(columns=["Amount"])

feature_names = X_train.columns.tolist()

print(f"  ✓ Training: {len(X_train):,} transactions ({int(y_train.sum())} frauds)")
print(f"  ✓ Testing:  {len(X_test):,} transactions ({int(y_test.sum())} frauds)")
print(f"  ✓ Total features: {len(feature_names)}")
print(f"  ✓ Scaler fit on training data only (no data leakage)")


# ══════════════════════════════════════════════════════════════════
# STEP 5: HANDLE CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════════
"""
THE IMBALANCE PROBLEM:
──────────────────────
Training data has ~394 frauds vs ~227,451 legitimate. The model
sees so few fraud examples that it barely learns what fraud
"looks like."

SMOTE (Synthetic Minority Over-sampling):
  Like this: if you have 5 photos of a rare bird, SMOTE creates
  new synthetic photos by blending features from existing ones.
  Now you have enough examples to learn from.

  Technically: SMOTE finds each fraud's nearest fraud neighbors
  and creates new synthetic frauds along the line between them.

DESIGN CHOICE — SMOTE OR class_weight, NOT BOTH:
  Using both can cause "double-correction" — the model over-
  penalizes the majority class and creates too many false alarms.
  Pick one strategy. We use SMOTE if available; otherwise we use
  class_weight='balanced'.

IMPORTANT: SMOTE is applied ONLY to training data. Never to test.
"""

print("\n[5/8] Handling class imbalance...")

use_smote = SMOTE_AVAILABLE
class_weight_setting = None  # Default when SMOTE is used

if use_smote:
    try:
        smote = SMOTE(sampling_strategy=0.5, random_state=42, n_jobs=-1)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"  ✓ SMOTE applied to training data")
        print(f"  ✓ Before: {int(y_train.sum())} frauds / {len(y_train):,} total")
        print(f"  ✓ After:  {int(y_train_balanced.sum()):,} frauds / {len(y_train_balanced):,} total")
    except Exception as e:
        print(f"  ⚠ SMOTE failed: {e}")
        print(f"  → Falling back to class_weight='balanced'")
        X_train_balanced, y_train_balanced = X_train, y_train
        class_weight_setting = "balanced"
        use_smote = False
else:
    X_train_balanced, y_train_balanced = X_train, y_train
    class_weight_setting = "balanced"
    print("  ✓ Using class_weight='balanced'")


# ══════════════════════════════════════════════════════════════════
# STEP 6: TRAIN BOTH MODELS
# ══════════════════════════════════════════════════════════════════
"""
WHY TWO MODELS?
───────────────
Different algorithms have different strengths. Train two and
pick the winner based on test performance.

MODEL 1: LOGISTIC REGRESSION
  Draws a mathematical boundary between fraud and non-fraud
  in feature space. Simple, fast, interpretable.

MODEL 2: RANDOM FOREST
  Builds 200 decision trees — each asks yes/no questions
  ("Is V14 < -5?") and votes. Majority vote wins. Like 200
  fraud analysts independently reviewing each transaction.
"""

print("\n[6/8] Training models...")

# ── Model 1: Logistic Regression ─────────────────────────────
print("\n  Training Model 1: Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weight_setting,  # None if SMOTE; "balanced" otherwise
    random_state=42,
    C=0.1,                              # Regularization strength
    solver="saga",                      # Fast on large datasets
    n_jobs=-1
)
lr_model.fit(X_train_balanced, y_train_balanced)

lr_proba = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
lr_ap = average_precision_score(y_test, lr_proba)
print(f"  ✓ Logistic Regression ROC-AUC: {lr_auc:.4f}")
print(f"  ✓ Logistic Regression PR-AUC:  {lr_ap:.4f}")

# ── Model 2: Random Forest ───────────────────────────────────
print("\n  Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    class_weight=class_weight_setting,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)

rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_ap = average_precision_score(y_test, rf_proba)
print(f"  ✓ Random Forest ROC-AUC: {rf_auc:.4f}")
print(f"  ✓ Random Forest PR-AUC:  {rf_ap:.4f}")

# Pick the winner
if rf_auc >= lr_auc:
    best_model, best_proba = rf_model, rf_proba
    best_name, best_auc, best_ap = "Random Forest", rf_auc, rf_ap
else:
    best_model, best_proba = lr_model, lr_proba
    best_name, best_auc, best_ap = "Logistic Regression", lr_auc, lr_ap

print(f"\n  ★ Best model: {best_name} (ROC-AUC: {best_auc:.4f})")


# ══════════════════════════════════════════════════════════════════
# STEP 7: OPTIMIZE THE DECISION THRESHOLD
# ══════════════════════════════════════════════════════════════════
"""
WHAT IS A THRESHOLD?
────────────────────
The model outputs a probability (0.0 to 1.0). By default, if
probability > 0.5, we call it fraud. But 0.5 is arbitrary!

In fraud detection, we want HIGH RECALL (catch as many frauds
as possible), even at the cost of some false alarms.

COST-SENSITIVE OPTIMIZATION:
  Missing a real fraud:  ~€5,000 loss (chargeback + investigation)
  Reviewing false alarm: ~€10 cost
  → Missing fraud is 500× worse than a false alarm!

We optimize the F2 score (recall weighted 2× more than precision),
which naturally favors catching fraud over avoiding false alarms.

F-beta formula:
  F_β = (1+β²) × (precision × recall) / (β² × precision + recall)
  With β=2: F_2 = 5×PR / (4×P + R) — recall weighted heavily
"""

print("\n[7/8] Optimizing decision threshold...")

thresholds = np.arange(0.05, 0.96, 0.01)
best_f2 = 0
best_threshold = 0.5

cost_per_fraud_missed = 5000
cost_per_false_alarm = 10
threshold_results = []

for t in thresholds:
    preds = (best_proba >= t).astype(int)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0
    
    cm = confusion_matrix(y_test, preds)
    tn_, fp_, fn_, tp_ = cm.ravel()
    total_cost = (fn_ * cost_per_fraud_missed) + (fp_ * cost_per_false_alarm)
    
    threshold_results.append({
        "threshold": round(float(t), 2),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "cost": int(total_cost),
        "tp": int(tp_), "fp": int(fp_), "tn": int(tn_), "fn": int(fn_)
    })
    
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = float(t)

print(f"  ✓ Optimal threshold: {best_threshold:.2f}")
print(f"  ✓ (Default was 0.50, optimized for fraud detection)")

# Apply optimal threshold to test set
final_preds = (best_proba >= best_threshold).astype(int)
cm_final = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = [int(x) for x in cm_final.ravel()]

final_recall = recall_score(y_test, final_preds)
final_precision = precision_score(y_test, final_preds, zero_division=0)
final_f1 = f1_score(y_test, final_preds, zero_division=0)

print(f"\n  ── Results at threshold {best_threshold:.2f} ──")
print(f"  True Positives  (fraud caught):  {tp}")
print(f"  False Positives (false alarms):  {fp}")
print(f"  True Negatives  (legit cleared): {tn:,}")
print(f"  False Negatives (fraud missed):  {fn}")
print(f"  Recall:    {final_recall:.4f} ({final_recall*100:.1f}% of frauds caught)")
print(f"  Precision: {final_precision:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"  ROC-AUC:   {best_auc:.4f}")


# ══════════════════════════════════════════════════════════════════
# STEP 8: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
"""
WHY EXPLAINABILITY MATTERS:
───────────────────────────
Regulators (Bank Negara Malaysia, ECB, the Fed) require banks
to explain WHY a transaction was flagged. "The model said so"
isn't acceptable. SHAP provides that explanation.

HOW SHAP WORKS:
  For each prediction, SHAP calculates how much each feature
  contributed. Example:
    "Flagged because:
     - V14 was unusually low (-8.2) → pushed toward fraud (+0.35)
     - V4 was abnormally high (6.1) → pushed toward fraud (+0.22)
     - Amount was small (€1.29)    → slight fraud signal (+0.05)"

  Based on cooperative game theory (Shapley values) — the same
  math used to fairly divide profits among business partners.
"""

print("\n[8/8] Computing SHAP explanations...")
shap_feature_importance = {}
shap_succeeded = False

if SHAP_AVAILABLE:
    try:
        shap_sample_size = 500
        X_shap = X_test.iloc[:shap_sample_size]
        
        if best_name == "Random Forest":
            explainer = shap.TreeExplainer(best_model)
            shap_raw = explainer.shap_values(X_shap)
        else:
            # LinearExplainer is the correct choice for logistic regression
            # Use a sample of training data as the background
            background = X_train_balanced.iloc[:1000] if hasattr(X_train_balanced, 'iloc') else X_train_balanced[:1000]
            explainer = shap.LinearExplainer(best_model, background)
            shap_raw = explainer.shap_values(X_shap)
        
        sv = extract_shap_values(shap_raw)
        
        # Sanity check on dimensions
        if sv.shape != (len(X_shap), len(feature_names)):
            raise ValueError(f"SHAP shape mismatch: got {sv.shape}, "
                             f"expected ({len(X_shap)}, {len(feature_names)})")
        
        # Global feature importance = mean absolute SHAP value
        mean_shap = np.abs(sv).mean(axis=0)
        for i, feat in enumerate(feature_names):
            shap_feature_importance[feat] = round(float(mean_shap[i]), 6)
        
        shap_feature_importance = dict(
            sorted(shap_feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        top_features = list(shap_feature_importance.items())[:10]
        max_importance = max(top_features[0][1], 1e-10)  # Prevent div-by-zero
        
        print("  ✓ SHAP computation complete")
        print("\n  Top 10 Features (SHAP global importance):")
        print("  " + "─" * 45)
        for feat, imp in top_features:
            bar_len = int((imp / max_importance) * 25)
            print(f"  {feat:>18s} │ {'█' * bar_len} {imp:.4f}")
        
        # Generate SHAP plots with dark theme
        print("\n  Generating SHAP plots...")
        plt.style.use("dark_background")
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, feature_names=feature_names,
                          show=False, max_display=15)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0a0a0f")
        for ax in fig.axes:
            ax.set_facecolor("#0a0a0f")
        plt.title("SHAP Feature Importance — Fraud Detection",
                  fontsize=14, pad=20, color="white")
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight",
                    facecolor="#0a0a0f", edgecolor="none")
        plt.close()
        print("  ✓ Saved: shap_summary.png")
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_shap, feature_names=feature_names,
                          show=False, plot_type="bar", max_display=15)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0a0a0f")
        for ax in fig.axes:
            ax.set_facecolor("#0a0a0f")
        plt.title("Mean |SHAP Value| — Feature Importance",
                  fontsize=14, pad=20, color="white")
        plt.tight_layout()
        plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight",
                    facecolor="#0a0a0f", edgecolor="none")
        plt.close()
        print("  ✓ Saved: shap_bar.png")
        
        shap_succeeded = True
        
    except Exception as e:
        print(f"  ⚠ SHAP failed: {e}")
        print(f"  → Using built-in feature importance instead")

if not shap_succeeded:
    if best_name == "Random Forest":
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    
    for i, feat in enumerate(feature_names):
        shap_feature_importance[feat] = round(float(importances[i]), 6)
    
    shap_feature_importance = dict(
        sorted(shap_feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    print("  ✓ Feature importance computed (built-in, not SHAP)")


# ══════════════════════════════════════════════════════════════════
# STEP 9: EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n  Generating evaluation plots...")
plt.style.use("dark_background")

# ── Combined: ROC + PR + Confusion Matrix ────────────────────
"""
ROC CURVE:
  X-axis: False Positive Rate (false alarms ÷ all legitimate)
  Y-axis: True Positive Rate (frauds caught ÷ all frauds)
  Diagonal = random guessing. Top-left corner = perfect.
  AUC = Area Under Curve. 1.0 = perfect, 0.5 = random.

PRECISION-RECALL CURVE:
  More informative than ROC for imbalanced data.
  Precision: of flagged, % that were real fraud
  Recall:    of real frauds, % we caught

CONFUSION MATRIX:
                     Predicted
                  Legit    Fraud
  Actual Legit  [ TN  |  FP  ]    TN = correctly cleared
  Actual Fraud  [ FN  |  TP  ]    FP = false alarm
                                  FN = missed fraud (dangerous!)
                                  TP = caught fraud (goal!)
"""

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0a0a0f")

for name, proba, color in [("Logistic Regression", lr_proba, "#00d4ff"),
                            ("Random Forest", rf_proba, "#ff6b6b")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name} (AUC={auc_val:.4f})")

axes[0].plot([0, 1], [0, 1], "--", color="#444", linewidth=1, label="Random (AUC=0.5)")
axes[0].set_xlabel("False Positive Rate", color="#ccc")
axes[0].set_ylabel("True Positive Rate", color="#ccc")
axes[0].set_title("ROC Curve", fontsize=14, color="#fff", pad=15)
axes[0].legend(fontsize=9, loc="lower right")
axes[0].set_facecolor("#0a0a0f")
axes[0].grid(True, alpha=0.15)

for name, proba, color in [("Logistic Regression", lr_proba, "#00d4ff"),
                            ("Random Forest", rf_proba, "#ff6b6b")]:
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba)
    ap_val = average_precision_score(y_test, proba)
    axes[1].plot(rec_arr, prec_arr, color=color, linewidth=2,
                 label=f"{name} (AP={ap_val:.4f})")

axes[1].set_xlabel("Recall", color="#ccc")
axes[1].set_ylabel("Precision", color="#ccc")
axes[1].set_title("Precision-Recall Curve", fontsize=14, color="#fff", pad=15)
axes[1].legend(fontsize=9, loc="upper right")
axes[1].set_facecolor("#0a0a0f")
axes[1].grid(True, alpha=0.15)

cm_labels = np.array([
    [f"TN\n{cm_final[0,0]:,}", f"FP\n{cm_final[0,1]:,}"],
    [f"FN\n{cm_final[1,0]:,}", f"TP\n{cm_final[1,1]:,}"]
])
axes[2].imshow(cm_final, cmap="RdYlGn_r", aspect="auto")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm_labels[i, j], ha="center", va="center",
                     fontsize=13, color="white", fontweight="bold")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Legitimate", "Fraud"], color="#ccc")
axes[2].set_yticklabels(["Legitimate", "Fraud"], color="#ccc")
axes[2].set_xlabel("Predicted", color="#ccc")
axes[2].set_ylabel("Actual", color="#ccc")
axes[2].set_title(f"Confusion Matrix (threshold={best_threshold:.2f})",
                  fontsize=14, color="#fff", pad=15)
axes[2].set_facecolor("#0a0a0f")

plt.tight_layout(pad=3)
plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f", edgecolor="none")
plt.close()
print("  ✓ Saved: evaluation_plots.png")

# Threshold analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0a0a0f")

thresh_vals = [r["threshold"] for r in threshold_results]
ax1.plot(thresh_vals, [r["recall"] for r in threshold_results],
         color="#ff6b6b", linewidth=2, label="Recall")
ax1.plot(thresh_vals, [r["precision"] for r in threshold_results],
         color="#00d4ff", linewidth=2, label="Precision")
ax1.plot(thresh_vals, [r["f1"] for r in threshold_results],
         color="#ffd700", linewidth=2, label="F1 Score")
ax1.axvline(x=best_threshold, color="#888", linestyle="--", linewidth=1,
            label=f"Optimal ({best_threshold:.2f})")
ax1.set_xlabel("Threshold", color="#ccc")
ax1.set_ylabel("Score", color="#ccc")
ax1.set_title("Precision / Recall / F1 vs Threshold",
              fontsize=13, color="#fff", pad=15)
ax1.legend(fontsize=9)
ax1.set_facecolor("#0a0a0f")
ax1.grid(True, alpha=0.15)

ax2.plot(thresh_vals, [r["cost"] / 1000 for r in threshold_results],
         color="#ff6b6b", linewidth=2)
ax2.axvline(x=best_threshold, color="#888", linestyle="--", linewidth=1)
ax2.set_xlabel("Threshold", color="#ccc")
ax2.set_ylabel("Total Cost (€ thousands)", color="#ccc")
ax2.set_title("Business Cost vs Threshold", fontsize=13, color="#fff", pad=15)
ax2.set_facecolor("#0a0a0f")
ax2.grid(True, alpha=0.15)

plt.tight_layout(pad=3)
plt.savefig("threshold_analysis.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f", edgecolor="none")
plt.close()
print("  ✓ Saved: threshold_analysis.png")

# Amount distribution
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0f")
ax.hist(legit_amt, bins=80, alpha=0.6, color="#00d4ff",
        label="Legitimate", density=True, range=(0, 500))
ax.hist(fraud_amt, bins=80, alpha=0.7, color="#ff6b6b",
        label="Fraud", density=True, range=(0, 500))
ax.set_xlabel("Transaction Amount (€)", color="#ccc")
ax.set_ylabel("Density", color="#ccc")
ax.set_title("Transaction Amount Distribution: Fraud vs Legitimate",
             fontsize=13, color="#fff", pad=15)
ax.legend(fontsize=10)
ax.set_facecolor("#0a0a0f")
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig("amount_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#0a0a0f", edgecolor="none")
plt.close()
print("  ✓ Saved: amount_distribution.png")


# ══════════════════════════════════════════════════════════════════
# STEP 10: SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════════
print("\n  Saving model and results...")

joblib.dump(best_model, "fraud_model.pkl")
joblib.dump(scaler, "amount_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
print("  ✓ Saved: fraud_model.pkl, amount_scaler.pkl, feature_names.pkl")

# Build dashboard JSON
roc_fpr, roc_tpr, _ = roc_curve(y_test, best_proba)
step = max(1, len(roc_fpr) // 100)
roc_data = [{"fpr": round(float(roc_fpr[i]), 4),
             "tpr": round(float(roc_tpr[i]), 4)}
            for i in range(0, len(roc_fpr), step)]

pr_prec, pr_rec, _ = precision_recall_curve(y_test, best_proba)
step = max(1, len(pr_prec) // 100)
pr_data = [{"recall": round(float(pr_rec[i]), 4),
            "precision": round(float(pr_prec[i]), 4)}
           for i in range(0, len(pr_prec), step)]

fraud_scores = best_proba[y_test == 1].tolist()
legit_scores = best_proba[y_test == 0]
np.random.seed(42)
legit_sample = np.random.choice(legit_scores,
                                 size=min(2000, len(legit_scores)),
                                 replace=False).tolist()

thresh_data = [r for r in threshold_results
               if int(round(r["threshold"] * 100)) % 5 == 0]

dashboard_data = {
    "model_name": best_name,
    "dataset": {
        "total_transactions": int(total),
        "total_frauds": int(frauds),
        "total_legitimate": int(legit),
        "fraud_percentage": round(fraud_pct, 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test))
    },
    "metrics": {
        "roc_auc": round(float(best_auc), 4),
        "pr_auc": round(float(best_ap), 4),
        "recall": round(float(final_recall), 4),
        "precision": round(float(final_precision), 4),
        "f1_score": round(float(final_f1), 4),
        "threshold": round(float(best_threshold), 2)
    },
    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "comparison": {
        "logistic_regression": {
            "roc_auc": round(float(lr_auc), 4),
            "pr_auc": round(float(lr_ap), 4)
        },
        "random_forest": {
            "roc_auc": round(float(rf_auc), 4),
            "pr_auc": round(float(rf_ap), 4)
        }
    },
    "roc_curve": roc_data,
    "pr_curve": pr_data,
    "score_distribution": {
        "fraud": [round(float(s), 4) for s in fraud_scores],
        "legitimate": [round(float(s), 4) for s in legit_sample]
    },
    "shap_importance": dict(list(shap_feature_importance.items())[:15]),
    "threshold_analysis": thresh_data,
    "cost_analysis": {
        "cost_per_fraud_missed": cost_per_fraud_missed,
        "cost_per_false_alarm": cost_per_false_alarm,
        "total_cost_at_optimal": int((fn * cost_per_fraud_missed) +
                                      (fp * cost_per_false_alarm)),
        "frauds_caught_value": int(tp * cost_per_fraud_missed)
    },
    "feature_engineering": [
        "Amount_scaled (StandardScaler, fit on train only)",
        "Hour_sin, Hour_cos (cyclical time encoding)",
        "Amount_log (log-transformed amount)"
    ]
}

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)
print("  ✓ Saved: dashboard_data.json")

# Classification report
report = classification_report(y_test, final_preds,
                                target_names=["Legitimate", "Fraud"])
with open("classification_report.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  FRAUD DETECTION MODEL — CLASSIFICATION REPORT\n")
    f.write(f"  Model: {best_name}\n")
    f.write(f"  Threshold: {best_threshold:.2f}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write(f"\nROC-AUC: {best_auc:.4f}\n")
    f.write(f"PR-AUC:  {best_ap:.4f}\n")
print("  ✓ Saved: classification_report.txt")


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ✓ MODEL TRAINING COMPLETE!")
print("=" * 65)
print(f"\n  Model:      {best_name}")
print(f"  ROC-AUC:    {best_auc:.4f}")
print(f"  PR-AUC:     {best_ap:.4f}")
print(f"  Recall:     {final_recall:.4f} ({final_recall*100:.1f}% of frauds caught)")
print(f"  Precision:  {final_precision:.4f}")
print(f"  F1 Score:   {final_f1:.4f}")
print(f"  Threshold:  {best_threshold:.2f}")
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn:>7,}  (legit cleared)")
print(f"    False Positives: {fp:>7,}  (false alarms)")
print(f"    False Negatives: {fn:>7,}  (fraud missed!)")
print(f"    True Positives:  {tp:>7,}  (fraud caught)")
print(f"\n  NEXT STEPS:")
print(f"    1. python generate_linkedin_image.py  → LinkedIn post image")
print(f"    2. python fraud_scorer.py             → Interactive scorer")
print(f"    3. Open dashboard.html in your browser")
print(f"    4. Push to GitHub and post on LinkedIn!\n")
