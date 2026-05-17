"""
╔══════════════════════════════════════════════════════════════════╗
║      INTERACTIVE FRAUD SCORER — Terminal Edition                ║
║      Score transactions in real-time using the trained model    ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE:
───────────
  1. First run fraud_detection_model.py to train the model
  2. Then run: python fraud_scorer.py
  3. Choose a mode:
     [1] Score a custom transaction
     [2] Score random transactions from real data
     [3] Batch score with cost analysis
     [4] SHAP explanation deep-dive
     [5] Show model metrics

  No file editing needed — everything runs in the terminal.

WHAT THIS TEACHES YOU:
──────────────────────
  - How ML models are deployed for real-time scoring
  - How probability thresholds affect business decisions
  - How SHAP explains individual predictions to analysts
  - How banks actually use fraud detection systems
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def extract_shap_values(shap_output):
    """
    Extract fraud-class SHAP values robustly across SHAP versions.
    Same helper as in fraud_detection_model.py.
    """
    if isinstance(shap_output, list):
        return shap_output[1]
    if hasattr(shap_output, "ndim") and shap_output.ndim == 3:
        return shap_output[:, :, 1]
    return shap_output


# ══════════════════════════════════════════════════════════════════
# LOAD THE TRAINED MODEL
# ══════════════════════════════════════════════════════════════════

def load_model():
    """Load the trained model and supporting files."""
    required = ["fraud_model.pkl", "amount_scaler.pkl",
                "feature_names.pkl", "dashboard_data.json"]
    
    for f in required:
        if not os.path.exists(f):
            print(f"\n  ✗ Missing file: {f}")
            print("  → Run fraud_detection_model.py first to train the model")
            sys.exit(1)
    
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("amount_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    
    with open("dashboard_data.json", "r") as f:
        dashboard = json.load(f)
    
    threshold = dashboard["metrics"]["threshold"]
    return model, scaler, feature_names, threshold, dashboard


def load_test_data():
    """Load the original dataset for demo scoring."""
    if not os.path.exists("creditcard.csv"):
        return None
    return pd.read_csv("creditcard.csv")


# ══════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════

def prepare_transaction(raw_features, scaler, feature_names):
    """
    Convert raw transaction data into model-ready format.
    
    In a real bank, this function sits between the payment
    processor and the model — transforming raw transaction
    data into the features the model expects.
    """
    row = {}
    
    # V1-V28 are PCA features (passed directly)
    for i in range(1, 29):
        key = f"V{i}"
        row[key] = float(raw_features.get(key, 0.0))
    
    # Scale the amount using the trained scaler
    amount = float(raw_features.get("Amount", 0.0))
    row["Amount_scaled"] = float(scaler.transform([[amount]])[0][0])
    
    # Time-based features
    time_val = float(raw_features.get("Time", 0.0))
    hour = (time_val / 3600) % 24
    row["Hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    row["Hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    
    # Log amount
    row["Amount_log"] = float(np.log1p(amount))
    
    # Build DataFrame in correct feature order
    df = pd.DataFrame([row])[feature_names]
    return df


def score_transaction(model, features_df, threshold):
    """
    Score a single transaction.
    Returns: (probability, is_fraud, risk_level)
    """
    probability = float(model.predict_proba(features_df)[0][1])
    is_fraud = probability >= threshold
    
    if probability < 0.1:
        risk_level = "LOW"
    elif probability < threshold:
        risk_level = "MEDIUM"
    elif probability < 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    return probability, is_fraud, risk_level


def get_risk_display(risk_level):
    """Get colored display for risk level."""
    displays = {
        "LOW":      "🟢 LOW RISK",
        "MEDIUM":   "🟡 MEDIUM RISK",
        "HIGH":     "🔴 HIGH RISK — FLAGGED",
        "CRITICAL": "🚨 CRITICAL — BLOCKED"
    }
    return displays.get(risk_level, risk_level)


# ══════════════════════════════════════════════════════════════════
# SHAP EXPLANATION
# ══════════════════════════════════════════════════════════════════

def explain_prediction(model, features_df, feature_names, model_name, 
                        background_data=None):
    """
    Explain WHY the model made this prediction using SHAP.
    
    In a real bank, this explanation is shown to:
      - Fraud analysts reviewing flagged transactions
      - Regulators during compliance audits
      - Customer service when a legitimate customer is blocked
    """
    if not SHAP_AVAILABLE:
        print("\n  ⚠ SHAP not installed. Install with: pip install shap")
        return
    
    print("\n  Computing SHAP explanation...")
    
    try:
        # Detect model type robustly
        is_tree_model = (hasattr(model, "estimators_") or 
                         "Forest" in model_name or 
                         "Tree" in str(type(model).__name__))
        
        if is_tree_model:
            explainer = shap.TreeExplainer(model)
            shap_raw = explainer.shap_values(features_df)
        else:
            # Linear model needs background data
            if background_data is None:
                # Use the transaction itself as a minimal background
                background_data = features_df
            explainer = shap.LinearExplainer(model, background_data)
            shap_raw = explainer.shap_values(features_df)
        
        sv = extract_shap_values(shap_raw)
        shap_vals = sv[0]
        feat_vals = features_df.iloc[0]
        
        explanations = []
        for i, feat in enumerate(feature_names):
            explanations.append((feat, float(shap_vals[i]), float(feat_vals[feat])))
        
        # Sort by absolute SHAP value (most impactful first)
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\n  ┌───────────────────────────────────────────────────────┐")
        print("  │         WHY THIS PREDICTION WAS MADE (SHAP)          │")
        print("  ├───────────────────────────────────────────────────────┤")
        print(f"  │ {'Feature':<18} {'Value':>8}  {'SHAP':>9}  {'Direction':<13}│")
        print("  ├───────────────────────────────────────────────────────┤")
        
        for feat, shap_val, feat_val in explanations[:10]:
            if shap_val > 0.001:
                direction = "→ FRAUD ⬆"
            elif shap_val < -0.001:
                direction = "→ LEGIT ⬇"
            else:
                direction = "→ neutral"
            print(f"  │ {feat:<18} {feat_val:>8.3f}  {shap_val:>+9.4f}  {direction:<13}│")
        
        print("  └───────────────────────────────────────────────────────┘")
        print("\n  READING THIS:")
        print("  • Positive SHAP → pushes prediction toward FRAUD")
        print("  • Negative SHAP → pushes prediction toward LEGITIMATE")
        print("  • Larger absolute value = stronger influence")
    
    except Exception as e:
        print(f"  ⚠ SHAP explanation failed: {e}")


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE MENU SYSTEM
# ══════════════════════════════════════════════════════════════════

def print_header():
    print("\n" + "=" * 60)
    print("  🔍 FRAUD DETECTION — Interactive Scorer")
    print("  Real-time transaction screening engine")
    print("=" * 60)


def safe_float(prompt, default=0.0):
    """Get a float from user input safely."""
    val = input(prompt).strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"  ⚠ Invalid number, using {default}")
        return default


def mode_custom(model, scaler, feature_names, threshold, dashboard):
    """Mode 1: Score a custom transaction."""
    print("\n  ── Custom Transaction Scorer ──")
    print("  Enter values for a transaction to score.\n")
    
    raw = {}
    raw["Amount"] = safe_float("  Transaction Amount (€): ", 0.0)
    raw["Time"] = safe_float("  Time (seconds, Enter for 0): ", 0.0)
    
    print("\n  V1-V28 features (PCA-transformed):")
    print("  [1] Use zeros (neutral baseline)")
    print("  [2] Use random values (simulated)")
    print("  [3] Use suspicious pattern (likely fraud)")
    print("  [4] Enter each value manually")
    choice = input("\n  Choice [1-4]: ").strip()
    
    if choice == "2":
        np.random.seed()  # Fresh randomness
        for i in range(1, 29):
            raw[f"V{i}"] = float(np.random.normal(0, 1))
    elif choice == "3":
        # Patterns derived from typical fraud transactions in the data
        suspicious = {
            "V1": -3.5, "V2": 2.8, "V3": -4.2, "V4": 4.5, "V5": -2.1,
            "V6": -1.5, "V7": -5.0, "V8": 0.5, "V9": -3.0, "V10": -5.5,
            "V11": 3.5, "V12": -8.0, "V13": 0.2, "V14": -10.0, "V15": 0.3,
            "V16": -6.0, "V17": -7.5, "V18": -2.5, "V19": 1.0, "V20": 0.5,
            "V21": 0.7, "V22": -0.2, "V23": -0.1, "V24": -0.5, "V25": 0.3,
            "V26": -0.2, "V27": 1.5, "V28": 0.8
        }
        raw.update(suspicious)
    elif choice == "4":
        print("  (Enter a number for each, or press Enter for 0)")
        for i in range(1, 29):
            raw[f"V{i}"] = safe_float(f"    V{i}: ", 0.0)
    else:
        for i in range(1, 29):
            raw[f"V{i}"] = 0.0
    
    # Score it
    features_df = prepare_transaction(raw, scaler, feature_names)
    probability, is_fraud, risk_level = score_transaction(model, features_df, threshold)
    
    print("\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  Amount:      €{raw['Amount']:<10.2f}                  │")
    print(f"  │  Probability: {probability:<10.4f}                  │")
    print(f"  │  Threshold:   {threshold:<10.2f}                  │")
    print(f"  │  Decision:    {'FRAUD' if is_fraud else 'LEGITIMATE':<10s}                  │")
    print(f"  │  Risk:        {get_risk_display(risk_level):<31s}│")
    print("  └──────────────────────────────────────────────┘")
    
    if is_fraud:
        print("\n  ⚠ ACTION: Transaction would be BLOCKED for manual review")
    else:
        print("\n  ✓ ACTION: Transaction would be APPROVED")
    
    if SHAP_AVAILABLE:
        explain = input("\n  Show SHAP explanation? (y/n): ").strip().lower()
        if explain == "y":
            model_name = dashboard.get("model_name", "")
            explain_prediction(model, features_df, feature_names, model_name)


def mode_random(model, scaler, feature_names, threshold, dashboard):
    """Mode 2: Score random transactions from the dataset."""
    df = load_test_data()
    if df is None:
        print("\n  ✗ creditcard.csv not found.")
        return
    
    print("\n  ── Random Transaction Scorer ──")
    print("  Pulls real transactions and scores them.\n")
    
    count_input = input("  How many transactions? (1-20, default 5): ").strip()
    count = min(int(count_input), 20) if count_input.isdigit() else 5
    count = max(1, count)
    
    samples = df.sample(n=count)
    
    print(f"\n  {'#':<4} {'Amount':>10} {'Actual':>10} {'Pred':>10} {'Prob':>8}  {'Risk':<22} {'✓/✗'}")
    print("  " + "─" * 78)
    
    correct = 0
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        raw = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
        raw["Amount"] = row["Amount"]
        raw["Time"] = row["Time"]
        actual = int(row["Class"])
        
        features_df = prepare_transaction(raw, scaler, feature_names)
        probability, is_fraud, risk_level = score_transaction(model, features_df, threshold)
        
        predicted = 1 if is_fraud else 0
        match = "✓" if predicted == actual else "✗"
        if predicted == actual:
            correct += 1
        
        actual_str = "FRAUD" if actual == 1 else "Legit"
        pred_str = "FRAUD" if is_fraud else "Legit"
        
        print(f"  {idx:<4} €{row['Amount']:>9.2f} {actual_str:>10} {pred_str:>10} "
              f"{probability:>8.4f}  {get_risk_display(risk_level):<22} {match}")
    
    print("  " + "─" * 78)
    print(f"  Accuracy on sample: {correct}/{count} ({correct/count*100:.0f}%)")


def mode_batch(model, scaler, feature_names, threshold, dashboard):
    """Mode 3: Batch score and show summary statistics."""
    df = load_test_data()
    if df is None:
        print("\n  ✗ creditcard.csv not found.")
        return
    
    print("\n  ── Batch Scoring Mode ──")
    print("  Scoring a large sample to show aggregate performance.\n")
    
    size_input = input("  Sample size? (1000-100000, default 10000): ").strip()
    try:
        size = int(size_input) if size_input else 10000
        size = max(1000, min(size, len(df)))
    except ValueError:
        size = 10000
    
    sample = df.sample(n=size, random_state=42)
    print(f"\n  Scoring {size:,} transactions...")
    
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    
    for _, row in sample.iterrows():
        raw = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
        raw["Amount"] = row["Amount"]
        raw["Time"] = row["Time"]
        actual = int(row["Class"])
        
        features_df = prepare_transaction(raw, scaler, feature_names)
        probability, is_fraud, _ = score_transaction(model, features_df, threshold)
        predicted = 1 if is_fraud else 0
        
        if actual == 1 and predicted == 1:   results["tp"] += 1
        elif actual == 0 and predicted == 1: results["fp"] += 1
        elif actual == 1 and predicted == 0: results["fn"] += 1
        else:                                 results["tn"] += 1
    
    total_fraud = results["tp"] + results["fn"]
    total_legit = results["tn"] + results["fp"]
    
    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  BATCH SCORING RESULTS                           │")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │  Total scored:     {size:>8,}                      │")
    print(f"  │  Frauds in data:   {total_fraud:>8,}                      │")
    print(f"  │  Legitimate:       {total_legit:>8,}                      │")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │  Frauds caught:    {results['tp']:>8,} / {total_fraud}                 │")
    print(f"  │  Frauds missed:    {results['fn']:>8,} / {total_fraud}                 │")
    print(f"  │  False alarms:     {results['fp']:>8,} / {total_legit:,}            │")
    if total_fraud > 0:
        print(f"  │  Recall:           {results['tp']/total_fraud:>8.1%}                      │")
    if results['tp'] + results['fp'] > 0:
        print(f"  │  Precision:        {results['tp']/(results['tp']+results['fp']):>8.1%}                      │")
    print(f"  └──────────────────────────────────────────────────┘")
    
    cost_missed = results["fn"] * 5000
    cost_false = results["fp"] * 10
    cost_total = cost_missed + cost_false
    saved = results["tp"] * 5000
    
    print(f"\n  💰 COST ANALYSIS (€5,000/missed fraud, €10/false alarm):")
    print(f"     Fraud losses prevented:  €{saved:>10,}")
    print(f"     Cost of missed frauds:   €{cost_missed:>10,}")
    print(f"     Cost of false alarms:    €{cost_false:>10,}")
    print(f"     Net value of model:      €{saved - cost_total:>10,}")


def mode_explain(model, scaler, feature_names, threshold, dashboard):
    """Mode 4: Deep-dive SHAP explanation."""
    if not SHAP_AVAILABLE:
        print("\n  ⚠ SHAP not installed. Install with: pip install shap")
        return
    
    df = load_test_data()
    if df is None:
        print("\n  ✗ creditcard.csv not found.")
        return
    
    print("\n  ── SHAP Explanation Mode ──")
    print("  [1] Explain a random FRAUD transaction")
    print("  [2] Explain a random LEGITIMATE transaction")
    choice = input("\n  Choice [1-2]: ").strip()
    
    if choice == "1":
        fraud_rows = df[df["Class"] == 1]
        if len(fraud_rows) == 0:
            print("\n  ⚠ No fraud transactions in data.")
            return
        sample = fraud_rows.sample(1)
        print("\n  Selected a real FRAUD transaction:")
    else:
        sample = df[df["Class"] == 0].sample(1)
        print("\n  Selected a real LEGITIMATE transaction:")
    
    row = sample.iloc[0]
    raw = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
    raw["Amount"] = row["Amount"]
    raw["Time"] = row["Time"]
    actual = int(row["Class"])
    
    features_df = prepare_transaction(raw, scaler, feature_names)
    probability, is_fraud, risk_level = score_transaction(model, features_df, threshold)
    
    print(f"  Amount: €{row['Amount']:.2f}")
    print(f"  Actual: {'FRAUD' if actual == 1 else 'LEGITIMATE'}")
    print(f"  Model says: {probability:.4f} probability → {'FRAUD' if is_fraud else 'LEGITIMATE'}")
    print(f"  Risk: {get_risk_display(risk_level)}")
    
    model_name = dashboard.get("model_name", "")
    
    # For linear models, build a sensible background from the data
    background = None
    if "Forest" not in model_name:
        # Sample some real data for background
        sample_bg = df.sample(n=min(100, len(df)), random_state=42)
        bg_rows = []
        for _, r in sample_bg.iterrows():
            br = {f"V{i}": r[f"V{i}"] for i in range(1, 29)}
            br["Amount"] = r["Amount"]
            br["Time"] = r["Time"]
            bg_rows.append(prepare_transaction(br, scaler, feature_names))
        background = pd.concat(bg_rows, ignore_index=True)
    
    explain_prediction(model, features_df, feature_names, model_name, 
                       background_data=background)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    try:
        model, scaler, feature_names, threshold, dashboard = load_model()
    except Exception as e:
        print(f"\n  ✗ Failed to load model: {e}")
        sys.exit(1)
    
    print_header()
    print(f"\n  Model loaded: {dashboard['model_name']}")
    print(f"  ROC-AUC: {dashboard['metrics']['roc_auc']:.4f}")
    print(f"  Threshold: {threshold}")
    
    while True:
        print("\n  ┌─────────────────────────────────────┐")
        print("  │  MENU                               │")
        print("  ├─────────────────────────────────────┤")
        print("  │  [1] Score custom transaction       │")
        print("  │  [2] Score random transactions      │")
        print("  │  [3] Batch scoring + cost analysis  │")
        print("  │  [4] SHAP explanation deep-dive     │")
        print("  │  [5] Show model metrics             │")
        print("  │  [Q] Quit                           │")
        print("  └─────────────────────────────────────┘")
        
        choice = input("\n  Choice: ").strip().lower()
        
        try:
            if choice == "1":
                mode_custom(model, scaler, feature_names, threshold, dashboard)
            elif choice == "2":
                mode_random(model, scaler, feature_names, threshold, dashboard)
            elif choice == "3":
                mode_batch(model, scaler, feature_names, threshold, dashboard)
            elif choice == "4":
                mode_explain(model, scaler, feature_names, threshold, dashboard)
            elif choice == "5":
                m = dashboard["metrics"]
                cm = dashboard["confusion_matrix"]
                print(f"\n  ── Model Performance ──")
                print(f"  Model:     {dashboard['model_name']}")
                print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
                print(f"  PR-AUC:    {m['pr_auc']:.4f}")
                print(f"  Recall:    {m['recall']:.4f} ({m['recall']*100:.1f}%)")
                print(f"  Precision: {m['precision']:.4f}")
                print(f"  F1 Score:  {m['f1_score']:.4f}")
                print(f"  Threshold: {m['threshold']:.2f}")
                print(f"\n  Confusion Matrix:")
                print(f"    TN: {cm['tn']:,}  FP: {cm['fp']:,}")
                print(f"    FN: {cm['fn']:,}  TP: {cm['tp']:,}")
            elif choice == "q":
                print("\n  Goodbye! Stay vigilant against fraud. 🔍\n")
                break
            else:
                print("  ⚠ Invalid choice. Enter 1-5 or Q.")
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Returning to menu...")
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            print("  Returning to menu...")


if __name__ == "__main__":
    main()
