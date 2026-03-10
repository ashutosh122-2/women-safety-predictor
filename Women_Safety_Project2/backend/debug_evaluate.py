import os
from pathlib import Path
import json

import pandas as pd
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    brier_score_loss,
)

CSV_PATH = os.getenv("CSV_PATH", r"C:\Users\user\Desktop\synthetic_up_locational_risk_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", r"backend/models/model.pkl")
OUTPUT_JSON = Path("backend/metrics_debug.json")

import sklearn

print(f"Using CSV: {CSV_PATH}")
print(f"Using model: {MODEL_PATH}")
print(f"Python executable: {sys.executable}")
print(f"sklearn version: {sklearn.__version__}")

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"Loaded model: {type(model)}")

# When models are trained with a different scikit-learn version than the runtime,
# unpickling can succeed but later .predict() may fail. Provide a clear hint early.
try:
    trained_version = getattr(model, "__sklearn_version__", None)
except Exception:
    trained_version = None
if trained_version is not None and trained_version != sklearn.__version__:
    print(
        f"Warning: this model was saved with scikit-learn {trained_version} but the current runtime is {sklearn.__version__}.\n"
        f"This can cause unpickling or predict-time errors (incompatibilities).\n"
        f"Recommended fixes: run the script with the project's .venv Python (preferred),\n"
        f"or reinstall scikit-learn to the version used for training (pip install scikit-learn=={trained_version})."
    )

# load data
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
feature_cols = [
    "District",
    "Latitude",
    "Longitude",
    "Hour",
    "Day_of_Week",
    "Is_Night_Risk",
    "Is_High_Density_Area",
]
X = df[feature_cols]
y = df["Target_Risk_Y"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Test size: {len(X_test)}")

# predict
try:
    y_pred = model.predict(X_test)
except Exception as e:
    print("Error during predict:", e)
    # Common cause: sklearn version mismatch or using the wrong Python environment.
    if isinstance(e, AttributeError) and "transform" in str(e):
        print("\nThis error often means the scikit-learn versions or environment differ from the one used when the model was trained.")
        print("Try running the script with the project's virtual environment python (\.venv), or install the original sklearn version used for training.")
        print(r"Example (PowerShell): .\.venv\Scripts\Activate.ps1; python backend/debug_evaluate.py")
        print(f"Or install the training sklearn version into the current interpreter: pip install scikit-learn=={trained_version or '1.3.2'}")
    raise

y_proba = None
if hasattr(model, "predict_proba"):
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print("Could not compute predict_proba:", e)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, digits=3, output_dict=True),
}
if y_proba is not None:
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    metrics["brier_score"] = float(brier_score_loss(y_test, y_proba))

with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)

print(f"Saved debug metrics to {OUTPUT_JSON}")
print(json.dumps({k: metrics[k] for k in ['accuracy','precision','recall','f1','roc_auc'] if k in metrics}, indent=2))
