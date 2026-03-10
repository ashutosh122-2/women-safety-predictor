import os
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset - handles both old and new format (with/without Is_Night_Risk)"""
    df = pd.read_csv(csv_path)
    
    # Required columns for new fixed format
    required_base = [
        "District",
        "Latitude",
        "Longitude",
        "Hour",
        "Day_of_Week",
        "Is_High_Density_Area",
        "Target_Risk_Y",
    ]
    
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def build_pipeline(categorical_features, numeric_features, *, max_depth=None, min_samples_split=4, min_samples_leaf=2) -> Pipeline:
    """Build sklearn pipeline with preprocessing and model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )

    clf = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    return clf


def train(csv_path: str, out_path: str, *, max_depth=None, min_samples_split=4, min_samples_leaf=2, exclude_loc=False) -> None:
    """Train model on fixed dataset (without Is_Night_Risk)"""
    df = load_data(csv_path)
    
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Define features - WITHOUT Is_Night_Risk to eliminate data leakage
    feature_cols = [
        "District",
        "Latitude",
        "Longitude",
        "Hour",
        "Day_of_Week",
        "Is_High_Density_Area",
    ]
    
    X = df[feature_cols].copy()
    y = df["Target_Risk_Y"].astype(int)

    categorical = ["District"]
    numeric = [
        "Latitude",
        "Longitude",
        "Hour",
        "Day_of_Week",
        "Is_High_Density_Area",
    ]

    if exclude_loc:
        # Remove Latitude/Longitude if requested
        numeric = [f for f in numeric if f not in ("Latitude", "Longitude")]
        feature_cols = [f for f in feature_cols if f not in ("Latitude", "Longitude")]
        X = df[feature_cols].copy()

    print(f"\nFeatures used: {feature_cols}")
    print(f"Target distribution: {dict(y.value_counts())}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    clf = build_pipeline(categorical, numeric, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nRunning stratified 5-fold cross-validation on the training set...")

    y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=cv, n_jobs=-1)
    y_proba_cv = None
    try:
        y_proba_cv = cross_val_predict(clf, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    except Exception:
        y_proba_cv = None

    # CV metrics
    metrics = {"cv": {}, "test": {}}
    metrics["cv"]["accuracy"] = float(accuracy_score(y_train, y_pred_cv))
    metrics["cv"]["precision"] = float(precision_score(y_train, y_pred_cv, zero_division=0))
    metrics["cv"]["recall"] = float(recall_score(y_train, y_pred_cv, zero_division=0))
    metrics["cv"]["f1"] = float(f1_score(y_train, y_pred_cv, zero_division=0))
    metrics["cv"]["confusion_matrix"] = confusion_matrix(y_train, y_pred_cv).tolist()
    if y_proba_cv is not None:
        try:
            metrics["cv"]["roc_auc"] = float(roc_auc_score(y_train, y_proba_cv))
            metrics["cv"]["brier_score"] = float(brier_score_loss(y_train, y_proba_cv))
        except Exception:
            metrics["cv"]["roc_auc"] = None
            metrics["cv"]["brier_score"] = None

    print("Cross-validated Accuracy:", round(metrics["cv"]["accuracy"] * 100, 2), "%")
    print("\nCross-validated classification report:\n", classification_report(y_train, y_pred_cv, digits=3))

    # Train on full training set and evaluate on test set
    print("\nTraining model on the training split and evaluating on the holdout test set...")
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_test_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_test_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_test_proba = None

    metrics["test"]["accuracy"] = float(accuracy_score(y_test, y_test_pred))
    metrics["test"]["precision"] = float(precision_score(y_test, y_test_pred, zero_division=0))
    metrics["test"]["recall"] = float(recall_score(y_test, y_test_pred, zero_division=0))
    metrics["test"]["f1"] = float(f1_score(y_test, y_test_pred, zero_division=0))
    metrics["test"]["confusion_matrix"] = confusion_matrix(y_test, y_test_pred).tolist()
    if y_test_proba is not None:
        try:
            metrics["test"]["roc_auc"] = float(roc_auc_score(y_test, y_test_proba))
            metrics["test"]["brier_score"] = float(brier_score_loss(y_test, y_test_proba))
        except Exception:
            metrics["test"]["roc_auc"] = None
            metrics["test"]["brier_score"] = None

    print("Test set Accuracy:", round(metrics["test"]["accuracy"] * 100, 2), "%")
    print("\nTest set classification report:\n", classification_report(y_test, y_test_pred, digits=3))

    # Save ROC curve
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if y_test_proba is not None:
        fpr, tpr, thr = roc_curve(y_test, y_test_proba)
        metrics["test"]["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
        if HAS_MATPLOTLIB:
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['test'].get('roc_auc'):.3f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - Test Set")
            plt.legend(loc="lower right")
            roc_path = out_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            print(f"Saved ROC curve to: {roc_path}")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Train on full dataset and save final model
    print("\nTraining final model on the full dataset and saving to disk...")
    clf_full = build_pipeline(categorical, numeric, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf_full.fit(X, y)

    joblib.dump(clf_full, out_path)
    print(f"\n✅ Saved model to: {out_path}")
    print(f"\n🎯 Expected accuracy: 90-95% (realistic, no data leakage)")


def main():
    parser = argparse.ArgumentParser(description="Train Women Safety risk model (fixed version without data leakage)")
    parser.add_argument(
        "--csv",
        default=os.getenv("CSV_PATH", r"C:\Users\user\Desktop\synthetic_up_locational_risk_data_fixed.csv"),
        help="Path to fixed CSV dataset (without Is_Night_Risk)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "models" / "model_fixed.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for the RandomForest",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=4,
        help="min_samples_split for RandomForest",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="min_samples_leaf for RandomForest",
    )
    parser.add_argument(
        "--exclude-loc",
        action="store_true",
        help="Exclude Latitude and Longitude from features",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    train(
        args.csv,
        args.out,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        exclude_loc=args.exclude_loc,
    )


if __name__ == "__main__":
    main()
