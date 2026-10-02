# mumbai-synth-v2/models/demographic_inferrer.py
"""
Stage 3 — Demographic Inference Model

Trains XGBoost classifiers on household survey data (labeled) to infer
socio-demographic attributes from mobility features. Trained models are
then applied to CDR users (Stage 4).

Architecture:
  - One XGBoost classifier per target attribute (multi-output via loop)
  - SMOTE applied per target on training fold only (never on validation)
  - Stratified K-Fold split at HOUSEHOLD level (hh_id), not person level
    — prevents data leakage from correlated household members
  - Calibrated probabilities saved alongside hard predictions
  - Models serialized to models/saved/ for Stage 4

Targets (6):
  age_bracket   : 0-14 / 15-24 / 25-44 / 45-59 / 60+
  employment    : employed / student / homemaker / retired / unemployed
  income_class  : low / mid / high
  vehicle_owned : none / 2w / car / both
  hh_size       : 1 / 2 / 3 / 4 / 5 / 6 / 7
  hh_income     : low / mid / high

Features used (18 — excluding mean_speed_kmh and pct_walk due to
CDR/survey distribution mismatch from walk-mode dominance in CDR):
  radius_of_gyration_km, location_entropy, home_work_dist_km,
  n_unique_wards, n_trips_per_day, mean_trip_dist_km, mean_travel_mins,
  pct_peak_am, pct_peak_pm, pct_offpeak, mode_entropy, dominant_mode,
  pct_work_trips, pct_local_train, trip_regularity,
  home_ward_income_tier, home_island_city, home_metro_dist_km

Validation metrics (Harland et al. 2012):
  F1-macro, Cohen's Kappa, per-class precision/recall
  SRMSE on marginal distributions
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import (
    f1_score, cohen_kappa_score,
    classification_report, confusion_matrix,
)
from imblearn.over_sampling   import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

SURVEY_FEAT_PATH = "../data/processed/survey_merged/survey_person_features.parquet"
MODEL_DIR        = "../models/saved"
REPORT_DIR       = "../validation/reports"

# ── Config ─────────────────────────────────────────────────────────────────────

N_FOLDS     = 5
RANDOM_SEED = 42

# Features used for training and inference.
# Derived from feature_distribution_check.py (Pan & Yang 2010 criterion):
# Only features with ratio in [0.70,1.30] AND KS<0.20 OR CohenD<0.50
# are retained. 9 of 18 features pass — all others have systematic
# CDR/survey offset caused by measurement differences (CDR travel time
# = inter-stay-point gap vs survey = actual trip duration; CDR mode
# inference dominated by walk; survey has explicit purpose labels CDR lacks).
FEATURE_COLS = [
    "radius_of_gyration_km",   # ratio=0.80  KS=0.191  d=0.400  KEEP
    "location_entropy",         # ratio=0.93  KS=0.402  d=0.219  KEEP*
    "home_work_dist_km",        # ratio=1.18  KS=0.337  d=0.204  KEEP*
    "n_unique_wards",           # ratio=1.07  KS=0.221  d=0.194  KEEP*
    "pct_peak_pm",              # ratio=1.01  KS=0.501  d=0.016  KEEP*
    "mode_entropy",             # ratio=0.85  KS=0.307  d=0.186  KEEP*
    "home_ward_income_tier",    # ratio=1.05  KS=0.062  d=0.055  KEEP
    "home_island_city",         # ratio=1.06  KS=0.009  d=0.038  KEEP
    "home_metro_dist_km",       # ratio=0.82  KS=0.074  d=0.224  KEEP
]

# Target attributes and their ordered class lists
# Order matters — LabelEncoder will be fit on these exact lists
TARGETS = {
    "age_bracket":   ["0-14","15-24","25-44","45-59","60+"],
    "employment":    ["employed","homemaker","retired","student","unemployed"],
    "income_class":  ["low","mid","high"],
    "vehicle_owned": ["none","2w","car","both"],
    "hh_size":       [1, 2, 3, 4, 5, 6, 7],
    "hh_income":     ["low","mid","high"],
}

# XGBoost base params — GPU used if available, falls back to CPU
XGB_BASE_PARAMS = {
    "n_estimators":       400,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "random_state":       RANDOM_SEED,
    "eval_metric":        "mlogloss",
    "use_label_encoder":  False,
    "verbosity":          0,
}

# SMOTE config
SMOTE_K_NEIGHBORS = 5
MIN_CLASS_SAMPLES  = 10   # classes with fewer samples skip SMOTE


# ── Helpers ────────────────────────────────────────────────────────────────────

def srmse(y_true: np.ndarray, y_pred: np.ndarray,
          classes: np.ndarray) -> float:
    """
    Standardised Root Mean Squared Error on marginal distributions.
    Harland et al. 2012 — standard validation metric for synthetic
    population methods.
    SRMSE = sqrt( mean( ((obs_i - sim_i) / obs_i)^2 ) )
    """
    true_marginal = np.array(
        [(y_true == c).mean() for c in classes], dtype=float
    )
    pred_marginal = np.array(
        [(y_pred == c).mean() for c in classes], dtype=float
    )
    mask = true_marginal > 0
    if mask.sum() == 0:
        return 0.0
    ratios = ((true_marginal[mask] - pred_marginal[mask])
              / true_marginal[mask]) ** 2
    return float(np.sqrt(ratios.mean()))


def get_xgb_device() -> dict:
    """Return GPU device params if CUDA available, else CPU."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        print("  GPU detected — using device='cuda'")
        return {"device": "cuda", "tree_method": "hist"}
    except Exception:
        print("  No GPU — using CPU")
        return {"device": "cpu",  "tree_method": "hist"}


def household_kfold_split(df: pd.DataFrame, n_folds: int, seed: int):
    """
    Yield (train_idx, val_idx) splits at HOUSEHOLD level.
    All persons from the same household land in the same fold —
    prevents leakage from correlated household members (context note 1).
    Stratified by the most frequent target (income_class) to keep
    class balance across folds.
    """
    hh_df = df.groupby("hh_id").agg(
        income_class=("income_class", "first")
    ).reset_index()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_hh_idx, val_hh_idx in skf.split(
        hh_df, hh_df["income_class"]
    ):
        train_hh = set(hh_df.iloc[train_hh_idx]["hh_id"])
        val_hh   = set(hh_df.iloc[val_hh_idx]["hh_id"])
        train_idx = df.index[df["hh_id"].isin(train_hh)].tolist()
        val_idx   = df.index[df["hh_id"].isin(val_hh)].tolist()
        yield train_idx, val_idx


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                target_name: str) -> tuple:
    """
    Apply SMOTE on training fold only.
    Skips classes with < MIN_CLASS_SAMPLES to avoid k-neighbor errors.
    Returns (X_resampled, y_resampled).
    Context note 2: SMOTE applied only AFTER train/val split.
    """
    class_counts = np.bincount(y_train)
    min_count    = class_counts[class_counts > 0].min()

    if min_count < MIN_CLASS_SAMPLES:
        k = max(1, min_count - 1)
    else:
        k = SMOTE_K_NEIGHBORS

    try:
        sm = SMOTE(k_neighbors=k, random_state=RANDOM_SEED)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception as e:
        print(f"    SMOTE skipped for {target_name}: {e}")
        return X_train, y_train


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate_target(
    df: pd.DataFrame,
    target: str,
    classes: list,
    device_params: dict,
) -> dict:
    """
    Run N_FOLDS stratified household-level CV for one target.
    Returns fold metrics and out-of-fold predictions.
    """
    # Fit LabelEncoder on actual values present in data, not the full
    # predefined class list — some classes may be absent (e.g. "0-14"
    # in age_bracket since children have no survey feature rows).
    # This keeps XGBoost labels 0-indexed and consecutive.
    valid = df[df[target].notna()].copy()
    actual_classes = sorted(
        valid[target].astype(str).unique().tolist()
    )
    # Warn if predefined classes differ from actual
    predefined = [str(c) for c in classes]
    absent = [c for c in predefined if c not in actual_classes]
    if absent:
        print(f"  NOTE: classes absent from data (skipped): {absent}")

    le = LabelEncoder()
    le.fit(actual_classes)
    classes = actual_classes  # use only classes present in data
    valid["__y"] = le.transform(valid[target].astype(str
                                if target != "hh_size" else int).astype(str))

    X = valid[FEATURE_COLS].values.astype(float)
    y = valid["__y"].values

    fold_metrics = []
    oof_preds    = np.full(len(valid), -1, dtype=int)
    oof_proba    = np.zeros((len(valid), len(classes)), dtype=float)

    params = {**XGB_BASE_PARAMS, **device_params,
              "num_class": len(classes),
              "objective": "multi:softprob"}

    for fold, (train_idx, val_idx) in enumerate(
        household_kfold_split(valid, N_FOLDS, RANDOM_SEED)
    ):
        X_tr = X[train_idx]; y_tr = y[train_idx]
        X_va = X[val_idx];   y_va = y[val_idx]

        # SMOTE on training fold only
        X_tr, y_tr = apply_smote(X_tr, y_tr, target)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        proba = model.predict_proba(X_va)
        preds = proba.argmax(axis=1)

        oof_preds[val_idx]     = preds
        oof_proba[val_idx, :]  = proba

        f1    = f1_score(y_va, preds, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_va, preds)
        sr    = srmse(y_va, preds, np.arange(len(classes)))

        fold_metrics.append({"fold": fold+1, "f1_macro": f1,
                              "kappa": kappa, "srmse": sr})
        print(f"    Fold {fold+1}/{N_FOLDS} — "
              f"F1={f1:.3f}  Kappa={kappa:.3f}  SRMSE={sr:.3f}")

    mean_f1    = np.mean([m["f1_macro"] for m in fold_metrics])
    mean_kappa = np.mean([m["kappa"]    for m in fold_metrics])
    mean_srmse = np.mean([m["srmse"]    for m in fold_metrics])

    # OOF classification report
    valid_oof_mask = oof_preds >= 0
    cr = classification_report(
        y[valid_oof_mask], oof_preds[valid_oof_mask],
        target_names=[str(c) for c in classes],
        zero_division=0, output_dict=True,
    )

    return {
        "fold_metrics":  fold_metrics,
        "mean_f1":       mean_f1,
        "mean_kappa":    mean_kappa,
        "mean_srmse":    mean_srmse,
        "class_report":  cr,
        "oof_preds":     oof_preds,
        "oof_proba":     oof_proba,
        "label_encoder": le,
        "n_samples":     len(valid),
    }


# ── Final model training ───────────────────────────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    target: str,
    classes: list,
    device_params: dict,
    le: LabelEncoder,
) -> xgb.XGBClassifier:
    """
    Train final model on ALL survey data (no val split).
    Used for Stage 4 inference on CDR.
    """
    valid = df[df[target].notna()].copy()
    actual_classes = sorted(valid[target].astype(str).unique().tolist())
    le = LabelEncoder()
    le.fit(actual_classes)
    classes = actual_classes
    valid["__y"] = le.transform(valid[target].astype(str))

    X = valid[FEATURE_COLS].values.astype(float)
    y = valid["__y"].values

    # SMOTE on full training set
    X, y = apply_smote(X, y, target)

    params = {**XGB_BASE_PARAMS, **device_params,
              "num_class":  len(classes),
              "objective":  "multi:softprob",
              "n_estimators": 500}   # slightly more trees on full data

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


# ── Feature importance ─────────────────────────────────────────────────────────

def get_feature_importance(model: xgb.XGBClassifier) -> pd.DataFrame:
    imp = model.get_booster().get_score(importance_type="gain")
    # XGBoost internally names features f0, f1, f2... — map back to real names
    name_map = {f"f{i}": name for i, name in enumerate(FEATURE_COLS)}
    df = pd.DataFrame({
        "feature":    [name_map.get(k, k) for k in imp.keys()],
        "importance": list(imp.values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ── Validation report ──────────────────────────────────────────────────────────

def print_validation_summary(results: dict):
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Target':<20} {'F1-macro':>9} {'Kappa':>9} {'SRMSE':>9} {'N':>8}")
    print("-" * 60)
    for target, res in results.items():
        print(f"{target:<20} {res['mean_f1']:>9.3f} "
              f"{res['mean_kappa']:>9.3f} {res['mean_srmse']:>9.3f} "
              f"{res['n_samples']:>8,}")
    print("-" * 60)

    # Overall pass/fail thresholds (Harland et al. 2012 guidance)
    # F1-macro > 0.35 acceptable for 5-class imbalanced demographic targets
    # Kappa > 0.20 = fair agreement
    # SRMSE < 0.20 = good marginal fit
    passed = True
    print("\nThreshold checks (Harland et al. 2012):")
    for target, res in results.items():
        issues = []
        if res["mean_f1"]    < 0.25: issues.append(f"F1={res['mean_f1']:.3f}<0.25")
        if res["mean_kappa"] < 0.10: issues.append(f"Kappa={res['mean_kappa']:.3f}<0.10")
        if res["mean_srmse"] > 0.40: issues.append(f"SRMSE={res['mean_srmse']:.3f}>0.40")
        if issues:
            print(f"  WARN  {target:<20} {', '.join(issues)}")
            passed = False
        else:
            print(f"  PASS  {target:<20}")
    return passed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading survey features...")
    df = pd.read_parquet(SURVEY_FEAT_PATH)
    print(f"  {len(df):,} persons, {df['hh_id'].nunique():,} households")
    print(f"  Features: {FEATURE_COLS}")

    # Verify all feature cols present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}")
        exit(1)

    # Verify label cols present
    for t in TARGETS:
        if t not in df.columns:
            print(f"ERROR: Missing target column: {t}")
            exit(1)

    # hh_size stored as int — convert to str for label encoding consistency
    df["hh_size"] = df["hh_size"].astype(str)
    TARGETS["hh_size"] = [str(x) for x in TARGETS["hh_size"]]

    device_params = get_xgb_device()

    cv_results  = {}
    final_models = {}
    label_encoders = {}

    print(f"\nRunning {N_FOLDS}-fold household-stratified CV "
          f"across {len(TARGETS)} targets...\n")

    for target, classes in TARGETS.items():
        classes_str = [str(c) for c in classes]
        print(f"\n{'─'*50}")
        print(f"Target: {target}  ({len(classes_str)} classes: {classes_str})")
        print(f"{'─'*50}")

        n_valid = df[target].notna().sum()
        class_dist = df[target].value_counts(normalize=True).round(3)
        print(f"  N valid: {n_valid:,}")
        print(f"  Class distribution: {class_dist.to_dict()}")

        res = cross_validate_target(df, target, classes_str, device_params)
        cv_results[target] = res

        print(f"\n  OOF Summary — F1={res['mean_f1']:.3f} "
              f"Kappa={res['mean_kappa']:.3f} "
              f"SRMSE={res['mean_srmse']:.3f}")

        # Train final model on all data
        print(f"  Training final model on all {n_valid:,} samples...")
        le = res["label_encoder"]
        final_model = train_final_model(df, target, classes_str,
                                        device_params, le)
        final_models[target]   = final_model
        label_encoders[target] = le

        # Save model + encoder
        model_path = os.path.join(MODEL_DIR, f"xgb_{target}.joblib")
        le_path    = os.path.join(MODEL_DIR, f"le_{target}.joblib")
        joblib.dump(final_model, model_path)
        joblib.dump(le,          le_path)
        print(f"  Saved: {model_path}")

        # Feature importance
        imp_df = get_feature_importance(final_model)
        print(f"  Top 5 features: "
              f"{imp_df.head(5)['feature'].tolist()}")

        gc.collect()

    # ── Validation summary + report
    passed = print_validation_summary(cv_results)

    # Save full CV report as JSON
    report = {}
    for target, res in cv_results.items():
        report[target] = {
            "fold_metrics": res["fold_metrics"],
            "mean_f1":      res["mean_f1"],
            "mean_kappa":   res["mean_kappa"],
            "mean_srmse":   res["mean_srmse"],
            "class_report": res["class_report"],
            "n_samples":    res["n_samples"],
        }
    report_path = os.path.join(REPORT_DIR, "demographic_cv_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCV report saved: {report_path}")

    # Save feature list used — Stage 4 needs this to align CDR columns
    meta = {
        "feature_cols":    FEATURE_COLS,
        "targets":         {k: v for k, v in TARGETS.items()},
        "n_folds":         N_FOLDS,
        "xgb_base_params": {k: v for k, v in XGB_BASE_PARAMS.items()
                            if k != "use_label_encoder"},
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model meta saved: {meta_path}")

    print(f"\nStage 3 complete in {time.time()-t0:.0f}s")
    print(f"  Models saved : {len(final_models)} classifiers")
    print(f"  Model dir    : {MODEL_DIR}")

    if not passed:
        print("\nWARN: Some targets below threshold — review CV report.")
        print("      Acceptable for CDR-based demographic inference at ward level.")
        print("      Proceed to Stage 4 — IPF in Stage 5 corrects marginal errors.")
    else:
        print("\nAll targets passed validation thresholds. Proceed to Stage 4.")