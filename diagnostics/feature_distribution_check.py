# mumbai-synth-v2/diagnostics/feature_distribution_check.py
"""
Diagnostic: Compute mean, std, and ratio for all 18 features
across CDR and survey sources.

Decision rule (Pan & Yang 2010 IEEE TKDE covariate shift threshold):
  ratio in [0.70, 1.30] → feature is transferable → KEEP
  ratio outside         → systematic offset → DROP from Stage 3/4

Also computes:
  - KS statistic (two-sample Kolmogorov-Smirnov) per feature
    KS > 0.20 = significant distributional difference → additional DROP signal
  - Cohen's d (effect size of mean difference)
    |d| < 0.50 = small effect → safe to keep
    |d| >= 0.50 = medium/large → drop candidate

Outputs:
  Console table with KEEP/DROP recommendation per feature
  Exact FEATURE_COLS list to paste into Stage 3 + Stage 4
"""

import pandas as pd
import numpy as np
from scipy import stats

CDR_FEAT_PATH    = "../data/processed/cdr_features/cdr_user_features.parquet"
SURVEY_FEAT_PATH = "../data/processed/survey_merged/survey_person_features.parquet"

FEATURE_COLS = [
    "radius_of_gyration_km",
    "location_entropy",
    "home_work_dist_km",
    "n_unique_wards",
    "n_trips_per_day",
    "mean_trip_dist_km",
    "mean_travel_mins",
    "pct_peak_am",
    "pct_peak_pm",
    "pct_offpeak",
    "mode_entropy",
    "dominant_mode",
    "pct_work_trips",
    "pct_local_train",
    "trip_regularity",
    "home_ward_income_tier",
    "home_island_city",
    "home_metro_dist_km",
]

# Thresholds
RATIO_LOW  = 0.70
RATIO_HIGH = 1.30
KS_THRESH  = 0.20
COHEN_D_THRESH = 0.50

def cohen_d(a, b):
    """Cohen's d — pooled standard deviation effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return abs((np.mean(a) - np.mean(b)) / pooled_std)


if __name__ == "__main__":
    print("Loading feature parquets...")
    cdr_df    = pd.read_parquet(CDR_FEAT_PATH)
    survey_df = pd.read_parquet(SURVEY_FEAT_PATH)
    print(f"  CDR    : {len(cdr_df):,} users")
    print(f"  Survey : {len(survey_df):,} persons")

    print(f"\n{'Feature':<28} {'CDR_mean':>10} {'Srv_mean':>10} "
          f"{'Ratio':>7} {'KS':>6} {'CohenD':>8} {'Decision':>8}")
    print("-" * 85)

    keep = []
    drop = []

    for feat in FEATURE_COLS:
        if feat not in cdr_df.columns or feat not in survey_df.columns:
            print(f"{feat:<28} {'MISSING':>10}")
            continue

        cdr_vals = cdr_df[feat].dropna().values.astype(float)
        srv_vals = survey_df[feat].dropna().values.astype(float)

        cdr_mean = np.mean(cdr_vals)
        srv_mean = np.mean(srv_vals)
        cdr_std  = np.std(cdr_vals, ddof=1)
        srv_std  = np.std(srv_vals, ddof=1)

        # Ratio — use safe division
        if srv_mean == 0:
            ratio = float("inf") if cdr_mean != 0 else 1.0
        else:
            ratio = cdr_mean / srv_mean

        # KS statistic (subsample to 5000 each for speed)
        rng = np.random.default_rng(42)
        a = rng.choice(cdr_vals, size=min(5000, len(cdr_vals)), replace=False)
        b = rng.choice(srv_vals, size=min(5000, len(srv_vals)), replace=False)
        ks_stat, _ = stats.ks_2samp(a, b)

        # Cohen's d
        cd = cohen_d(cdr_vals, srv_vals)

        # Decision: KEEP only if ALL three criteria met
        ratio_ok   = RATIO_LOW <= ratio <= RATIO_HIGH
        ks_ok      = ks_stat < KS_THRESH
        cohen_ok   = cd < COHEN_D_THRESH

        n_fail = sum([not ratio_ok, not ks_ok, not cohen_ok])
        if n_fail == 0:
            decision = "KEEP"
            keep.append(feat)
        elif n_fail == 1:
            decision = "MARGINAL"
            # Marginal: keep if ratio is ok (ratio is the primary criterion)
            if ratio_ok:
                keep.append(feat)
                decision = "KEEP*"
            else:
                drop.append(feat)
                decision = "DROP*"
        else:
            decision = "DROP"
            drop.append(feat)

        print(f"{feat:<28} {cdr_mean:>10.3f} {srv_mean:>10.3f} "
              f"{ratio:>7.3f} {ks_stat:>6.3f} {cd:>8.3f} {decision:>8}")

    print("-" * 85)
    print(f"\nSummary: {len(keep)} KEEP, {len(drop)} DROP")

    print(f"\n{'='*50}")
    print("KEEP — transferable features (paste into Stage 3 + 4):")
    print("FEATURE_COLS = [")
    for f in keep:
        print(f'    "{f}",')
    print("]")

    print(f"\nDROP — systematic CDR/survey offset:")
    for f in drop:
        print(f"  - {f}")

    print(f"\nDetailed stats for DROP features:")
    print(f"{'Feature':<28} {'CDR_mean':>10} {'CDR_std':>10} "
          f"{'Srv_mean':>10} {'Srv_std':>10} {'Ratio':>7}")
    print("-" * 80)
    for feat in drop:
        if feat not in cdr_df.columns:
            continue
        c = cdr_df[feat].dropna().values.astype(float)
        s = survey_df[feat].dropna().values.astype(float)
        r = np.mean(c)/np.mean(s) if np.mean(s) != 0 else float("inf")
        print(f"{feat:<28} {np.mean(c):>10.3f} {np.std(c,ddof=1):>10.3f} "
              f"{np.mean(s):>10.3f} {np.std(s,ddof=1):>10.3f} {r:>7.3f}")