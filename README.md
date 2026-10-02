# Mumbai Synthetic Population Engine v2

A multi-stage pipeline that generates a calibrated synthetic population for Mumbai's 24 BMC wards — inferred from synthetic CDR (Call Detail Records) and household travel survey data. Designed for urban mobility modelling, activity-based simulation, and transport policy analysis.

---

## Overview

The pipeline takes no real data as input. It generates statistically realistic CDR and survey records from scratch, processes them into mobility features, trains XGBoost demographic classifiers on the survey data, and applies them to the CDR population — producing a 200,000-person synthetic base sample with inferred socio-demographic attributes.

```
Stage 0  →  Stage 1  →  Stage 2  →  Stage 3  →  Stage 4
Generate     Process      Feature     Train        Apply
CDR &        CDR into     Engineer    XGBoost      models to
Survey       trips &      mobility    classifiers  CDR users
             anchors      vectors     on survey    → base sample
```

---

## Project Structure

```
mumbai_synth_v2/
│
├── generation/
│   ├── generate_cdr.py               # Stage 0a — Synthetic CDR generation (200k users, 5 days)
│   ├── generate_survey.py            # Stage 0b — Synthetic household travel survey
│   └── diagnose.py                   # Debug utility for CDR ping spatial coherence
│
├── processing/
│   ├── cdr_processor.py              # Stage 1 — Stay points, anchors, trip extraction
│   ├── feature_engineer.py           # Stage 2 — Mobility feature vectors (CDR + survey)
│   └── coherence                     # Coherence check notes
│
├── models/
│   ├── demographic_inferrer.py       # Stage 3 — Train XGBoost classifiers on survey
│   ├── cdr_applicator.py             # Stage 4 — Apply classifiers to CDR users
│   └── saved/
│       ├── xgb_age_bracket.joblib
│       ├── xgb_employment.joblib
│       ├── xgb_hh_income.joblib
│       ├── xgb_hh_size.joblib
│       ├── xgb_income_class.joblib
│       ├── xgb_vehicle_owned.joblib
│       ├── le_*.joblib               # Label encoders for each target
│       └── model_meta.json           # Feature list, targets, XGBoost hyperparameters
│
├── diagnostics/
│   └── feature_distribution_check.py # CDR vs survey feature distribution comparison (KS, Cohen's d)
│
├── data/
│   ├── base_sample/
│   │   └── cdr_demographics.parquet  # Stage 4 output — 91,654 CDR users with inferred demographics
│   └── processed/
│       ├── cdr_features/
│       │   └── cdr_user_features.parquet
│       ├── cdr_trips/
│       │   ├── cdr_stay_points.parquet
│       │   ├── cdr_anchors.parquet
│       │   └── cdr_trips.parquet
│       └── survey_merged/
│           └── survey_person_features.parquet
│
├── validation/
│   └── reports/
│       ├── demographic_cv_report.json     # Stage 3 cross-validation results
│       └── stage4_validation_report.json  # Stage 4 marginal distribution + OOD summary
│
├── survey_sociodem.parquet           # Raw synthetic survey — socio-demographic attributes
├── survey_trips.parquet              # Raw synthetic survey — trip records
├── towers.parquet                    # Cell tower locations across BMC wards
└── users.parquet                     # CDR user metadata (home ward, work ward, etc.)
```

---

## Pipeline Stages

### Stage 0 — Data Generation

**`generate_cdr.py`** generates 200,000 synthetic CDR users across all 24 BMC wards, with ward populations proportional to census data. Each user gets 5 days of ping records with spatially coherent commute corridors — pings during commute are restricted to towers along the corridor between home and work wards (using a ward adjacency map) to ensure stay point detection works correctly.

**`generate_survey.py`** generates a synthetic household travel survey (~36k persons) with closed home-based trip tours. Employment-class-specific departure time distributions are calibrated to real Mumbai patterns (e.g. student departure N(9.0, 0.5) from university gravity model; employed departure N(8.0, 0.6) matching MMRDA AM peak).

### Stage 1 — CDR Processing (`cdr_processor.py`)

Converts raw ping records into structured travel behaviour per user:
- **Stay point detection** — vectorized NumPy, 0.5 km radius / 20 min dwell threshold
- **Home/work anchor inference** — DBSCAN clustering on night-hour and day-hour pings
- **Trip extraction** — transitions between consecutive stay points
- **Mode inference** — rule-based on speed, corridor geometry (Western/Central/Harbour rail lines, Metro Line 1)

### Stage 2 — Feature Engineering (`feature_engineer.py`)

Extracts 20 mobility features from both CDR and survey sources using the same schema, so the classifier trained on survey data can be applied to CDR users without distribution mismatch.

Key features: `radius_of_gyration_km`, `location_entropy`, `home_work_dist_km`, `pct_peak_am`, `pct_peak_pm`, `pct_local_train`, `trip_regularity`, `home_ward_income_tier`, `home_metro_dist_km`, and 11 others.

Two features (`mean_speed_kmh`, `pct_walk`) are excluded due to systematic CDR/survey distribution mismatch confirmed by the diagnostics script.

### Stage 3 — Demographic Inference (`demographic_inferrer.py`)

Trains one XGBoost classifier per target attribute on the survey feature vectors (labeled):

| Target | Classes |
|---|---|
| `age_bracket` | 0-14 / 15-24 / 25-44 / 45-59 / 60+ |
| `employment` | employed / student / homemaker / retired / unemployed |
| `income_class` | low / mid / high |
| `vehicle_owned` | none / 2w / car / both |
| `hh_size` | 1–7 |
| `hh_income` | low / mid / high |

- Stratified K-Fold split at **household level** (not person level) to prevent data leakage from correlated household members
- SMOTE applied per target on training fold only
- Validation: F1-macro, Cohen's Kappa, SRMSE on marginal distributions

### Stage 4 — Apply to CDR (`cdr_applicator.py`)

Applies trained classifiers to 91,654 CDR users. Also runs **Isolation Forest OOD detection** (contamination=0.05) to flag CDR users whose mobility features fall outside the survey training distribution. OOD users are not dropped — they receive inferred demographics with an `ood_flag` and a `confidence_score` for downstream IPF reweighting.

Temperature scaling (T=1.5) is applied to soften overconfident XGBoost softmax outputs.

---

## Installation & Setup

### Prerequisites

- Python **3.11+**
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mumbai-synth-v2.git
cd mumbai-synth-v2
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install numpy pandas geopandas scikit-learn xgboost imbalanced-learn joblib scipy shapely pyarrow
```

---

## Running the Pipeline

Run each stage in order from the project root. Each stage reads the output of the previous one.

```bash
# Stage 0 — Generate raw data
python generation/generate_cdr.py
python generation/generate_survey.py

# Stage 1 — Process CDR
python processing/cdr_processor.py

# Stage 2 — Feature engineering
python processing/feature_engineer.py

# Stage 3 — Train demographic classifiers
python models/demographic_inferrer.py

# Stage 4 — Apply to CDR population
python models/cdr_applicator.py
```

The final output is `data/base_sample/cdr_demographics.parquet` — one row per CDR user with inferred demographic attributes, ready for PopulationSim IPF reweighting.

### Optional — Diagnostics

```bash
# Check CDR vs survey feature distributions (KS statistic, Cohen's d, keep/drop recommendation)
python diagnostics/feature_distribution_check.py

# Debug CDR spatial coherence
python generation/diagnose.py
```

---

## Calibration Sources

| Source | Usage |
|---|---|
| BMC Census Ward Data | Ward populations, centroids |
| MMRDA Mumbai Metropolitan Plan 2016–2036 | Land-use zone weights, peak hour definitions |
| Mumbai University campus locations | Student work-ward gravity model (IIT Bombay, VJTI, ICT, Mumbai University Fort) |
| TomTom Traffic Index 2023 | Peak hour validation |
| Western / Central / Harbour rail geometry | Mode inference corridor boundaries |
| Harland et al. (2012) | Validation metrics (SRMSE) |
| Pan & Yang (2010) IEEE TKDE | Covariate shift threshold for feature transfer |

---

## License

MIT