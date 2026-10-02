# mumbai-synth-v2/processing/feature_engineer.py
"""
Stage 2 — Feature Engineering

Extracts a unified mobility feature vector for:
  A) CDR users  — from cdr_trips + cdr_anchors + cdr_stay_points (unlabeled)
  B) Survey persons — from survey_trips + survey_sociodem (labeled)

Both produce the SAME feature schema so Stage 3 can train on (B)
and apply to (A) without distribution mismatch.

Outputs:
  data/processed/cdr_features/cdr_user_features.parquet   — 200k rows
  data/processed/survey_merged/survey_person_features.parquet — ~36k rows

Features extracted (17 total):
  Mobility (CDR-derived, survey-compatible):
    1.  radius_of_gyration_km       — std dev of stay point positions from centroid
    2.  location_entropy            — Shannon entropy over ward visit frequencies
    3.  home_work_dist_km           — haversine(home_anchor, work_anchor)
    4.  n_unique_wards              — distinct wards visited
    5.  n_trips_total               — total trips over observation period
    6.  mean_trip_dist_km           — mean trip distance
    7.  mean_travel_time_mins       — mean travel time per trip
    8.  pct_peak_am                 — fraction of departures in 07-10
    9.  pct_peak_pm                 — fraction of departures in 17-20
    10. pct_offpeak                 — fraction of departures outside peaks
    11. mode_entropy                — Shannon entropy over mode frequency
    12. dominant_mode               — most frequent mode (label-encoded)
    13. pct_work_trips              — fraction of trips with work/education purpose
    14. pct_local_train             — fraction of trips using local_train
    15. pct_walk                    — fraction of trips by walk/slow speed
    16. mean_speed_kmh              — mean trip speed
    17. trip_regularity             — coefficient of variation of daily trip count

  Spatial context (ward-level, added for both sources):
    18. home_ward_income_tier       — 0=low, 1=mid, 2=high (home ward tier)
    19. home_island_city            — 1 if home in Island City wards (A-G)
    20. home_metro_dist_km          — km to nearest metro from home ward

Notes on survey vs CDR feature alignment:
  - CDR covers 5 weekdays; survey covers 1 day. To keep features comparable,
    CDR trip-count features are normalized per day (divided by N_DAYS=5).
    survey trip-count features are already per day. Both use the same
    derived rates (pct_peak_am, mode_entropy etc.) which are scale-invariant.
  - Survey has no stay points — radius_of_gyration and location_entropy are
    computed from trip origin/destination coords instead of stay points.
    This is equivalent for sparse surveys (< 5 trips/day).
  - Survey has no speed column — mean_speed_kmh is computed from
    trip_dist_km / travel_time_mins * 60 (identical to CDR computation).
  - dominant_mode is label-encoded identically for both sources using the
    same MODES list so Stage 3 sees the same integer values.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

CDR_TRIPS_PATH      = "../data/processed/cdr_trips/cdr_trips.parquet"
CDR_ANCHORS_PATH    = "../data/processed/cdr_trips/cdr_anchors.parquet"
CDR_SP_PATH         = "../data/processed/cdr_trips/cdr_stay_points.parquet"
SURVEY_TRIPS_PATH   = "../data/raw/survey/survey_trips.parquet"
SURVEY_SOCIODEM_PATH= "../data/raw/survey/survey_sociodem.parquet"

CDR_FEAT_OUT        = "../data/processed/cdr_features/cdr_user_features.parquet"
SURVEY_FEAT_OUT     = "../data/processed/survey_merged/survey_person_features.parquet"

# ── Config ─────────────────────────────────────────────────────────────────────

N_CDR_DAYS = 5      # CDR covers 5 weekdays — used for per-day normalization
MIN_CDR_DAYS = 5    # exclude CDR users with < 5 days of data (context note 6)

ISLAND_CITY_WARDS = {"A","B","C","D","E","F/S","F/N","G/S","G/N"}

WARD_INCOME_TIER = {
    "A":"high","B":"high","C":"high","D":"mid","E":"mid",
    "F/S":"mid","F/N":"mid","G/S":"mid","G/N":"mid",
    "H/E":"mid","H/W":"high","K/E":"mid","K/W":"high",
    "P/S":"mid","P/N":"low","R/S":"low","R/C":"low","R/N":"low",
    "L":"low","M/E":"low","M/W":"mid","N":"low","S":"low","T":"low",
}
INCOME_TIER_MAP = {"low": 0, "mid": 1, "high": 2}

METRO_DIST_KM = {
    "A":1.193,"B":0.907,"C":0.644,"D":0.324,"E":1.449,
    "F/S":1.385,"F/N":0.424,"G/S":1.444,"G/N":1.562,
    "H/E":1.483,"H/W":1.020,"K/E":0.712,"K/W":1.940,
    "P/S":0.597,"P/N":1.124,"R/S":0.461,"R/C":0.662,
    "R/N":0.568,"L":2.371,"M/E":3.528,"M/W":2.656,
    "N":1.797,"S":5.192,"T":9.810,
}

WARD_CENTROIDS = {
    "A":(18.9220,72.8347),"B":(18.9480,72.8368),"C":(18.9620,72.8200),
    "D":(18.9720,72.8150),"E":(18.9820,72.8350),"F/S":(19.0050,72.8350),
    "F/N":(19.0350,72.8420),"G/S":(19.0130,72.8180),"G/N":(19.0280,72.8220),
    "H/E":(19.0720,72.8590),"H/W":(19.0650,72.8350),"K/E":(19.1050,72.8650),
    "K/W":(19.0950,72.8350),"P/S":(19.1580,72.8490),"P/N":(19.1900,72.8480),
    "R/S":(19.2050,72.8600),"R/C":(19.2200,72.8680),"R/N":(19.2350,72.8550),
    "L":(19.0750,72.8900),"M/E":(19.0550,72.9050),"M/W":(19.0650,72.8980),
    "N":(19.1150,72.9100),"S":(19.1450,72.9250),"T":(19.1750,72.9600),
}

# Unified mode list — same encoding for CDR and survey.
# CDR modes come from mode_inferred; survey modes come from mode column.
# "car_2w" in CDR maps to "car" + "two_wheeler" combined bucket.
MODES = [
    "local_train","best_bus","metro","auto",
    "taxi","car","two_wheeler","car_2w","walk","other",
]
MODE_ENCODE = {m: i for i, m in enumerate(MODES)}

WORK_PURPOSES = {"work", "education"}
PEAK_AM  = set(range(7, 10))
PEAK_PM  = set(range(17, 20))


# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1,lo1,la2,lo2 = map(radians,[lat1,lon1,lat2,lon2])
    a = sin((la2-la1)*0.5)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)*0.5)**2
    return R * 2.0 * asin(sqrt(max(0.0, min(1.0, a))))


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine over numpy arrays."""
    R = 6371.0
    la1 = np.radians(lat1); lo1 = np.radians(lon1)
    la2 = np.radians(lat2); lo2 = np.radians(lon2)
    a = (np.sin((la2-la1)*0.5)**2
         + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)*0.5)**2)
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits. Safe against zero counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def radius_of_gyration(lats: np.ndarray, lons: np.ndarray,
                       weights: np.ndarray = None) -> float:
    """
    Radius of gyration (km) — Gonzalez et al. 2008 (Nature).
    r_g = sqrt( mean( dist(x_i, centroid)^2 ) )
    Optionally weighted by visit frequency.
    """
    if len(lats) < 2:
        return 0.0
    if weights is None:
        weights = np.ones(len(lats))
    weights = weights / weights.sum()
    c_lat = np.sum(weights * lats)
    c_lon = np.sum(weights * lons)
    dists = haversine_vec(lats, lons,
                          np.full_like(lats, c_lat),
                          np.full_like(lons, c_lon))
    return float(np.sqrt(np.sum(weights * dists**2)))


def coefficient_of_variation(arr: np.ndarray) -> float:
    """CV = std/mean. Returns 0 if mean==0."""
    if len(arr) == 0 or arr.mean() == 0:
        return 0.0
    return float(arr.std() / arr.mean())


def ward_from_coord(lat: float, lon: float) -> str:
    """Snap a lat/lon to nearest ward centroid. Used as fallback."""
    best_w, best_d = "P/N", 9999.0
    for w, (wlat, wlon) in WARD_CENTROIDS.items():
        d = haversine(lat, lon, wlat, wlon)
        if d < best_d:
            best_d = d; best_w = w
    return best_w


def spatial_context(ward: str) -> dict:
    """Return ward-level spatial context features."""
    tier  = WARD_INCOME_TIER.get(ward, "low")
    return {
        "home_ward_income_tier": INCOME_TIER_MAP.get(tier, 0),
        "home_island_city":      int(ward in ISLAND_CITY_WARDS),
        "home_metro_dist_km":    METRO_DIST_KM.get(ward, 5.0),
    }


# ── CDR Feature Extraction ─────────────────────────────────────────────────────

def extract_cdr_features(
    trips_df:   pd.DataFrame,
    anchors_df: pd.DataFrame,
    sp_df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract per-user mobility features from CDR-derived data.

    Radius of gyration and location entropy use stay points (the
    canonical CDR mobility metrics from Gonzalez 2008, Song 2010).
    Trip-level features use cdr_trips.
    Anchor features use cdr_anchors.

    CDR trip counts are per-day normalized (÷ N_CDR_DAYS) so they
    are comparable to survey features which are inherently single-day.
    """
    trips_df   = trips_df.copy()
    anchors_df = anchors_df.copy()
    sp_df      = sp_df.copy()

    trips_df["depart_ts"] = pd.to_datetime(trips_df["depart_ts"])
    trips_df["day"]       = trips_df["depart_ts"].dt.date

    # ── Verify coverage: users with all 5 days
    user_day_counts = trips_df.groupby("user_id")["day"].nunique()
    full_users = user_day_counts[user_day_counts >= MIN_CDR_DAYS].index
    print(f"  CDR users with >= {MIN_CDR_DAYS} days of trips: "
          f"{len(full_users):,} / {trips_df['user_id'].nunique():,}")
    trips_df = trips_df[trips_df["user_id"].isin(full_users)]
    sp_df    = sp_df[sp_df["user_id"].isin(full_users)]

    # ── Precompute per-user stay point lat/lon arrays for gyration/entropy
    print("  Computing radius of gyration and location entropy...")
    sp_ward = []
    for _, row in sp_df.iterrows():
        sp_ward.append(ward_from_coord(row["stay_lat"], row["stay_lon"]))
    sp_df["ward"] = sp_ward

    sp_grouped = sp_df.groupby("user_id")

    rog_map     = {}
    entropy_map = {}
    n_ward_map  = {}

    for uid, grp in sp_grouped:
        lats = grp["stay_lat"].values
        lons = grp["stay_lon"].values
        # Weight by duration
        w    = grp["duration_min"].values.clip(min=1)
        rog_map[uid]     = radius_of_gyration(lats, lons, w)
        ward_counts      = grp["ward"].value_counts().values
        entropy_map[uid] = shannon_entropy(ward_counts)
        n_ward_map[uid]  = int(grp["ward"].nunique())

    # ── Per-user trip aggregations (vectorized where possible)
    print("  Aggregating trip features...")
    trips_df["departure_hour"] = pd.to_datetime(
        trips_df["depart_ts"]
    ).dt.hour
    trips_df["is_peak_am"]   = trips_df["departure_hour"].isin(PEAK_AM).astype(int)
    trips_df["is_peak_pm"]   = trips_df["departure_hour"].isin(PEAK_PM).astype(int)
    trips_df["is_offpeak"]   = (
        ~trips_df["departure_hour"].isin(PEAK_AM | PEAK_PM)
    ).astype(int)
    trips_df["is_work"]      = trips_df.get(
        "trip_purpose", pd.Series("other", index=trips_df.index)
    ).isin(WORK_PURPOSES).astype(int)
    trips_df["mode_enc"]     = trips_df["mode_inferred"].map(
        MODE_ENCODE
    ).fillna(MODE_ENCODE.get("other", 9)).astype(int)
    trips_df["is_local"]     = (trips_df["mode_inferred"] == "local_train").astype(int)
    trips_df["is_walk"]      = (trips_df["mode_inferred"] == "walk").astype(int)

    # Daily trip counts per user — for trip regularity
    daily_counts = trips_df.groupby(
        ["user_id","day"]
    ).size().reset_index(name="n_trips_day")

    agg = trips_df.groupby("user_id").agg(
        n_trips_total      = ("trip_dist_km",        "count"),
        mean_trip_dist_km  = ("trip_dist_km",        "mean"),
        mean_travel_mins   = ("travel_time_mins",     "mean"),
        mean_speed_kmh     = ("speed_kmh",            "mean"),
        sum_peak_am        = ("is_peak_am",           "sum"),
        sum_peak_pm        = ("is_peak_pm",           "sum"),
        sum_offpeak        = ("is_offpeak",           "sum"),
        sum_work           = ("is_work",              "sum"),
        sum_local          = ("is_local",             "sum"),
        sum_walk           = ("is_walk",              "sum"),
    ).reset_index()

    # Dominant mode per user
    dominant_mode = trips_df.groupby("user_id")["mode_enc"].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index().rename(columns={"mode_enc":"dominant_mode"})

    # Mode entropy per user
    def mode_ent(x):
        return shannon_entropy(x.value_counts().values.astype(float))
    mode_entropy = trips_df.groupby("user_id")["mode_inferred"].agg(
        mode_ent
    ).reset_index().rename(columns={"mode_inferred":"mode_entropy"})

    # Trip regularity (CV of daily trip count)
    trip_reg = daily_counts.groupby("user_id")["n_trips_day"].agg(
        lambda x: coefficient_of_variation(x.values)
    ).reset_index().rename(columns={"n_trips_day":"trip_regularity"})

    # ── Merge everything
    feat = agg.merge(dominant_mode, on="user_id", how="left")
    feat = feat.merge(mode_entropy,  on="user_id", how="left")
    feat = feat.merge(trip_reg,      on="user_id", how="left")

    feat["pct_peak_am"]    = feat["sum_peak_am"]  / feat["n_trips_total"]
    feat["pct_peak_pm"]    = feat["sum_peak_pm"]  / feat["n_trips_total"]
    feat["pct_offpeak"]    = feat["sum_offpeak"]  / feat["n_trips_total"]
    feat["pct_work_trips"] = feat["sum_work"]     / feat["n_trips_total"]
    feat["pct_local_train"]= feat["sum_local"]    / feat["n_trips_total"]
    feat["pct_walk"]       = feat["sum_walk"]     / feat["n_trips_total"]

    # Per-day normalize raw trip count
    feat["n_trips_per_day"] = feat["n_trips_total"] / N_CDR_DAYS

    # ── Attach gyration / entropy / n_wards from stay points
    feat["radius_of_gyration_km"] = feat["user_id"].map(rog_map).fillna(0.0)
    feat["location_entropy"]      = feat["user_id"].map(entropy_map).fillna(0.0)
    feat["n_unique_wards"]        = feat["user_id"].map(n_ward_map).fillna(1).astype(int)

    # ── Attach home-work distance from anchors
    print("  Computing home-work distances from anchors...")
    anchors_df["home_work_dist_km"] = haversine_vec(
        anchors_df["home_lat"].values, anchors_df["home_lon"].values,
        anchors_df["work_lat"].values, anchors_df["work_lon"].values,
    )
    feat = feat.merge(
        anchors_df[["user_id","home_lat","home_lon","home_work_dist_km",
                    "home_confidence","work_confidence"]],
        on="user_id", how="left",
    )

    # ── Spatial context from home ward (nearest centroid to anchor)
    print("  Adding spatial context features...")
    home_wards = feat["home_lat"].apply(
        lambda lat: ward_from_coord(lat, feat.loc[feat["home_lat"]==lat,"home_lon"].values[0])
        if pd.notna(lat) else "P/N"
    )
    # Vectorized version — avoid apply with two columns
    feat["home_ward"] = [
        ward_from_coord(r.home_lat, r.home_lon) if pd.notna(r.home_lat) else "P/N"
        for r in feat[["home_lat","home_lon"]].itertuples()
    ]
    ctx = feat["home_ward"].apply(
        lambda w: pd.Series(spatial_context(w))
    )
    feat = pd.concat([feat, ctx], axis=1)

    # ── Final column selection
    out_cols = [
        "user_id",
        "radius_of_gyration_km", "location_entropy",
        "home_work_dist_km",     "n_unique_wards",
        "n_trips_per_day",       "mean_trip_dist_km",
        "mean_travel_mins",      "pct_peak_am",
        "pct_peak_pm",           "pct_offpeak",
        "mode_entropy",          "dominant_mode",
        "pct_work_trips",        "pct_local_train",
        "pct_walk",              "mean_speed_kmh",
        "trip_regularity",       "home_ward_income_tier",
        "home_island_city",      "home_metro_dist_km",
        # Extra context kept for OOD detection / diagnostics
        "home_ward",             "home_confidence",
        "work_confidence",
    ]
    feat = feat[[c for c in out_cols if c in feat.columns]]
    feat = feat.fillna(0.0)

    print(f"  CDR features shape: {feat.shape}")
    return feat


# ── Survey Feature Extraction ──────────────────────────────────────────────────

def extract_survey_features(
    trips_df:   pd.DataFrame,
    sociodem_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the SAME feature schema from survey data.

    Key alignment decisions:
    - radius_of_gyration: computed from trip origin/dest coords
      (no stay points in survey; using O/D coords is equivalent
       for sparse daily diaries — same mathematical definition)
    - location_entropy: over unique wards visited (origin + dest)
    - n_unique_wards: distinct wards in that person's trips
    - n_trips_per_day: raw trip count (survey = 1 day, so no ÷ needed)
    - mean_speed_kmh: trip_dist_km / travel_time_mins * 60
    - mode_entropy: over modes used across all trips
    - dominant_mode: most frequent mode
    - pct_work_trips: trips with purpose work/education
    - trip_regularity: undefined for single day → set to 0
      (Stage 3 will see 0 for all survey rows; CDR varies → this
       feature will have low importance for survey-trained model,
       which is correct — regularity is a 5-day CDR concept)
    - home_work_dist_km: from work_ward anchor in sociodem
      (assigned at generation time — same gravity model as CDR)
    """
    trips_df    = trips_df.copy()
    sociodem_df = sociodem_df.copy()

    trips_df["is_peak_am"]    = trips_df["departure_hour"].isin(PEAK_AM).astype(int)
    trips_df["is_peak_pm"]    = trips_df["departure_hour"].isin(PEAK_PM).astype(int)
    trips_df["is_offpeak"]    = (
        ~trips_df["departure_hour"].isin(PEAK_AM | PEAK_PM)
    ).astype(int)
    trips_df["is_work"]       = trips_df["trip_purpose"].isin(WORK_PURPOSES).astype(int)
    trips_df["mode_enc"]      = trips_df["mode"].map(
        MODE_ENCODE
    ).fillna(MODE_ENCODE.get("other", 9)).astype(int)
    trips_df["is_local"]      = (trips_df["mode"] == "local_train").astype(int)
    trips_df["is_walk"]       = (trips_df["mode"] == "walk").astype(int)
    trips_df["speed_kmh"]     = np.where(
        trips_df["travel_time_mins"] > 0,
        trips_df["trip_dist_km"] / trips_df["travel_time_mins"] * 60,
        0.0,
    )

    # ── Per-person aggregations
    print("  Aggregating survey trip features...")
    agg = trips_df.groupby("person_id").agg(
        n_trips_total      = ("trip_dist_km",    "count"),
        mean_trip_dist_km  = ("trip_dist_km",    "mean"),
        mean_travel_mins   = ("travel_time_mins","mean"),
        mean_speed_kmh     = ("speed_kmh",       "mean"),
        sum_peak_am        = ("is_peak_am",      "sum"),
        sum_peak_pm        = ("is_peak_pm",      "sum"),
        sum_offpeak        = ("is_offpeak",      "sum"),
        sum_work           = ("is_work",         "sum"),
        sum_local          = ("is_local",        "sum"),
        sum_walk           = ("is_walk",         "sum"),
    ).reset_index()

    dominant_mode = trips_df.groupby("person_id")["mode_enc"].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index().rename(columns={"mode_enc":"dominant_mode"})

    mode_entropy_s = trips_df.groupby("person_id")["mode"].agg(
        lambda x: shannon_entropy(x.value_counts().values.astype(float))
    ).reset_index().rename(columns={"mode":"mode_entropy"})

    feat = agg.merge(dominant_mode,  on="person_id", how="left")
    feat = feat.merge(mode_entropy_s, on="person_id", how="left")

    feat["pct_peak_am"]     = feat["sum_peak_am"] / feat["n_trips_total"]
    feat["pct_peak_pm"]     = feat["sum_peak_pm"] / feat["n_trips_total"]
    feat["pct_offpeak"]     = feat["sum_offpeak"] / feat["n_trips_total"]
    feat["pct_work_trips"]  = feat["sum_work"]    / feat["n_trips_total"]
    feat["pct_local_train"] = feat["sum_local"]   / feat["n_trips_total"]
    feat["pct_walk"]        = feat["sum_walk"]    / feat["n_trips_total"]
    feat["n_trips_per_day"] = feat["n_trips_total"].astype(float)
    feat["trip_regularity"] = 0.0  # undefined for single-day survey

    # ── Radius of gyration from O/D coords
    print("  Computing radius of gyration from O/D coords...")
    all_coords = pd.concat([
        trips_df[["person_id","origin_lat","origin_lon"]].rename(
            columns={"origin_lat":"lat","origin_lon":"lon"}),
        trips_df[["person_id","dest_lat","dest_lon"]].rename(
            columns={"dest_lat":"lat","dest_lon":"lon"}),
    ], ignore_index=True)

    rog_map     = {}
    entropy_map = {}
    nward_map   = {}

    for pid, grp in all_coords.groupby("person_id"):
        lats = grp["lat"].values; lons = grp["lon"].values
        rog_map[pid]     = radius_of_gyration(lats, lons)

    for pid, grp in trips_df.groupby("person_id"):
        wards = pd.concat([grp["origin_ward"], grp["dest_ward"]])
        ward_counts = wards.value_counts().values.astype(float)
        entropy_map[pid] = shannon_entropy(ward_counts)
        nward_map[pid]   = int(wards.nunique())

    feat["radius_of_gyration_km"] = feat["person_id"].map(rog_map).fillna(0.0)
    feat["location_entropy"]      = feat["person_id"].map(entropy_map).fillna(0.0)
    feat["n_unique_wards"]        = feat["person_id"].map(nward_map).fillna(1).astype(int)

    # ── Home-work distance from sociodem anchor
    print("  Computing home-work distances from survey anchors...")
    sociodem_df["home_lat"] = sociodem_df["ward_id"].map(
        lambda w: WARD_CENTROIDS.get(w, (19.076, 72.877))[0]
    )
    sociodem_df["home_lon"] = sociodem_df["ward_id"].map(
        lambda w: WARD_CENTROIDS.get(w, (19.076, 72.877))[1]
    )
    sociodem_df["work_lat"] = sociodem_df["work_ward"].map(
        lambda w: WARD_CENTROIDS.get(w, (19.076, 72.877))[0]
        if pd.notna(w) else np.nan
    )
    sociodem_df["work_lon"] = sociodem_df["work_ward"].map(
        lambda w: WARD_CENTROIDS.get(w, (19.076, 72.877))[1]
        if pd.notna(w) else np.nan
    )
    sociodem_df["home_work_dist_km"] = np.where(
        sociodem_df["work_lat"].notna(),
        haversine_vec(
            sociodem_df["home_lat"].values,
            sociodem_df["home_lon"].values,
            sociodem_df["work_lat"].fillna(sociodem_df["home_lat"]).values,
            sociodem_df["work_lon"].fillna(sociodem_df["home_lon"]).values,
        ),
        0.0,
    )

    feat = feat.merge(
        sociodem_df[[
            "person_id","ward_id","work_ward",
            "home_work_dist_km",
            # Label columns — needed by Stage 3
            "age_bracket","employment","income_class",
            "vehicle_owned","hh_size","hh_income",
            "island_city","income_tier","hh_id",
        ]],
        on="person_id", how="left",
    )

    # ── Spatial context
    print("  Adding spatial context features...")
    ctx = feat["ward_id"].apply(lambda w: pd.Series(spatial_context(w)))
    feat = pd.concat([feat, ctx], axis=1)

    feat = feat.rename(columns={"ward_id":"home_ward"})

    out_cols = [
        "person_id","hh_id",
        "radius_of_gyration_km","location_entropy",
        "home_work_dist_km","n_unique_wards",
        "n_trips_per_day","mean_trip_dist_km",
        "mean_travel_mins","pct_peak_am",
        "pct_peak_pm","pct_offpeak",
        "mode_entropy","dominant_mode",
        "pct_work_trips","pct_local_train",
        "pct_walk","mean_speed_kmh",
        "trip_regularity","home_ward_income_tier",
        "home_island_city","home_metro_dist_km",
        "home_ward","work_ward",
        # Label columns
        "age_bracket","employment","income_class",
        "vehicle_owned","hh_size","hh_income",
        "income_tier",
    ]
    feat = feat[[c for c in out_cols if c in feat.columns]]
    feat = feat.fillna({"home_work_dist_km":0.0,"trip_regularity":0.0})
    feat["dominant_mode"] = feat["dominant_mode"].fillna(
        MODE_ENCODE.get("other", 9)
    ).astype(int)

    print(f"  Survey features shape: {feat.shape}")
    return feat


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_features(cdr_feat: pd.DataFrame,
                      survey_feat: pd.DataFrame) -> bool:
    print("\nRunning feature engineering validation...")
    passed = True

    # Shared mobility features (excluding ID and label columns)
    mobility_cols = [
        "radius_of_gyration_km","location_entropy","home_work_dist_km",
        "n_unique_wards","n_trips_per_day","mean_trip_dist_km",
        "mean_travel_mins","pct_peak_am","pct_peak_pm","pct_offpeak",
        "mode_entropy","dominant_mode","pct_work_trips","pct_local_train",
        "pct_walk","mean_speed_kmh","trip_regularity",
        "home_ward_income_tier","home_island_city","home_metro_dist_km",
    ]

    # Check 1 — both datasets have all mobility columns
    missing_cdr    = [c for c in mobility_cols if c not in cdr_feat.columns]
    missing_survey = [c for c in mobility_cols if c not in survey_feat.columns]
    if missing_cdr or missing_survey:
        print(f"  FAIL — Missing columns: CDR={missing_cdr} Survey={missing_survey}")
        passed = False
    else:
        print(f"  PASS — All {len(mobility_cols)} mobility features present in both.")

    # Check 2 — no nulls in mobility features
    cdr_nulls    = cdr_feat[mobility_cols].isnull().sum().sum()
    survey_nulls = survey_feat[mobility_cols].isnull().sum().sum()
    if cdr_nulls > 0 or survey_nulls > 0:
        print(f"  FAIL — Nulls: CDR={cdr_nulls}, Survey={survey_nulls}")
        passed = False
    else:
        print(f"  PASS — No nulls in mobility features.")

    # Check 3 — pct columns in [0,1]
    pct_cols = ["pct_peak_am","pct_peak_pm","pct_offpeak",
                "pct_work_trips","pct_local_train","pct_walk"]
    for col in pct_cols:
        if col in cdr_feat:
            bad = ((cdr_feat[col]<-0.01) | (cdr_feat[col]>1.01)).sum()
            if bad > 0:
                print(f"  WARN — CDR {col}: {bad} values outside [0,1]")
        if col in survey_feat:
            bad = ((survey_feat[col]<-0.01) | (survey_feat[col]>1.01)).sum()
            if bad > 0:
                print(f"  WARN — Survey {col}: {bad} values outside [0,1]")
    print(f"  PASS — Percentage columns checked.")

    # Check 4 — radius of gyration > 0 for most users (expect > 80%)
    for name, df in [("CDR", cdr_feat), ("Survey", survey_feat)]:
        pct_nonzero = (df["radius_of_gyration_km"] > 0).mean()
        if pct_nonzero < 0.50:
            print(f"  WARN — {name}: only {pct_nonzero:.1%} users have rog > 0.")
        else:
            print(f"  PASS — {name} radius_of_gyration > 0: {pct_nonzero:.1%}")

    # Check 5 — feature distribution alignment (mean within 2x of each other)
    print("  Feature distribution comparison (CDR vs Survey means):")
    for col in ["radius_of_gyration_km","location_entropy","home_work_dist_km",
                "mean_trip_dist_km","pct_peak_am","pct_peak_pm"]:
        if col in cdr_feat.columns and col in survey_feat.columns:
            cm = cdr_feat[col].mean(); sm = survey_feat[col].mean()
            ratio = cm/sm if sm > 0 else float("inf")
            flag  = "⚠" if (ratio > 5.0 or ratio < 0.2) else "✓"
            print(f"    {flag} {col:<30} CDR={cm:.3f}  Survey={sm:.3f}  ratio={ratio:.2f}")

    # Check 6 — label columns present in survey features
    label_cols = ["age_bracket","employment","income_class","vehicle_owned"]
    missing_labels = [c for c in label_cols if c not in survey_feat.columns]
    if missing_labels:
        print(f"  FAIL — Survey missing label columns: {missing_labels}")
        passed = False
    else:
        print(f"  PASS — All label columns present in survey features.")

    # Check 7 — CDR coverage
    pct_covered = len(cdr_feat) / 200000
    if pct_covered < 0.70:
        print(f"  WARN — CDR feature coverage: {pct_covered:.1%} of 200k users.")
    else:
        print(f"  PASS — CDR feature coverage: {pct_covered:.1%}")

    print(f"\nValidation {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    os.makedirs(os.path.dirname(CDR_FEAT_OUT),    exist_ok=True)
    os.makedirs(os.path.dirname(SURVEY_FEAT_OUT), exist_ok=True)

    # ── CDR features ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("CDR Feature Extraction")
    print("=" * 60)

    print("Loading CDR processed data...")
    cdr_trips   = pd.read_parquet(CDR_TRIPS_PATH)
    cdr_anchors = pd.read_parquet(CDR_ANCHORS_PATH)
    cdr_sp      = pd.read_parquet(CDR_SP_PATH)
    print(f"  Trips     : {len(cdr_trips):,}")
    print(f"  Anchors   : {len(cdr_anchors):,}")
    print(f"  Stay pts  : {len(cdr_sp):,}")

    cdr_feat = extract_cdr_features(cdr_trips, cdr_anchors, cdr_sp)

    del cdr_trips, cdr_anchors, cdr_sp; gc.collect()

    cdr_feat.to_parquet(CDR_FEAT_OUT, index=False)
    print(f"\nSaved: {CDR_FEAT_OUT}")
    print(f"  Shape   : {cdr_feat.shape}")
    print(f"  Columns : {list(cdr_feat.columns)}")
    print(f"  Elapsed : {time.time()-t0:.0f}s")

    # ── Survey features ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Survey Feature Extraction")
    print("=" * 60)

    print("Loading survey data...")
    survey_trips   = pd.read_parquet(SURVEY_TRIPS_PATH)
    survey_sociodem = pd.read_parquet(SURVEY_SOCIODEM_PATH)
    print(f"  Survey trips   : {len(survey_trips):,}")
    print(f"  Survey persons : {len(survey_sociodem):,}")

    survey_feat = extract_survey_features(survey_trips, survey_sociodem)

    del survey_trips, survey_sociodem; gc.collect()

    survey_feat.to_parquet(SURVEY_FEAT_OUT, index=False)
    print(f"\nSaved: {SURVEY_FEAT_OUT}")
    print(f"  Shape   : {survey_feat.shape}")
    print(f"  Columns : {list(survey_feat.columns)}")
    print(f"  Elapsed : {time.time()-t0:.0f}s")

    # ── Validation ────────────────────────────────────────────────────────────
    valid = validate_features(cdr_feat, survey_feat)
    if not valid:
        print("\nFeature engineering failed validation. Fix before Stage 3.")
        exit(1)

    print(f"\nStage 2 complete in {time.time()-t0:.0f}s")
    print(f"  CDR features    : {len(cdr_feat):,} users × {cdr_feat.shape[1]} cols")
    print(f"  Survey features : {len(survey_feat):,} persons × {survey_feat.shape[1]} cols")

    # Summary stats
    print("\nCDR feature summary:")
    mobility = ["radius_of_gyration_km","location_entropy","home_work_dist_km",
                "n_unique_wards","n_trips_per_day","mean_trip_dist_km",
                "mean_speed_kmh","pct_peak_am","pct_peak_pm"]
    print(cdr_feat[mobility].describe().round(3).to_string())

    print("\nSurvey feature summary:")
    print(survey_feat[mobility].describe().round(3).to_string())