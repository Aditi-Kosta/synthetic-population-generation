# mumbai-synth-v2/processing/cdr_processor.py
"""
Stage 1 — CDR Processing
Converts raw CDR ping records into structured travel patterns per user.

Steps:
  1. Stay point detection — vectorized NumPy per user, no Numba/joblib
  2. Home/work anchor inference — sklearn DBSCAN (CPU)
  3. Trip extraction — vectorized NumPy
  4. Mode inference — vectorized rule-based

Outputs:
  data/processed/cdr_trips/cdr_stay_points.parquet
  data/processed/cdr_trips/cdr_anchors.parquet
  data/processed/cdr_trips/cdr_trips.parquet
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ─────────────────────────────────────────────────────────────────────

CDR_PATH        = "../data/raw/cdr/mumbai_cdr.parquet"
OUT_DIR         = "../data/processed/cdr_trips"

DIST_THRESH_KM  = 0.5    # max radius for a stay point cluster
TIME_THRESH_MIN = 20.0   # min dwell time to count as stay point
DBSCAN_EPS_KM   = 0.5    # DBSCAN neighbourhood radius
DBSCAN_MIN_PTS  = 3      # DBSCAN min samples (lowered: ~175 pings/user/5days)
NIGHT_HOURS     = set(range(0, 6)) | {22, 23}
DAY_HOURS       = set(range(9, 17))

WESTERN_LINE_LON = (72.810, 72.870)
CENTRAL_LINE_LON = (72.830, 72.960)
HARBOUR_LINE_LON = (72.830, 73.020)
METRO_LINE1_LAT  = (19.086, 19.135)

CHUNK_SIZE      = 10000  # users per processing chunk


# ── Haversine (vectorized numpy) ───────────────────────────────────────────────

def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine — inputs are numpy arrays."""
    R = 6371.0
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    dlat = la2 - la1; dlon = lo2 - lo1
    a = np.sin(dlat*0.5)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon*0.5)**2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_scalar(lat1, lon1, lat2, lon2):
    return haversine_vec(
        np.array([lat1]), np.array([lon1]),
        np.array([lat2]), np.array([lon2])
    )[0]


# ── Step 1: Stay Point Detection ───────────────────────────────────────────────

def detect_stay_points_user(lats, lons, ts_sec, hours):
    """
    Stay point detection for one user — pure Python/numpy, no Numba.
    Zheng et al. (2009): collect consecutive pings within DIST_THRESH_KM,
    accept as stay point if dwell >= TIME_THRESH_MIN.

    Returns list of dicts (one per stay point).
    """
    n   = len(lats)
    sps = []
    i   = 0

    while i < n - 1:
        j        = i + 1
        mlat     = lats[i]
        mlon     = lons[i]

        while j < n:
            dist = haversine_scalar(mlat, mlon, lats[j], lons[j])
            if dist > DIST_THRESH_KM:
                break
            cnt  = j - i + 1
            mlat = (mlat * (cnt-1) + lats[j]) / cnt
            mlon = (mlon * (cnt-1) + lons[j]) / cnt
            j   += 1

        dwell = (ts_sec[j-1] - ts_sec[i]) / 60.0
        if dwell >= TIME_THRESH_MIN:
            sps.append({
                "arrive_ts":    int(ts_sec[i]),
                "depart_ts":    int(ts_sec[j-1]),
                "stay_lat":     round(float(mlat), 6),
                "stay_lon":     round(float(mlon), 6),
                "duration_min": round(float(dwell), 1),
                "arrive_hour":  int(hours[i]),
                "n_pings":      j - i,
            })
            i = j
        else:
            i += 1

    return sps


def detect_stay_points_all(cdr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect stay points for all 200k users, processed in chunks.
    Pure Python loop per user — fast enough (~3-5 min for 200k users
    at ~175 pings/user since the inner loop is short per user).
    """
    print("  Preparing CDR data...")
    cdr_df = cdr_df.copy()
    cdr_df["ts_sec"] = cdr_df["timestamp"].astype(np.int64) // 10**6  # us -> seconds
    cdr_df["hour"]   = cdr_df["timestamp"].dt.hour
    cdr_df = cdr_df.sort_values(["user_id", "timestamp"])

    # Pre-group into numpy arrays per user — avoids repeated pandas filtering
    print("  Pre-grouping by user...")
    groups = {}
    for col in ["tower_lat", "tower_lon", "ts_sec", "hour"]:
        cdr_df[col] = cdr_df[col].astype(np.float64 if col != "hour" else np.int32)

    uid_arr  = cdr_df["user_id"].values
    lat_arr  = cdr_df["tower_lat"].values
    lon_arr  = cdr_df["tower_lon"].values
    ts_arr   = cdr_df["ts_sec"].values.astype(np.int64)
    hr_arr   = cdr_df["hour"].values.astype(np.int32)

    # Find group boundaries
    unique_uids, counts = np.unique(uid_arr, return_counts=True)
    n_users = len(unique_uids)
    print(f"  Processing {n_users:,} users...")

    all_records = []
    boundaries  = np.concatenate([[0], np.cumsum(counts)])

    for chunk_start in range(0, n_users, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_users)
        chunk_sps = []

        for ui in range(chunk_start, chunk_end):
            s = boundaries[ui]; e = boundaries[ui+1]
            uid  = unique_uids[ui]
            lats = lat_arr[s:e]; lons = lon_arr[s:e]
            ts   = ts_arr[s:e];  hrs  = hr_arr[s:e]
            sps  = detect_stay_points_user(lats, lons, ts, hrs)
            for sp in sps:
                sp["user_id"] = uid
                chunk_sps.append(sp)

        all_records.extend(chunk_sps)
        print(f"  Progress: {chunk_end:,}/{n_users:,} users — "
              f"{len(all_records):,} stay points")

    if not all_records:
        print("  WARNING: 0 stay points detected. Check DIST_THRESH_KM/TIME_THRESH_MIN.")
        return pd.DataFrame(columns=[
            "user_id","arrive_ts","depart_ts","stay_lat","stay_lon",
            "duration_min","arrive_hour","n_pings"
        ])

    df = pd.DataFrame(all_records)
    df["arrive_ts"] = pd.to_datetime(df["arrive_ts"], unit="s")
    df["depart_ts"] = pd.to_datetime(df["depart_ts"], unit="s")
    return df


# ── Step 2: Anchor Inference ───────────────────────────────────────────────────

def infer_anchors(stay_points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer home/work anchors per user via DBSCAN.
    Home  = dominant cluster in night hours (22-05).
    Work  = dominant cluster in day hours   (09-16).
    Fallback: mean of all stay points when no clean cluster found.
    """
    anchors    = []
    eps_rad    = DBSCAN_EPS_KM / 6371.0
    user_groups = list(stay_points_df.groupby("user_id"))
    n_users    = len(user_groups)
    print(f"  Inferring anchors for {n_users:,} users...")

    for idx, (user_id, group) in enumerate(user_groups):
        if idx % 20000 == 0:
            print(f"  Anchor inference: {idx:,}/{n_users:,}")

        result = {
            "user_id":         user_id,
            "home_lat":        None, "home_lon": None,
            "work_lat":        None, "work_lon": None,
            "home_confidence": 0.0,
            "work_confidence": 0.0,
        }

        mean_lat = group["stay_lat"].mean()
        mean_lon = group["stay_lon"].mean()

        if len(group) < DBSCAN_MIN_PTS:
            result["home_lat"] = mean_lat; result["home_lon"] = mean_lon
            result["work_lat"] = mean_lat; result["work_lon"] = mean_lon
            anchors.append(result)
            continue

        coords_rad = np.radians(group[["stay_lat","stay_lon"]].values)
        labels = DBSCAN(
            eps=eps_rad, min_samples=DBSCAN_MIN_PTS,
            algorithm="ball_tree", metric="haversine",
        ).fit_predict(coords_rad)

        group = group.copy()
        group["cluster"] = labels

        # Home — dominant night cluster
        night = group[group["arrive_hour"].isin(NIGHT_HOURS)]
        nv    = night[night["cluster"] >= 0]
        if len(nv) > 0:
            hc = nv["cluster"].value_counts().idxmax()
            hp = group[group["cluster"] == hc]
            result["home_lat"]        = hp["stay_lat"].mean()
            result["home_lon"]        = hp["stay_lon"].mean()
            result["home_confidence"] = len(nv) / max(len(night), 1)

        # Work — dominant day cluster, must differ from home cluster
        # Work — dominant day cluster, prefer different cluster than home
        day = group[group["arrive_hour"].isin(DAY_HOURS)]
        dv  = day[day["cluster"] >= 0]
        if len(dv) > 0:
            night_valid = group[
                group["arrive_hour"].isin(NIGHT_HOURS) & (group["cluster"] >= 0)
            ]
            home_cluster = (
                night_valid["cluster"].value_counts().idxmax()
                if len(night_valid) > 0 else -99
            )
            work_counts = dv["cluster"].value_counts()
            for wc in work_counts.index:
                if wc != home_cluster:
                    wp = group[group["cluster"] == wc]
                    result["work_lat"]        = wp["stay_lat"].mean()
                    result["work_lon"]        = wp["stay_lon"].mean()
                    result["work_confidence"] = len(dv) / max(len(day), 1)
                    break
            if result["work_lat"] is None:
                wc = work_counts.index[0]
                wp = group[group["cluster"] == wc]
                result["work_lat"]        = wp["stay_lat"].mean()
                result["work_lon"]        = wp["stay_lon"].mean()
                result["work_confidence"] = len(dv) / max(len(day), 1)

        # Fallbacks
        if result["home_lat"] is None:
            result["home_lat"] = mean_lat; result["home_lon"] = mean_lon
        if result["work_lat"] is None:
            result["work_lat"] = mean_lat; result["work_lon"] = mean_lon

        anchors.append(result)

    return pd.DataFrame(anchors)


# ── Step 3: Trip Extraction ────────────────────────────────────────────────────

def extract_trips_vectorized(stay_points_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized trip extraction from consecutive stay point pairs.
    """
    sp = stay_points_df.sort_values(
        ["user_id","arrive_ts"]
    ).reset_index(drop=True)

    sp_next   = sp.shift(-1)
    same_user = sp["user_id"] == sp_next["user_id"]
    pairs     = sp[same_user].copy()
    pairs_next = sp_next[same_user].copy()

    lat1 = pairs["stay_lat"].values.astype(float)
    lon1 = pairs["stay_lon"].values.astype(float)
    lat2 = pairs_next["stay_lat"].values.astype(float)
    lon2 = pairs_next["stay_lon"].values.astype(float)

    dist_km   = haversine_vec(lat1, lon1, lat2, lon2)
    depart_ts = pd.to_datetime(pairs["depart_ts"].values)
    arrive_ts = pd.to_datetime(pairs_next["arrive_ts"].values)

    travel_min = (arrive_ts - depart_ts).total_seconds() / 60.0
    speed_kmh  = np.where(travel_min > 0, dist_km / travel_min * 60, 0.0)

    # departure_hour = hour user LEFT the origin stay point
    departure_hour = pd.DatetimeIndex(pairs["depart_ts"].values).hour

    valid = (dist_km >= 0.3) & (travel_min >= 2) & (travel_min <= 300)

    trips_df = pd.DataFrame({
        "user_id":          pairs["user_id"].values[valid],
        "origin_lat":       lat1[valid],
        "origin_lon":       lon1[valid],
        "dest_lat":         lat2[valid],
        "dest_lon":         lon2[valid],
        "depart_ts":        depart_ts[valid],
        "arrive_ts":        arrive_ts[valid],
        "departure_hour":   departure_hour[valid],
        "travel_time_mins": np.round(travel_min[valid], 1),
        "trip_dist_km":     np.round(dist_km[valid], 3),
        "speed_kmh":        np.round(speed_kmh[valid], 2),
    })

    print(f"  Extracted {len(trips_df):,} trips from "
          f"{trips_df['user_id'].nunique():,} users")
    return trips_df


# ── Step 4: Mode Inference ────────────────────────────────────────────────────

def infer_modes_vectorized(trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized mode inference. Priority order:
      walk        : speed <= 7 AND dist <= 2 km
      metro       : speed >= 28 AND near metro lat band
      local_train : speed >= 28 AND near rail corridor
      auto        : speed 15-24 km/h
      best_bus    : speed 8-15 km/h
      car_2w      : speed 24-40 km/h (not rail/metro)
      walk        : catch-all low speed
    Speed thresholds calibrated to Mumbai modes:
      local_train ~35 km/h avg, metro ~40, bus ~18, auto ~22, car ~25
    """
    trips_df = trips_df.copy()
    speed    = trips_df["speed_kmh"].values
    dist     = trips_df["trip_dist_km"].values
    o_lat    = trips_df["origin_lat"].values
    o_lon    = trips_df["origin_lon"].values
    d_lat    = trips_df["dest_lat"].values
    d_lon    = trips_df["dest_lon"].values

    near_rail = (
        (((o_lon >= WESTERN_LINE_LON[0]) & (o_lon <= WESTERN_LINE_LON[1])) |
         ((o_lon >= CENTRAL_LINE_LON[0]) & (o_lon <= CENTRAL_LINE_LON[1])) |
         ((o_lon >= HARBOUR_LINE_LON[0]) & (o_lon <= HARBOUR_LINE_LON[1]))) &
        (((d_lon >= WESTERN_LINE_LON[0]) & (d_lon <= WESTERN_LINE_LON[1])) |
         ((d_lon >= CENTRAL_LINE_LON[0]) & (d_lon <= CENTRAL_LINE_LON[1])) |
         ((d_lon >= HARBOUR_LINE_LON[0]) & (d_lon <= HARBOUR_LINE_LON[1])))
    )
    near_metro = (
        (o_lat >= METRO_LINE1_LAT[0]) & (o_lat <= METRO_LINE1_LAT[1]) &
        (d_lat >= METRO_LINE1_LAT[0]) & (d_lat <= METRO_LINE1_LAT[1])
    )

    mode = np.where(
        (speed <= 7) & (dist <= 2),           "walk",
        np.where(
            (speed >= 28) & near_metro,       "metro",
            np.where(
                (speed >= 28) & near_rail,    "local_train",
                np.where(
                    speed >= 28,              "local_train",
                    np.where(
                        speed < 8,            "walk",
                        np.where(
                            speed <= 15,      "best_bus",
                            np.where(
                                speed <= 24,  "auto",
                                              "car_2w",
                            )
                        )
                    )
                )
            )
        )
    )

    trips_df["mode_inferred"] = mode
    print("  Mode distribution:")
    for m, share in pd.Series(mode).value_counts(normalize=True).items():
        print(f"    {m:<15} {share:.1%}")
    return trips_df


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_processed(stay_points_df, anchors_df_raw, trips_df, n_users):
    print("\nRunning CDR processing validation...")
    passed = True

    sp_users = stay_points_df["user_id"].nunique()
    coverage = sp_users / n_users
    if coverage < 0.80:
        print(f"  FAIL — Stay point coverage {coverage:.1%} < 80%.")
        passed = False
    else:
        print(f"  PASS — Stay point coverage: {coverage:.1%}")

    null_anchors = anchors_df_raw[["home_lat","work_lat"]].isnull().sum().sum()
    if null_anchors > 0:
        print(f"  WARN — {null_anchors:,} null anchor coords (will be dropped).")
    else:
        print(f"  PASS — All anchors have valid coordinates.")

    tpu  = trips_df.groupby("user_id").size()
    mean = tpu.mean()
    if not (1.0 <= mean <= 10.0):
        print(f"  WARN — Mean trips/user: {mean:.2f} (expected 1-10).")
    else:
        print(f"  PASS — Mean trips/user: {mean:.2f}")

    neg = (trips_df["travel_time_mins"] < 0).sum()
    if neg > 0:
        print(f"  FAIL — {neg:,} negative travel times."); passed = False
    else:
        print(f"  PASS — No negative travel times.")

    max_speed = trips_df["speed_kmh"].max()
    if max_speed > 150:
        print(f"  WARN — Max speed {max_speed:.1f} km/h.")
    else:
        print(f"  PASS — Max speed {max_speed:.1f} km/h.")

    coord_nulls = trips_df[
        ["origin_lat","origin_lon","dest_lat","dest_lon"]
    ].isnull().sum().sum()
    if coord_nulls > 0:
        print(f"  FAIL — {coord_nulls:,} null trip coords."); passed = False
    else:
        print(f"  PASS — No null trip coordinates.")

    bad_hours = (~trips_df["departure_hour"].between(0, 23)).sum()
    if bad_hours > 0:
        print(f"  FAIL — {bad_hours:,} departure_hour outside 0-23."); passed = False
    else:
        print(f"  PASS — All departure_hour values in 0-23.")

    print(f"\nValidation {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CDR data...")
    cdr_df = pd.read_parquet(CDR_PATH)
    cdr_df["timestamp"] = pd.to_datetime(cdr_df["timestamp"])
    n_users = cdr_df["user_id"].nunique()
    print(f"Loaded: {cdr_df.shape} — {n_users:,} users")

    print("\nStep 1: Stay point detection...")
    stay_points_df = detect_stay_points_all(cdr_df)
    print(f"Total stay points: {len(stay_points_df):,}")

    if len(stay_points_df) == 0:
        print("ERROR: 0 stay points. Cannot proceed.")
        exit(1)

    stay_points_df.to_parquet(
        os.path.join(OUT_DIR, "cdr_stay_points.parquet"), index=False
    )
    print(f"Saved: cdr_stay_points.parquet")

    del cdr_df; gc.collect()
    print(f"CDR freed. Elapsed: {time.time()-t0:.0f}s")

    print("\nStep 2: Anchor inference (DBSCAN)...")
    anchors_df_raw = infer_anchors(stay_points_df)
    print(f"Anchors inferred: {len(anchors_df_raw):,} users")

    print("\nStep 3: Trip extraction...")
    trips_df = extract_trips_vectorized(stay_points_df)

    print("\nStep 4: Mode inference...")
    trips_df = infer_modes_vectorized(trips_df)

    valid = validate_processed(stay_points_df, anchors_df_raw, trips_df, n_users)

    anchors_df = anchors_df_raw.dropna(subset=["home_lat","work_lat"])
    print(f"Anchors after dropna: {len(anchors_df):,} "
          f"({len(anchors_df_raw)-len(anchors_df):,} dropped)")

    anchors_df.to_parquet(os.path.join(OUT_DIR,"cdr_anchors.parquet"), index=False)
    trips_df.to_parquet(os.path.join(OUT_DIR,"cdr_trips.parquet"), index=False)
    print(f"Saved: cdr_anchors.parquet, cdr_trips.parquet")

    if not valid:
        print("\nValidation failed. Fix before proceeding.")
        exit(1)

    print(f"\nProcessing complete in {time.time()-t0:.0f}s")
    print(f"  Stay points : {len(stay_points_df):,}")
    print(f"  Anchors     : {len(anchors_df):,}")
    print(f"  Trips       : {len(trips_df):,}")
    print(f"  Columns     : {list(trips_df.columns)}")