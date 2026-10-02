# mumbai-synth-v2/generation/generate_cdr.py
"""
Stage 0 — CDR Generation (Fixed)
Generates synthetic but statistically realistic Call Detail Records
for Mumbai's BMC area.

Fix: Commute period tower pool restricted to corridor wards between
home and work instead of all towers citywide. This ensures pings
during commute are spatially coherent and stay point detection works.

Output: data/raw/cdr/mumbai_cdr.parquet
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from math import radians, sin, cos, sqrt, asin
import os
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────────────────────────

WARD_POPULATIONS = {
    "A": 193520,   "B": 133142,   "C": 173800,   "D": 362812,
    "E": 411366,   "F/S": 377567, "F/N": 553355, "G/S": 395115,
    "G/N": 626578, "H/E": 582857, "H/W": 321722, "K/E": 861762,
    "K/W": 783107, "P/S": 484816, "P/N": 984647, "R/S": 723007,
    "R/C": 588006, "R/N": 451199, "L":   943703,  "M/E": 844854,
    "M/W": 430829, "N":   651487,  "S":   777977,  "T":   357161,
}

WARD_CENTROIDS = {
    "A":   (18.9220, 72.8347), "B":   (18.9480, 72.8368),
    "C":   (18.9620, 72.8200), "D":   (18.9720, 72.8150),
    "E":   (18.9820, 72.8350), "F/S": (19.0050, 72.8350),
    "F/N": (19.0350, 72.8420), "G/S": (19.0130, 72.8180),
    "G/N": (19.0280, 72.8220), "H/E": (19.0720, 72.8590),
    "H/W": (19.0650, 72.8350), "K/E": (19.1050, 72.8650),
    "K/W": (19.0950, 72.8350), "P/S": (19.1580, 72.8490),
    "P/N": (19.1900, 72.8480), "R/S": (19.2050, 72.8600),
    "R/C": (19.2200, 72.8680), "R/N": (19.2350, 72.8550),
    "L":   (19.0750, 72.8900), "M/E": (19.0550, 72.9050),
    "M/W": (19.0650, 72.8980), "N":   (19.1150, 72.9100),
    "S":   (19.1450, 72.9250), "T":   (19.1750, 72.9600),
}

ISLAND_CITY_WARDS = {"A", "B", "C", "D", "E", "F/S", "F/N", "G/S", "G/N"}

TOTAL_USERS     = 200000
TOWERS_PER_WARD = 40
N_DAYS          = 5
GEOJSON_PATH    = "../data/raw/mumbai_wards.geojson"

HOURLY_WEIGHTS = np.array([
     8,  5,  3,  3,  6, 16,
    44, 82, 98, 74, 58, 54,
    56, 50, 52, 58, 64, 88,
    95, 82, 66, 46, 30, 16,
], dtype=float)
HOURLY_WEIGHTS /= HOURLY_WEIGHTS.sum()

# ── Ward adjacency map ─────────────────────────────────────────────────────────

WARD_ADJACENCY = {
    "A":   ["A", "B", "C", "D"],
    "B":   ["A", "B", "C", "E"],
    "C":   ["A", "B", "C", "D", "E"],
    "D":   ["C", "D", "E", "F/S", "G/S"],
    "E":   ["B", "C", "D", "E", "F/N", "F/S"],
    "F/S": ["D", "E", "F/S", "F/N", "G/S", "G/N"],
    "F/N": ["E", "F/S", "F/N", "G/N", "H/E", "L"],
    "G/S": ["D", "F/S", "G/S", "G/N"],
    "G/N": ["F/S", "F/N", "G/S", "G/N", "H/W", "H/E"],
    "H/E": ["F/N", "G/N", "H/E", "H/W", "K/E", "L"],
    "H/W": ["G/N", "H/E", "H/W", "K/W"],
    "K/E": ["H/E", "K/E", "K/W", "L", "N"],
    "K/W": ["H/W", "K/E", "K/W", "P/S"],
    "P/S": ["K/W", "P/S", "P/N", "R/S"],
    "P/N": ["P/S", "P/N", "R/S", "R/C"],
    "R/S": ["P/S", "P/N", "R/S", "R/C", "R/N"],
    "R/C": ["P/N", "R/S", "R/C", "R/N"],
    "R/N": ["R/S", "R/C", "R/N"],
    "L":   ["F/N", "H/E", "K/E", "L", "M/E", "M/W", "N"],
    "M/E": ["L", "M/E", "M/W", "N"],
    "M/W": ["L", "M/E", "M/W"],
    "N":   ["K/E", "L", "M/E", "N", "S"],
    "S":   ["N", "S", "T"],
    "T":   ["S", "T"],
}


# NOTE: get_corridor_wards() / ward-adjacency-based corridor pools removed.
# Sampling uniformly from a ward-level pool during commute — even a
# restricted one — still lets consecutive pings jump anywhere within that
# pool (5-11km for adjacent wards). Replaced below with straight-line
# interpolation + nearest-tower snapping, so commute pings actually trace
# a path and consecutive pings are close in space AND time.

# ── Haversine ──────────────────────────────────────────────────────────────────

def haversine_simple(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


# ── Tower Grid Generation ──────────────────────────────────────────────────────

def generate_tower_grid(
    ward_boundaries: gpd.GeoDataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    total_pop = sum(WARD_POPULATIONS.values())
    towers    = []
    tower_id  = 0

    for ward_id, pop in WARD_POPULATIONS.items():
        n_towers = max(5, int(TOWERS_PER_WARD * pop / total_pop * 24))

        if ward_id in ward_boundaries.index:
            polygon  = ward_boundaries.loc[ward_id, "geometry"]
            minx, miny, maxx, maxy = polygon.bounds
            placed   = 0
            attempts = 0
            while placed < n_towers and attempts < n_towers * 50:
                x = rng.uniform(minx, maxx)
                y = rng.uniform(miny, maxy)
                if polygon.contains(Point(x, y)):
                    towers.append({
                        "tower_id":  f"T_{tower_id:05d}",
                        "ward_id":   ward_id,
                        "tower_lat": round(y, 6),
                        "tower_lon": round(x, 6),
                    })
                    tower_id += 1
                    placed   += 1
                attempts += 1
        else:
            clat, clon = WARD_CENTROIDS[ward_id]
            for _ in range(n_towers):
                towers.append({
                    "tower_id":  f"T_{tower_id:05d}",
                    "ward_id":   ward_id,
                    "tower_lat": round(clat + rng.normal(0, 0.005), 6),
                    "tower_lon": round(clon + rng.normal(0, 0.005), 6),
                })
                tower_id += 1

    df = pd.DataFrame(towers)
    print(f"Generated {len(df):,} towers across {df['ward_id'].nunique()} wards.")
    return df


# ── User Home/Work Assignment ──────────────────────────────────────────────────

def assign_users(
    tower_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    total_pop  = sum(WARD_POPULATIONS.values())
    wards      = list(WARD_POPULATIONS.keys())
    home_probs = np.array(
        [WARD_POPULATIONS[w] for w in wards], dtype=float
    )
    home_probs /= home_probs.sum()

    COMMERCIAL_WEIGHT = {
        "A": 2.5, "B": 2.0, "C": 1.8, "D": 1.5, "E": 1.3,
        "F/S": 1.5, "F/N": 1.2, "G/S": 1.8, "G/N": 1.4,
        "H/E": 2.0, "H/W": 1.8, "K/E": 1.3, "K/W": 1.2,
        "P/S": 0.9, "P/N": 0.8, "R/S": 0.7, "R/C": 0.7,
        "R/N": 0.6, "L":   1.1, "M/E": 0.9, "M/W": 0.8,
        "N":   0.9, "S":   0.7, "T":   0.6,
    }
    work_probs = np.array([
        WARD_POPULATIONS[w] * COMMERCIAL_WEIGHT[w] for w in wards
    ], dtype=float)
    work_probs /= work_probs.sum()

    home_wards = rng.choice(wards, size=TOTAL_USERS, p=home_probs)
    work_wards = rng.choice(wards, size=TOTAL_USERS, p=work_probs)

    users = []
    for uid, (hw, ww) in enumerate(zip(home_wards, work_wards)):
        home_towers = tower_df[tower_df["ward_id"] == hw]
        work_towers = tower_df[tower_df["ward_id"] == ww]
        if len(home_towers) == 0 or len(work_towers) == 0:
            continue
        home_tower = home_towers.iloc[rng.integers(len(home_towers))]
        work_tower = work_towers.iloc[rng.integers(len(work_towers))]
        users.append({
            "user_id":       f"U_{uid:07d}",
            "home_ward":     hw,
            "work_ward":     ww,
            "home_tower_id": home_tower["tower_id"],
            "work_tower_id": work_tower["tower_id"],
            "home_lat":      home_tower["tower_lat"],
            "home_lon":      home_tower["tower_lon"],
            "work_lat":      work_tower["tower_lat"],
            "work_lon":      work_tower["tower_lon"],
        })

    df = pd.DataFrame(users)
    print(f"Assigned {len(df):,} users with home/work towers.")
    return df


# ── CDR Ping Generation (Fixed + Vectorized) ─────────────────────────────────

def generate_user_pings(
    user_id: str,
    home_ward: str,
    work_ward: str,
    home_lat: float, home_lon: float,
    work_lat: float, work_lon: float,
    tower_id_arr: np.ndarray, tower_lat_arr: np.ndarray,
    tower_lon_arr: np.ndarray, tower_ward_arr: np.ndarray,
    tower_kdtree: cKDTree,
    rng: np.random.Generator,
    base_date: datetime,
) -> dict:
    """
    Generate realistic CDR pings for one user across N_DAYS weekdays.
    Fully vectorized — no per-ping Python-level loop, batched KDTree
    queries per user instead of per-ping queries.

    Fix (v4): Tower selection now uses a distance threshold instead of
    k-nearest. Home/work pings are drawn only from towers within
    anchor_radius_km (0.5km) of the user's home/work coordinate.
    Commute pings snap to the nearest tower within commute_radius_km
    (1.0km) of the interpolated position. Both fall back to the single
    nearest tower if no towers exist within the radius. This guarantees
    consecutive same-phase pings stay within ~1km of each other
    regardless of ward size or tower density.
    """
    n_pings_day = int(rng.integers(20, 51))
    n_total     = n_pings_day * N_DAYS

    day_idx = np.repeat(np.arange(N_DAYS), n_pings_day)

    hour   = rng.choice(24, size=n_total, p=HOURLY_WEIGHTS)
    minute = rng.integers(0, 60, size=n_total)
    second = rng.integers(0, 60, size=n_total)
    hour_val = hour + minute / 60.0

    dist_km     = haversine_simple(home_lat, home_lon, work_lat, work_lon)
    commute_hrs = dist_km / 25.0

    depart_hour_d = rng.normal(8.0, 0.6, size=N_DAYS)
    arrive_work_d = depart_hour_d + np.maximum(
        0.25, commute_hrs + rng.normal(0, 0.15, size=N_DAYS)
    )
    depart_work_d = rng.normal(18.0, 1.2, size=N_DAYS)
    arrive_home_d = depart_work_d + np.maximum(
        0.25, commute_hrs + rng.normal(0, 0.15, size=N_DAYS)
    )

    depart_hour = depart_hour_d[day_idx]
    arrive_work = arrive_work_d[day_idx]
    depart_work = depart_work_d[day_idx]
    arrive_home = arrive_home_d[day_idx]

    morning_mask = (hour_val >= depart_hour) & (hour_val < arrive_work)
    evening_mask = (hour_val >= depart_work) & (hour_val < arrive_home)
    work_mask    = (hour_val >= arrive_work) & (hour_val < depart_work)
    home_mask    = ~(morning_mask | evening_mask | work_mask)

    out_tower_id = np.empty(n_total, dtype=object)
    out_lat      = np.empty(n_total, dtype=float)
    out_lon      = np.empty(n_total, dtype=float)
    out_ward     = np.empty(n_total, dtype=object)

    # ── Build commute tower sequence (pre-computed, ordered waypoints) ─────
    # Divide the home→work line into N_SEGMENTS equal segments. For each
    # segment midpoint find the nearest tower. This gives an ORDERED list
    # of towers along the commute path so that consecutive commute pings
    # always use adjacent entries — guaranteed spatial coherence regardless
    # of tower density or ward size.
    N_SEGMENTS = max(10, int(dist_km / 0.4) + 1)  # ~1 segment per 0.4km; min 10 ensures short commutes have fine coverage
    seg_fracs  = (np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS  # midpoints

    seg_lats = home_lat + seg_fracs * (work_lat - home_lat)
    seg_lons = home_lon + seg_fracs * (work_lon - home_lon)
    seg_points = np.column_stack([seg_lats, seg_lons])
    _, seg_idx = tower_kdtree.query(seg_points)   # nearest tower per segment

    # Deduplicate consecutive identical tower indices (multiple segments may
    # map to the same tower) while preserving order
    commute_seq = [seg_idx[0]]
    for ti in seg_idx[1:]:
        if ti != commute_seq[-1]:
            commute_seq.append(ti)
    commute_seq = np.array(commute_seq)           # ordered tower indices H→W
    n_seq       = len(commute_seq)

    # ── Home pings: always the single nearest tower to home coord ──────────
    _, home_nn = tower_kdtree.query([home_lat, home_lon], k=1)
    home_nn = int(home_nn)

    n_home = home_mask.sum()
    if n_home:
        out_tower_id[home_mask] = tower_id_arr[home_nn]
        out_lat[home_mask]      = tower_lat_arr[home_nn]
        out_lon[home_mask]      = tower_lon_arr[home_nn]
        out_ward[home_mask]     = tower_ward_arr[home_nn]

    # ── Work pings: always the single nearest tower to work coord ──────────
    _, work_nn = tower_kdtree.query([work_lat, work_lon], k=1)
    work_nn = int(work_nn)

    n_work = work_mask.sum()
    if n_work:
        out_tower_id[work_mask] = tower_id_arr[work_nn]
        out_lat[work_mask]      = tower_lat_arr[work_nn]
        out_lon[work_mask]      = tower_lon_arr[work_nn]
        out_ward[work_mask]     = tower_ward_arr[work_nn]

    # ── Commute pings: map frac → segment index → tower ────────────────────
    # frac=0 → home end of commute_seq; frac=1 → work end.
    commute_mask = morning_mask | evening_mask
    n_commute    = commute_mask.sum()
    if n_commute:
        frac = np.empty(n_total, dtype=float)
        frac[morning_mask] = np.clip(
            (hour_val[morning_mask] - depart_hour[morning_mask])
            / np.maximum(1e-6,
                arrive_work[morning_mask] - depart_hour[morning_mask]),
            0, 1,
        )
        frac[evening_mask] = np.clip(
            (hour_val[evening_mask] - depart_work[evening_mask])
            / np.maximum(1e-6,
                arrive_home[evening_mask] - depart_work[evening_mask]),
            0, 1,
        )

        # Map frac to an index in commute_seq (forward for morning, reverse
        # for evening so the path always goes home↔work in the right direction)
        seq_idx_morning = np.clip(
            (frac[morning_mask] * n_seq).astype(int), 0, n_seq - 1
        )
        seq_idx_evening = np.clip(
            ((1.0 - frac[evening_mask]) * n_seq).astype(int), 0, n_seq - 1
        )

        tower_idx_morning = commute_seq[seq_idx_morning]
        tower_idx_evening = commute_seq[seq_idx_evening]

        out_tower_id[morning_mask] = tower_id_arr[tower_idx_morning]
        out_lat[morning_mask]      = tower_lat_arr[tower_idx_morning]
        out_lon[morning_mask]      = tower_lon_arr[tower_idx_morning]
        out_ward[morning_mask]     = tower_ward_arr[tower_idx_morning]

        out_tower_id[evening_mask] = tower_id_arr[tower_idx_evening]
        out_lat[evening_mask]      = tower_lat_arr[tower_idx_evening]
        out_lon[evening_mask]      = tower_lon_arr[tower_idx_evening]
        out_ward[evening_mask]     = tower_ward_arr[tower_idx_evening]

    # Timestamps
    timestamps = (
        np.datetime64(base_date)
        + day_idx.astype("timedelta64[D]")
        + hour.astype("timedelta64[h]")
        + minute.astype("timedelta64[m]")
        + second.astype("timedelta64[s]")
    )

    duration = rng.integers(30, 601, size=n_total)

    # ── Guarantee full commute coverage per day ──────────────────────────────
    # Problem: users whose random pings fall outside the commute window jump
    # directly home_nn → work_nn when sorted by timestamp, producing a gap
    # equal to the full commute distance. Injecting only the midpoint tower
    # still leaves two ~half-commute jumps, both of which can exceed 2km for
    # long commutes (>4km). Fix: inject one synthetic ping per tower in
    # commute_seq, evenly spaced across the commute window. This caps the
    # maximum consecutive distance at the inter-tower spacing (~0.8km).
    extra = {k: [] for k in
             ["user_id","timestamp","tower_id","tower_lat",
              "tower_lon","ward_id","duration_seconds"]}

    for d in range(N_DAYS):
        day_mask = day_idx == d
        for phase, (t_start_d, t_end_d, mask_d, seq_direction) in [
            ("morning", (depart_hour_d, arrive_work_d,  morning_mask, commute_seq)),
            ("evening", (depart_work_d, arrive_home_d,  evening_mask, commute_seq[::-1])),
        ]:
            if not (mask_d & day_mask).any():
                t_start = t_start_d[d]
                t_end   = t_end_d[d]
                n_inj   = len(seq_direction)
                # Evenly space one ping per tower across the commute window
                for step, tidx in enumerate(seq_direction):
                    frac     = (step + 0.5) / n_inj
                    h_frac   = t_start + frac * (t_end - t_start)
                    h        = int(h_frac) % 24
                    mn       = int((h_frac - int(h_frac)) * 60)
                    ts = (np.datetime64(base_date)
                          + np.timedelta64(d, "D")
                          + np.timedelta64(h, "h")
                          + np.timedelta64(mn, "m"))
                    extra["user_id"].append(user_id)
                    extra["timestamp"].append(ts)
                    extra["tower_id"].append(tower_id_arr[tidx])
                    extra["tower_lat"].append(float(tower_lat_arr[tidx]))
                    extra["tower_lon"].append(float(tower_lon_arr[tidx]))
                    extra["ward_id"].append(tower_ward_arr[tidx])
                    extra["duration_seconds"].append(int(rng.integers(30, 601)))

    base_dict = {
        "user_id":          np.full(n_total, user_id, dtype=object),
        "timestamp":        timestamps,
        "tower_id":         out_tower_id,
        "tower_lat":        out_lat,
        "tower_lon":        out_lon,
        "ward_id":          out_ward,
        "duration_seconds": duration,
    }

    if extra["user_id"]:
        return {k: np.concatenate([base_dict[k],
                                   np.array(extra[k], dtype=base_dict[k].dtype)])
                for k in base_dict}

    return base_dict


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_cdr(
    cdr_df:  pd.DataFrame,
    user_df: pd.DataFrame,
) -> bool:
    print("\nRunning CDR validation...")
    passed = True

    # Check 1 — all 24 wards covered
    wards_in_cdr = cdr_df["ward_id"].nunique()
    if wards_in_cdr < 24:
        print(f"  FAIL — Only {wards_in_cdr}/24 wards covered.")
        passed = False
    else:
        print(f"  PASS — All 24 wards covered.")

    # Check 2 — user count
    n_users = cdr_df["user_id"].nunique()
    if n_users < TOTAL_USERS * 0.95:
        print(f"  FAIL — Only {n_users:,} users, expected ~{TOTAL_USERS:,}.")
        passed = False
    else:
        print(f"  PASS — {n_users:,} unique users.")

    # Check 3 — minimum 100 pings per user
    pings_per_user = cdr_df.groupby("user_id").size()
    low_ping_users = (pings_per_user < 100).sum()
    if low_ping_users > TOTAL_USERS * 0.05:
        print(f"  FAIL — {low_ping_users:,} users with fewer than 100 pings.")
        passed = False
    else:
        print(f"  PASS — Ping count per user sufficient.")

    # Check 4 — ward proportionality within 10%
    home_ward_counts = user_df["home_ward"].value_counts()
    total_pop        = sum(WARD_POPULATIONS.values())
    max_deviation    = 0.0
    worst_ward       = ""
    for ward, pop in WARD_POPULATIONS.items():
        expected  = TOTAL_USERS * pop / total_pop
        actual    = home_ward_counts.get(ward, 0)
        deviation = abs(actual - expected) / expected
        if deviation > max_deviation:
            max_deviation = deviation
            worst_ward    = ward
    if max_deviation > 0.10:
        print(f"  FAIL — Ward {worst_ward} deviates "
              f"{max_deviation:.1%} from population proportion.")
        passed = False
    else:
        print(f"  PASS — All wards within 10% of population proportions.")

    # Check 5 — weekdays only
    weekdays = pd.to_datetime(cdr_df["timestamp"]).dt.weekday
    if weekdays.max() > 4:
        print(f"  FAIL — Weekend records found.")
        passed = False
    else:
        print(f"  PASS — All records are weekday.")

    # Check 6 — no null values
    nulls = cdr_df.isnull().sum().sum()
    if nulls > 0:
        print(f"  FAIL — {nulls:,} null values found.")
        passed = False
    else:
        print(f"  PASS — No null values.")

    # Check 7 — minimum users per ward
    users_per_ward = user_df["home_ward"].value_counts()
    min_ward       = users_per_ward.idxmin()
    min_count      = users_per_ward.min()
    if min_count < 1000:
        print(f"  FAIL — Ward {min_ward} has only {min_count} users.")
        passed = False
    else:
        print(f"  PASS — Min users per ward: {min_count} (Ward {min_ward}).")

    # Check 8 — spatial coherence (sample 500 users, ALL consecutive pairs)
    # Check 8 — spatial coherence (within-day only, cross-day jumps excluded)
    rng_check    = np.random.default_rng(0)
    all_uids     = cdr_df["user_id"].unique()
    sample_users = rng_check.choice(
        all_uids, size=min(100, len(all_uids)), replace=False
    )
    sample_cdr   = cdr_df[cdr_df["user_id"].isin(sample_users)].copy()
    sample_cdr["date"] = pd.to_datetime(sample_cdr["timestamp"]).dt.date

    max_consec_dist = 0.0
    worst_uid       = ""

    for (uid, date), grp in sample_cdr.groupby(["user_id", "date"]):
        grp  = grp.sort_values("timestamp").reset_index(drop=True)
        lats = grp["tower_lat"].values
        lons = grp["tower_lon"].values
        for i in range(len(lats) - 1):
            d = haversine_simple(lats[i], lons[i], lats[i+1], lons[i+1])
            if d > max_consec_dist:
                max_consec_dist = d
                worst_uid       = uid

    if max_consec_dist > 3.0:
        print(f"  FAIL — Max within-day consecutive ping distance: "
              f"{max_consec_dist:.1f} km (user {worst_uid}).")
        passed = False
    else:
        print(f"  PASS — Max within-day ping distance: "
              f"{max_consec_dist:.1f} km. Spatially coherent.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng       = np.random.default_rng(42)
    base_date = datetime(2024, 1, 1)

    # Load ward boundaries
    print("Loading ward boundaries...")
    ward_gdf = gpd.read_file(GEOJSON_PATH)
    ward_gdf["ward_id"] = ward_gdf["Name"].str.strip()
    ward_gdf = ward_gdf[["ward_id", "geometry"]].set_index("ward_id")
    if ward_gdf.crs is None:
        ward_gdf = ward_gdf.set_crs("EPSG:4326")
    else:
        ward_gdf = ward_gdf.to_crs("EPSG:4326")
    print(f"Loaded {len(ward_gdf)} ward polygons.")

    # Generate tower grid
    print("\nGenerating tower grid...")
    tower_df = generate_tower_grid(ward_gdf, rng)
    os.makedirs("../data/raw/cdr", exist_ok=True)
    tower_df.to_parquet("../data/raw/cdr/towers.parquet", index=False)
    print(f"Saved: ../data/raw/cdr/towers.parquet")

    # Assign users
    print("\nAssigning users...")
    user_df = assign_users(tower_df, rng)
    user_df.to_parquet("../data/raw/cdr/users.parquet", index=False)
    print(f"Saved: ../data/raw/cdr/users.parquet")

    # Build tower KDTree once — used both to find each user's local
    # home/work tower cluster and to snap interpolated commute
    # positions to the nearest real tower
    print("\nBuilding tower spatial index...")
    tower_id_arr   = tower_df["tower_id"].to_numpy()
    tower_lat_arr  = tower_df["tower_lat"].to_numpy()
    tower_lon_arr  = tower_df["tower_lon"].to_numpy()
    tower_ward_arr = tower_df["ward_id"].to_numpy()
    tower_kdtree   = cKDTree(np.column_stack([tower_lat_arr, tower_lon_arr]))
    print(f"Indexed {len(tower_lat_arr):,} towers.")

    # Generate CDR pings — iterate via numpy arrays, not .iterrows()
    print(f"\nGenerating CDR pings for {len(user_df):,} users...")
    user_ids   = user_df["user_id"].to_numpy()
    home_lats  = user_df["home_lat"].to_numpy()
    home_lons  = user_df["home_lon"].to_numpy()
    work_lats  = user_df["work_lat"].to_numpy()
    work_lons  = user_df["work_lon"].to_numpy()
    home_wards = user_df["home_ward"].to_numpy()
    work_wards = user_df["work_ward"].to_numpy()

    per_user_results = []
    n_users = len(user_df)
    for i in range(n_users):
        result = generate_user_pings(
            user_ids[i], home_wards[i], work_wards[i],
            home_lats[i], home_lons[i], work_lats[i], work_lons[i],
            tower_id_arr, tower_lat_arr, tower_lon_arr, tower_ward_arr,
            tower_kdtree, rng, base_date,
        )
        per_user_results.append(result)
        if (i + 1) % 10000 == 0:
            n_so_far = sum(len(r["user_id"]) for r in per_user_results)
            print(f"  Progress: {i+1:,}/{n_users:,} users — "
                  f"{n_so_far:,} pings so far")

    # Concatenate all per-user arrays once — avoids building a
    # 35M-row Python list of dicts before handing off to pandas
    cdr_df = pd.DataFrame({
        col: np.concatenate([r[col] for r in per_user_results])
        for col in per_user_results[0].keys()
    })
    cdr_df["timestamp"] = pd.to_datetime(cdr_df["timestamp"])
    cdr_df = cdr_df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)
    print(f"\nTotal pings generated: {len(cdr_df):,}")

    # Validate
    valid = validate_cdr(cdr_df, user_df)
    if not valid:
        print("\nCDR failed validation. Fix before proceeding.")
        exit(1)

    # Save
    out_path = "../data/raw/cdr/mumbai_cdr.parquet"
    cdr_df.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}")
    print(f"Shape: {cdr_df.shape}")
    print(f"Columns: {list(cdr_df.columns)}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"\nPing stats per ward:")
    print(cdr_df.groupby("ward_id").size().describe().round(0))