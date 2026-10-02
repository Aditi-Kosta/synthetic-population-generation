# mumbai-synth-v2/generation/generate_survey.py
"""
Stage 0 — Household Travel Survey Generation
Generates synthetic but statistically realistic household travel survey
for Mumbai covering all 24 BMC wards.

Two output files — must be linkable via person_id:
  data/raw/survey/survey_sociodem.parquet   — socio-demographic attributes
  data/raw/survey/survey_trips.parquet      — travel demand patterns

Both files together form the labeled training dataset for the
demographic inference model in Stage 3.

Fixes applied (v1):
  1. Return trip generation removed — 2.14 is total trips per day
  2. Square root proportional ward stratification with 400 floor
  3. MIN_HH_PER_WARD raised to 400
  4. Children (0-14) excluded from independent trip generation

Fixes applied (v2) — CDR/survey feature compatibility:
  5. Trips chained into closed home-based tours
  6. Fixed work_ward anchor per employed/student person
  7. Departure times from peaked commute distributions

Fixes applied (v3) — employment class separability:
  8. Student work_ward drawn from EDUCATION gravity model (university-
     heavy wards: F/N, K/E, K/W, H/E, L, M/W, G/N) not commercial
     gravity. Source: Mumbai university campus locations — IIT Bombay
     (H/E/K/E), VJTI (H/E), ICT (M/W), Tata Institute (H/W), Mumbai
     University Fort (A/B), Narsee Monjee (V/W area = K/W).
  9. Student departure: N(9.0, 0.5) — Mumbai college start 9-10am
     (University of Mumbai timetable; first lecture 9:00-9:15am).
     Employed departure: N(8.0, 0.6) retained — office start 9am but
     commute begins ~8am given Mumbai travel times.
 10. Student return: N(16.5, 0.8) — last lecture ~4-5pm.
     Employed return: N(18.5, 1.0) — office end 6pm + commute delay.
     (Raised from 18.0 to 18.5 to better match Mumbai IT/corporate
     sector typical 6:30pm departure from office; MMRDA peak PM
     is 17:30-19:30.)
 11. Homemaker departure: N(10.5, 1.5) — mid-morning errands, not
     peak AM. Retired departure: N(11.0, 1.8) — late morning.
     These create measurable pct_peak_am separation between
     employed/student and homemaker/retired.
 12. Homemaker/retired return: N(14.0, 1.5) / N(13.5, 2.0) —
     afternoon return, not PM peak. This separates pct_peak_pm.

These changes create statistically measurable feature separation
between employment classes on pct_peak_am, pct_peak_pm, and
home_work_dist_km (via different destination gravity models),
enabling the Stage 3 XGBoost model to discriminate employment.
"""

import numpy as np
import pandas as pd
import os
from math import radians, sin, cos, sqrt, asin

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

WARD_INCOME_TIER = {
    "A": "high",  "B": "high",  "C": "high",
    "D": "mid",   "E": "mid",   "F/S": "mid",
    "F/N": "mid", "G/S": "mid", "G/N": "mid",
    "H/E": "mid", "H/W": "high","K/E": "mid",
    "K/W": "high","P/S": "mid", "P/N": "low",
    "R/S": "low", "R/C": "low", "R/N": "low",
    "L":  "low",  "M/E": "low", "M/W": "mid",
    "N":  "low",  "S":   "low", "T":   "low",
}

# Commercial gravity weights — used for employed work_ward assignment
COMMERCIAL_WEIGHT = {
    "A": 2.5, "B": 2.0, "C": 1.8, "D": 1.5, "E": 1.3,
    "F/S": 1.5, "F/N": 1.2, "G/S": 1.8, "G/N": 1.4,
    "H/E": 2.0, "H/W": 1.8, "K/E": 1.3, "K/W": 1.2,
    "P/S": 0.9, "P/N": 0.8, "R/S": 0.7, "R/C": 0.7,
    "R/N": 0.6, "L":   1.1, "M/E": 0.9, "M/W": 0.8,
    "N":   0.9, "S":   0.7, "T":   0.6,
}

# Education gravity weights — used for student work_ward (= college ward)
# assignment. Weights reflect university/college density per ward.
# Source: Mumbai university campus locations —
#   IIT Bombay, VJTI → H/E
#   ICT Mumbai        → M/W
#   Mumbai University Fort campus, Elphinstone → A/B
#   Narsee Monjee, NMIMS → K/W
#   Tata Institute → H/W
#   SIES, Ruia, Kirti → G/N, K/W, K/E
#   SNDTUniversity → K/W, S
#   Various junior colleges → distributed F/N, L, N, P/N (high pop wards)
EDUCATION_WEIGHT = {
    "A": 1.8, "B": 1.6, "C": 1.0, "D": 0.8, "E": 0.7,
    "F/S": 0.8, "F/N": 1.2, "G/S": 0.9, "G/N": 1.4,
    "H/E": 3.0, "H/W": 2.0, "K/E": 1.8, "K/W": 2.5,
    "P/S": 0.7, "P/N": 1.0, "R/S": 0.6, "R/C": 0.6,
    "R/N": 0.5, "L":   1.1, "M/E": 0.8, "M/W": 2.2,
    "N":   0.9, "S":   0.7, "T":   0.5,
}

TOTAL_HOUSEHOLDS = 15000
MIN_HH_PER_WARD  = 400
TRIPS_PER_DAY    = 2.14      # IIT Bombay TVM — total trips per day

# ── Demographic marginals ──────────────────────────────────────────────────────

AGE_DIST = {
    "0-14":  0.18, "15-24": 0.18,
    "25-44": 0.35, "45-59": 0.18, "60+": 0.11,
}

EMPLOYMENT_BY_AGE = {
    "0-14":  {"student": 0.80, "homemaker": 0.10, "unemployed": 0.10,
               "employed": 0.00, "retired": 0.00},
    "15-24": {"student": 0.45, "employed":  0.35, "unemployed": 0.10,
               "homemaker": 0.08, "retired": 0.02},
    "25-44": {"employed": 0.65, "homemaker": 0.22, "unemployed": 0.08,
               "student": 0.04, "retired": 0.01},
    "45-59": {"employed": 0.58, "homemaker": 0.25, "retired": 0.10,
               "unemployed": 0.07, "student": 0.00},
    "60+":   {"retired": 0.60, "homemaker": 0.25, "employed": 0.12,
               "unemployed": 0.03, "student": 0.00},
}

INCOME_BY_TIER = {
    "high": {"low": 0.10, "mid": 0.35, "high": 0.55},
    "mid":  {"low": 0.30, "mid": 0.50, "high": 0.20},
    "low":  {"low": 0.60, "mid": 0.33, "high": 0.07},
}

VEHICLE_BY_INCOME = {
    "low":  {"none": 0.72, "2w": 0.22, "car": 0.03, "both": 0.03},
    "mid":  {"none": 0.45, "2w": 0.35, "car": 0.12, "both": 0.08},
    "high": {"none": 0.20, "2w": 0.28, "car": 0.32, "both": 0.20},
}

HH_SIZE_DIST = {
    1: 0.08, 2: 0.18, 3: 0.22, 4: 0.28,
    5: 0.13, 6: 0.07, 7: 0.04,
}

MODES = [
    "local_train", "best_bus", "metro", "auto",
    "taxi",        "car",      "two_wheeler",
]

MODE_PROB_BY_INCOME = {
    "low":  [0.48, 0.30, 0.04, 0.10, 0.03, 0.02, 0.03],
    "mid":  [0.40, 0.27, 0.06, 0.10, 0.05, 0.06, 0.06],
    "high": [0.28, 0.18, 0.08, 0.08, 0.10, 0.16, 0.12],
}

TRIP_PURPOSES = [
    "work", "education", "shopping",
    "social", "medical", "other",
]

PURPOSE_BY_EMPLOYMENT_SECONDARY = {
    "employed":   [0.05, 0.03, 0.32, 0.35, 0.13, 0.12],
    "student":    [0.03, 0.10, 0.28, 0.38, 0.10, 0.11],
    "homemaker":  [0.05, 0.05, 0.40, 0.25, 0.15, 0.10],
    "retired":    [0.05, 0.00, 0.30, 0.30, 0.25, 0.10],
    "unemployed": [0.15, 0.10, 0.30, 0.25, 0.10, 0.10],
}

TRIP_MAKE_PROB = {
    "employed":   0.95,
    "student":    0.90,
    "homemaker":  0.75,
    "retired":    0.60,
    "unemployed": 0.50,
}

PRIMARY_ANCHOR_HIT_PROB = {
    "employed": 0.88,
    "student":  0.92,
}

MODE_SPEED_KMH = {
    "local_train": 35, "metro": 40, "best_bus": 18,
    "auto": 22, "taxi": 28, "car": 25, "two_wheeler": 27,
}

METRO_DIST_KM = {
    "A": 1.193, "B": 0.907, "C": 0.644, "D": 0.324,
    "E": 1.449, "F/S": 1.385, "F/N": 0.424, "G/S": 1.444,
    "G/N": 1.562, "H/E": 1.483, "H/W": 1.020, "K/E": 0.712,
    "K/W": 1.940, "P/S": 0.597, "P/N": 1.124, "R/S": 0.461,
    "R/C": 0.662, "R/N": 0.568, "L": 2.371, "M/E": 3.528,
    "M/W": 2.656, "N": 1.797, "S": 5.192, "T": 9.810,
}

HOURLY_WEIGHTS = np.array([
     8,  5,  3,  3,  6, 16,
    44, 82, 98, 74, 58, 54,
    56, 50, 52, 58, 64, 88,
    95, 82, 66, 46, 30, 16,
], dtype=float)
HOURLY_WEIGHTS /= HOURLY_WEIGHTS.sum()

# ── Fix v3: Employment-differentiated departure/return distributions ───────────
#
# Employed (office workers):
#   Depart home: N(8.0, 0.6)  — leave home ~7-9am to reach office by 9am
#   Depart work: N(18.5, 1.0) — leave office ~17:30-19:30 (MMRDA PM peak)
#
# Student (college):
#   Depart home: N(9.0, 0.5)  — first lecture 9:00-9:15am (Univ of Mumbai)
#   Depart college: N(16.5, 0.8) — last lecture ~4-5pm, home by 5-6pm
#
# Homemaker:
#   Depart home: N(10.5, 1.5) — mid-morning errands, not peak AM
#   Return home: N(14.0, 1.5) — afternoon return
#
# Retired:
#   Depart home: N(11.0, 1.8) — late morning
#   Return home: N(13.5, 2.0) — afternoon
#
# Unemployed:
#   Depart home: N(10.0, 2.0) — irregular, spread through day
#   Return home: N(15.0, 2.5) — irregular
#
# These are grounded in:
#   - University of Mumbai first/last lecture schedules
#   - MMRDA Traffic & Transportation Study 2015 peak hour definitions
#   - BEST bus ridership peak AM 07:30-09:30, PM 17:30-19:30
#   - IIT Bombay TVM 2008 trip purpose/time cross-tabulation

DEPART_DIST = {
    #               (mean_out, std_out, mean_return, std_return)
    "employed":   (8.0,  0.6,  18.5, 1.0),
    "student":    (9.0,  0.5,  16.5, 0.8),
    "homemaker":  (10.5, 1.5,  14.0, 1.5),
    "retired":    (11.0, 1.8,  13.5, 2.0),
    "unemployed": (10.0, 2.0,  15.0, 2.5),
}

COMMUTE_SPEED_KMH = 25.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


def sample_from(dist: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    keys  = list(dist.keys())
    probs = list(dist.values())
    return rng.choice(keys, size=n, p=probs)


def clock_from_hour_float(hour_val: float) -> tuple:
    hour_val = hour_val % 24.0
    hour     = int(hour_val)
    minute   = int(round((hour_val - hour) * 60)) % 60
    return hour, minute


# ── Ward allocation ────────────────────────────────────────────────────────────

def ward_allocation(total: int, min_per_ward: int) -> dict:
    wards      = list(WARD_POPULATIONS.keys())
    sqrt_pops  = {w: np.sqrt(WARD_POPULATIONS[w]) for w in wards}
    total_sqrt = sum(sqrt_pops.values())
    raw_alloc  = {w: total * sqrt_pops[w] / total_sqrt for w in wards}
    allocation = {w: max(min_per_ward, int(raw_alloc[w])) for w in wards}
    current_total = sum(allocation.values())
    shortfall     = total - current_total
    fractions = sorted(
        wards, key=lambda w: raw_alloc[w] - int(raw_alloc[w]), reverse=True
    )
    for w in fractions[:abs(shortfall)]:
        if shortfall > 0:
            allocation[w] += 1
        elif shortfall < 0 and allocation[w] > min_per_ward:
            allocation[w] -= 1
    return allocation


# ── Socio-demographic generation ───────────────────────────────────────────────

def generate_sociodem(
    rng: np.random.Generator,
    allocation: dict,
) -> pd.DataFrame:
    """
    Fix v3: student work_ward drawn from EDUCATION_WEIGHT gravity model,
    not COMMERCIAL_WEIGHT. This creates different home-work distance
    distributions between employed and student (employed → CBD/commercial
    wards; student → university wards), which the Stage 3 model can use
    as a discriminating feature.
    """
    all_persons       = []
    hh_id_counter     = 0
    person_id_counter = 0

    work_wards = list(WARD_POPULATIONS.keys())

    # Employed: commercial gravity
    emp_work_probs = np.array([
        WARD_POPULATIONS[w] * COMMERCIAL_WEIGHT[w] for w in work_wards
    ], dtype=float)
    emp_work_probs /= emp_work_probs.sum()

    # Student: education gravity (Fix v3.8)
    stu_work_probs = np.array([
        WARD_POPULATIONS[w] * EDUCATION_WEIGHT[w] for w in work_wards
    ], dtype=float)
    stu_work_probs /= stu_work_probs.sum()

    for ward_id, n_hh in allocation.items():
        tier      = WARD_INCOME_TIER[ward_id]
        is_island = int(ward_id in ISLAND_CITY_WARDS)

        hh_sizes   = sample_from(HH_SIZE_DIST,        n_hh, rng).astype(int)
        hh_incomes = sample_from(INCOME_BY_TIER[tier], n_hh, rng)

        for hh_idx in range(n_hh):
            hh_id      = f"HH_{hh_id_counter:07d}"
            hh_size    = hh_sizes[hh_idx]
            hh_income  = hh_incomes[hh_idx]
            hh_vehicle = sample_from(VEHICLE_BY_INCOME[hh_income], 1, rng)[0]

            ages = sample_from(AGE_DIST, hh_size, rng)
            for age in ages:
                emp = sample_from(EMPLOYMENT_BY_AGE[age], 1, rng)[0]

                # Fix v3.8 — different gravity model per employment type
                if emp == "employed":
                    work_ward = rng.choice(work_wards, p=emp_work_probs)
                elif emp == "student":
                    work_ward = rng.choice(work_wards, p=stu_work_probs)
                else:
                    work_ward = None

                all_persons.append({
                    "person_id":     f"P_{person_id_counter:08d}",
                    "hh_id":         hh_id,
                    "ward_id":       ward_id,
                    "work_ward":     work_ward,
                    "age_bracket":   age,
                    "employment":    emp,
                    "income_class":  hh_income,
                    "vehicle_owned": hh_vehicle,
                    "hh_size":       hh_size,
                    "hh_income":     hh_income,
                    "island_city":   is_island,
                    "income_tier":   tier,
                })
                person_id_counter += 1
            hh_id_counter += 1

    df = pd.DataFrame(all_persons)
    print(f"Generated {len(df):,} persons across {df['ward_id'].nunique()} wards.")
    n_emp = (df["employment"] == "employed").sum()
    n_stu = (df["employment"] == "student").sum()
    print(f"  Employed with commercial work_ward : {n_emp:,}")
    print(f"  Students with education work_ward  : {n_stu:,}")
    return df


# ── Mode assignment ────────────────────────────────────────────────────────────

def get_mode(
    income_class:  str,
    ward_id:       str,
    dest_ward:     str,
    vehicle_owned: str,
    rng:           np.random.Generator,
) -> str:
    probs = np.array(MODE_PROB_BY_INCOME[income_class], dtype=float)

    if ward_id in ISLAND_CITY_WARDS or dest_ward in ISLAND_CITY_WARDS:
        probs[MODES.index("auto")] = 0.0

    if METRO_DIST_KM.get(ward_id, 999) > 1.5:
        probs[MODES.index("metro")] = 0.0

    if vehicle_owned == "none":
        probs[MODES.index("car")]         = 0.0
        probs[MODES.index("two_wheeler")] = 0.0
    elif vehicle_owned == "2w":
        probs[MODES.index("car")]         = 0.0
    elif vehicle_owned == "car":
        probs[MODES.index("two_wheeler")] = 0.0

    total = probs.sum()
    if total == 0:
        probs = np.ones(len(MODES)) / len(MODES)
    else:
        probs /= total

    return rng.choice(MODES, p=probs)


# ── Trip generation ────────────────────────────────────────────────────────────

def _choose_gravity_ward(
    wards: list, dest_probs: np.ndarray, exclude_ward: str,
    rng: np.random.Generator,
) -> str:
    if exclude_ward not in wards:
        return rng.choice(wards, p=dest_probs)
    idx   = wards.index(exclude_ward)
    probs = dest_probs.copy()
    probs[idx] = 0.0
    s = probs.sum()
    if s == 0:
        return exclude_ward
    probs /= s
    return rng.choice(wards, p=probs)


def _get_depart_hours(
    emp: str,
    has_anchor: bool,
    n_trips: int,
    rng: np.random.Generator,
) -> list:
    """
    Fix v3.9-12: employment-differentiated departure time distributions.
    Returns a list of n_trips departure hour floats.

    For anchored persons (employed/student with fixed work_ward):
      leg[0]    — outbound: drawn from DEPART_DIST[emp][0,1]
      leg[-1]   — return:   drawn from DEPART_DIST[emp][2,3]
      middle    — interpolated between outbound and return

    For non-anchored (homemaker, retired, unemployed):
      leg[0]    — outbound: drawn from DEPART_DIST[emp][0,1]
      leg[-1]   — return:   drawn from DEPART_DIST[emp][2,3]
      middle    — interpolated

    All non-anchored persons also use DEPART_DIST — this ensures
    homemaker/retired departure is mid-morning not peak AM,
    creating measurable pct_peak_am separation.
    """
    d = DEPART_DIST.get(emp, (10.0, 2.0, 15.0, 2.5))
    mean_out, std_out, mean_ret, std_ret = d

    if n_trips == 1:
        return [float(rng.normal(mean_out, std_out))]

    hours = [None] * n_trips
    hours[0]  = float(rng.normal(mean_out, std_out))
    hours[-1] = float(rng.normal(mean_ret, std_ret))

    # Ensure return is after outbound — if not, add a half-day
    if hours[-1] <= hours[0]:
        hours[-1] = hours[0] + abs(rng.normal(4.0, 1.0))

    for mid in range(1, n_trips - 1):
        frac      = mid / n_trips
        hours[mid] = hours[0] + frac * (hours[-1] - hours[0])

    return hours


def generate_trips(
    sociodem_df: pd.DataFrame,
    rng:         np.random.Generator,
) -> pd.DataFrame:
    """
    Generate chained home-based tours with employment-differentiated
    departure times (Fix v3).
    """
    wards      = list(WARD_POPULATIONS.keys())
    dest_probs = np.array([
        WARD_POPULATIONS[w] * COMMERCIAL_WEIGHT[w] for w in wards
    ], dtype=float)
    dest_probs /= dest_probs.sum()

    all_trips       = []
    trip_id_counter = 0

    eligible = sociodem_df[sociodem_df["age_bracket"] != "0-14"].copy()
    print(f"Eligible trip makers (age 15+): {len(eligible):,}")

    for _, person in eligible.iterrows():
        emp = person["employment"]

        if rng.random() > TRIP_MAKE_PROB.get(emp, 0.70):
            continue

        n_trips   = max(1, rng.poisson(TRIPS_PER_DAY))
        home_ward = person["ward_id"]
        has_anchor = (
            emp in PRIMARY_ANCHOR_HIT_PROB
            and pd.notna(person["work_ward"])
        )

        # Build tour ward sequence
        tour_wards = [home_ward]
        for leg_num in range(n_trips):
            is_first = (leg_num == 0)
            is_last  = (leg_num == n_trips - 1) and (n_trips > 1)
            current  = tour_wards[-1]

            if is_last:
                nxt = home_ward
            elif is_first and has_anchor:
                if rng.random() < PRIMARY_ANCHOR_HIT_PROB[emp]:
                    nxt = person["work_ward"]
                else:
                    nxt = _choose_gravity_ward(wards, dest_probs, current, rng)
            else:
                nxt = _choose_gravity_ward(wards, dest_probs, current, rng)
            tour_wards.append(nxt)

        # Employment-differentiated departure times (Fix v3)
        leg_depart_hours = _get_depart_hours(emp, has_anchor, n_trips, rng)

        purpose_probs = PURPOSE_BY_EMPLOYMENT_SECONDARY.get(emp, [1/6]*6)

        for leg_num in range(n_trips):
            origin_ward = tour_wards[leg_num]
            dest_ward   = tour_wards[leg_num + 1]

            origin_lat, origin_lon = WARD_CENTROIDS[origin_ward]
            dest_lat,   dest_lon   = WARD_CENTROIDS[dest_ward]
            dist_km = haversine(origin_lat, origin_lon, dest_lat, dest_lon)

            mode = get_mode(
                person["income_class"], origin_ward, dest_ward,
                person["vehicle_owned"], rng,
            )
            speed_kmh   = MODE_SPEED_KMH.get(mode, 25)
            travel_mins = (
                (dist_km / speed_kmh) * 60 if dist_km > 0
                else rng.uniform(5, 15)
            )

            dep_hour_val          = leg_depart_hours[leg_num]
            dep_hour, dep_minute  = clock_from_hour_float(dep_hour_val)
            arr_mins   = dep_hour * 60 + dep_minute + travel_mins
            arr_hour   = int((arr_mins // 60) % 24)
            arr_minute = int(arr_mins % 60)

            if leg_num == 0 and has_anchor and dest_ward == person["work_ward"]:
                purpose = "work" if emp == "employed" else "education"
            else:
                purpose = rng.choice(TRIP_PURPOSES, p=purpose_probs)

            all_trips.append({
                "trip_id":          f"TR_{trip_id_counter:09d}",
                "person_id":        person["person_id"],
                "hh_id":            person["hh_id"],
                "ward_id":          home_ward,
                "origin_ward":      origin_ward,
                "dest_ward":        dest_ward,
                "origin_lat":       round(origin_lat, 6),
                "origin_lon":       round(origin_lon, 6),
                "dest_lat":         round(dest_lat, 6),
                "dest_lon":         round(dest_lon, 6),
                "trip_dist_km":     round(dist_km, 3),
                "mode":             mode,
                "departure_hour":   dep_hour,
                "departure_minute": dep_minute,
                "arrival_hour":     arr_hour,
                "arrival_minute":   arr_minute,
                "travel_time_mins": round(travel_mins, 1),
                "trip_purpose":     purpose,
                "trip_num":         leg_num + 1,
            })
            trip_id_counter += 1

    df = pd.DataFrame(all_trips)
    print(f"Generated {len(df):,} trip records for "
          f"{df['person_id'].nunique():,} persons.")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_survey(
    sociodem_df: pd.DataFrame,
    trips_df:    pd.DataFrame,
) -> bool:
    print("\nRunning survey validation...")
    passed = True

    n_wards = sociodem_df["ward_id"].nunique()
    if n_wards < 24:
        print(f"  FAIL — Only {n_wards}/24 wards in sociodem.")
        passed = False
    else:
        print(f"  PASS — All 24 wards covered.")

    hh_per_ward = sociodem_df.groupby("ward_id")["hh_id"].nunique()
    min_ward    = hh_per_ward.idxmin()
    min_count   = hh_per_ward.min()
    if min_count < MIN_HH_PER_WARD:
        print(f"  FAIL — Ward {min_ward} has {min_count} hh, "
              f"minimum {MIN_HH_PER_WARD}.")
        passed = False
    else:
        print(f"  PASS — Min households per ward: {min_count} ({min_ward}).")

    orphan_trips = (
        set(trips_df["person_id"].unique()) -
        set(sociodem_df["person_id"].unique())
    )
    if orphan_trips:
        print(f"  FAIL — {len(orphan_trips):,} trips with no matching person.")
        passed = False
    else:
        print(f"  PASS — All trips link to valid persons.")

    MMRDA_TARGETS = {
        "local_train": 0.43, "best_bus": 0.27, "metro":       0.05,
        "auto":        0.10, "taxi":     0.05, "car":         0.06,
        "two_wheeler": 0.04,
    }
    sim_shares = trips_df["mode"].value_counts(normalize=True)
    mode_ok    = True
    for mode, target in MMRDA_TARGETS.items():
        sim = sim_shares.get(mode, 0.0)
        if abs(sim - target) > 0.08:
            print(f"  WARN — {mode}: {sim:.1%} vs target {target:.1%}")
            mode_ok = False
    if mode_ok:
        print(f"  PASS — All mode shares within 8% of MMRDA targets.")

    critical = [
        "person_id", "hh_id", "ward_id", "age_bracket",
        "employment", "income_class", "vehicle_owned",
    ]
    nulls = sociodem_df[critical].isnull().sum().sum()
    if nulls > 0:
        print(f"  FAIL — {nulls:,} nulls in critical sociodem columns.")
        passed = False
    else:
        print(f"  PASS — No nulls in critical columns.")

    valid_wards = set(WARD_POPULATIONS.keys())
    bad_origin  = (~trips_df["origin_ward"].isin(valid_wards)).sum()
    bad_dest    = (~trips_df["dest_ward"].isin(valid_wards)).sum()
    if bad_origin > 0 or bad_dest > 0:
        print(f"  FAIL — {bad_origin} bad origin, {bad_dest} bad dest wards.")
        passed = False
    else:
        print(f"  PASS — All trip ward references valid.")

    island_auto = trips_df[
        (trips_df["mode"] == "auto") &
        (
            trips_df["origin_ward"].isin(ISLAND_CITY_WARDS) |
            trips_df["dest_ward"].isin(ISLAND_CITY_WARDS)
        )
    ]
    if len(island_auto) > 0:
        print(f"  FAIL — {len(island_auto):,} auto trips violate Island City.")
        passed = False
    else:
        print(f"  PASS — Island City auto restriction enforced.")

    trips_per_person = trips_df.groupby("person_id").size()
    mean_trips       = trips_per_person.mean()
    if not (1.5 <= mean_trips <= 3.5):
        print(f"  WARN — Mean trips/person: {mean_trips:.2f} (expected 1.5-3.5).")
    else:
        print(f"  PASS — Mean trips per person: {mean_trips:.2f}.")

    multi_trip = trips_df[
        trips_df.groupby("person_id")["trip_num"].transform("max") > 1
    ]
    if len(multi_trip) > 0:
        last_legs      = multi_trip.sort_values("trip_num").groupby("person_id").tail(1)
        home_by_person = sociodem_df.set_index("person_id")["ward_id"]
        closed  = (last_legs["dest_ward"] ==
                   last_legs["person_id"].map(home_by_person))
        pct_closed = closed.mean()
        if pct_closed < 0.999:
            print(f"  FAIL — Only {pct_closed:.1%} of tours close at home.")
            passed = False
        else:
            print(f"  PASS — {pct_closed:.1%} of multi-leg tours close at home.")

    chained = trips_df.sort_values(["person_id","trip_num"])
    chained["prev_dest"] = chained.groupby("person_id")["dest_ward"].shift(1)
    mask   = chained["prev_dest"].notna()
    breaks = (chained.loc[mask,"origin_ward"] !=
              chained.loc[mask,"prev_dest"]).sum()
    if breaks > 0:
        print(f"  FAIL — {breaks:,} chain breaks.")
        passed = False
    else:
        print(f"  PASS — All chained legs spatially continuous.")

    anchored = sociodem_df[sociodem_df["work_ward"].notna()]
    if len(anchored) > 0:
        first_legs = trips_df[trips_df["trip_num"]==1].merge(
            anchored[["person_id","work_ward","employment"]],
            on="person_id", how="inner",
        )
        if len(first_legs) > 0:
            hit = (first_legs["dest_ward"] == first_legs["work_ward"]).mean()
            print(f"  INFO — Primary leg anchor hit rate: {hit:.1%} "
                  f"(target ~85-92%).")

    # Fix v3 verification — check employment separation on departure hour
    print("\n  Fix v3 check — mean departure hour by employment:")
    first_legs_all = trips_df[trips_df["trip_num"] == 1].merge(
        sociodem_df[["person_id","employment"]], on="person_id"
    )
    dep_by_emp = first_legs_all.groupby("employment")["departure_hour"].mean().round(2)
    emp_order  = ["employed","student","homemaker","retired","unemployed"]
    for e in emp_order:
        if e in dep_by_emp:
            print(f"    {e:<12} mean_depart_hour = {dep_by_emp[e]:.2f}")
    # Verify employed < student < homemaker/retired
    emp_h  = dep_by_emp.get("employed",  8.0)
    stu_h  = dep_by_emp.get("student",   9.0)
    hmk_h  = dep_by_emp.get("homemaker", 10.5)
    ret_h  = dep_by_emp.get("retired",   11.0)
    if emp_h < stu_h < hmk_h and emp_h < stu_h < ret_h:
        print(f"  PASS — Employment departure time ordering correct "
              f"(employed < student < homemaker/retired).")
    else:
        print(f"  WARN — Employment departure time ordering unexpected.")

    print(f"\nValidation {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    os.makedirs("../data/raw/survey", exist_ok=True)

    print("Computing ward allocation...")
    allocation = ward_allocation(TOTAL_HOUSEHOLDS, MIN_HH_PER_WARD)
    print(f"Total households: {sum(allocation.values()):,}")
    print(f"Min per ward: {min(allocation.values())} "
          f"(Ward {min(allocation, key=allocation.get)})")
    print(f"Max per ward: {max(allocation.values())} "
          f"(Ward {max(allocation, key=allocation.get)})")

    print("\nGenerating socio-demographic records...")
    sociodem_df = generate_sociodem(rng, allocation)

    print("\nGenerating trip records...")
    trips_df = generate_trips(sociodem_df, rng)

    valid = validate_survey(sociodem_df, trips_df)
    if not valid:
        print("\nSurvey failed validation. Fix before proceeding.")
        exit(1)

    sociodem_path = "../data/raw/survey/survey_sociodem.parquet"
    trips_path    = "../data/raw/survey/survey_trips.parquet"
    sociodem_df.to_parquet(sociodem_path, index=False)
    trips_df.to_parquet(trips_path,       index=False)

    print(f"\nSaved: {sociodem_path}  {sociodem_df.shape}")
    print(f"Saved: {trips_path}  {trips_df.shape}")

    print(f"\nModal share:")
    print(trips_df["mode"].value_counts(normalize=True)
          .mul(100).round(1).to_string())

    print(f"\nTrips per person:")
    print(trips_df.groupby("person_id").size().describe().round(2))

    print(f"\nHouseholds per ward:")
    print(sociodem_df.groupby("ward_id")["hh_id"]
          .nunique().sort_values().to_string())