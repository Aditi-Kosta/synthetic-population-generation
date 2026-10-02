"""
Diagnose the validation failure: find the exact consecutive ping pair that
exceeds 2km for the failing user, then trace back what phase each ping was in.
Run from: mumbai-synth-v2/generation/
"""
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

def haversine_simple(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

cdr   = pd.read_parquet("../data/raw/cdr/mumbai_cdr.parquet")
users = pd.read_parquet("../data/raw/cdr/users.parquet")

# Find the worst pair across ALL users (same logic as validation check 8)
rng_check    = np.random.default_rng(0)
all_uids     = cdr["user_id"].unique()
sample_users = rng_check.choice(all_uids, size=min(500, len(all_uids)), replace=False)
sample_cdr   = cdr[cdr["user_id"].isin(sample_users)]

max_d   = 0.0
fail_uid = ""
fail_i   = -1
fail_grp = None

for uid, grp in sample_cdr.groupby("user_id"):
    grp  = grp.sort_values("timestamp").reset_index(drop=True)
    lats = grp["tower_lat"].values
    lons = grp["tower_lon"].values
    for i in range(len(lats) - 1):
        d = haversine_simple(lats[i], lons[i], lats[i+1], lons[i+1])
        if d > max_d:
            max_d    = d
            fail_uid = uid
            fail_i   = i
            fail_grp = grp.copy()

print(f"Worst pair: {max_d:.2f} km — user {fail_uid}, ping index {fail_i}/{fail_i+1}")
print()

row_a = fail_grp.iloc[fail_i]
row_b = fail_grp.iloc[fail_i + 1]
print("Ping A:")
print(f"  timestamp : {row_a['timestamp']}")
print(f"  tower_id  : {row_a['tower_id']}")
print(f"  ward_id   : {row_a['ward_id']}")
print(f"  lat/lon   : {row_a['tower_lat']:.4f}, {row_a['tower_lon']:.4f}")

print("Ping B:")
print(f"  timestamp : {row_b['timestamp']}")
print(f"  tower_id  : {row_b['tower_id']}")
print(f"  ward_id   : {row_b['ward_id']}")
print(f"  lat/lon   : {row_b['tower_lat']:.4f}, {row_b['tower_lon']:.4f}")

# Look up user home/work
u = users[users["user_id"] == fail_uid].iloc[0]
print(f"\nUser home: ward={u['home_ward']}  lat={u['home_lat']:.4f}  lon={u['home_lon']:.4f}")
print(f"User work: ward={u['work_ward']}  lat={u['work_lat']:.4f}  lon={u['work_lon']:.4f}")
print(f"Commute dist: {haversine_simple(u['home_lat'], u['home_lon'], u['work_lat'], u['work_lon']):.2f} km")

# Check time-of-day phase for each ping
from datetime import datetime
def hour_val(ts):
    return ts.hour + ts.minute / 60.0

tsa = pd.to_datetime(row_a["timestamp"])
tsb = pd.to_datetime(row_b["timestamp"])
print(f"\nPing A hour: {hour_val(tsa):.2f}")
print(f"Ping B hour: {hour_val(tsb):.2f}")

# Show full day sorted pings around the jump
day_a = tsa.date()
day_pings = fail_grp[pd.to_datetime(fail_grp["timestamp"]).dt.date == day_a].sort_values("timestamp")
print(f"\nAll pings on {day_a} for user {fail_uid} (sorted by time):")
for _, r in day_pings.iterrows():
    ts = pd.to_datetime(r["timestamp"])
    hv = hour_val(ts)
    marker = " <-- JUMP" if r["tower_id"] in (row_a["tower_id"], row_b["tower_id"]) else ""
    print(f"  {ts.strftime('%H:%M')}  ward={r['ward_id']:5s}  tower={r['tower_id']}  "
          f"({r['tower_lat']:.4f},{r['tower_lon']:.4f}){marker}")