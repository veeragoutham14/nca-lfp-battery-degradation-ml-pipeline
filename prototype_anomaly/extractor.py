from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG - EDIT THIS
# =========================
DATA_DIR = Path(r"CSV LOG DATA")  # <-- change
OUT_DIR = Path(r"Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_GLOB = "*.csv"

# Core columns
TS_COL = "timestamp"
Y_COL = "batt.soc_pct"

X_RAW = [
    "batt.soh_pct",
    "batt.current_A",
    "batt.voltage_V",
    "batt.power_W",  # optional; keep for now, we’ll check redundancy
]

# Feature engineering params
REST_A = 0.5
ROLL_S = 120  # seconds

# =========================
# READER FOR YOUR BLM CSV
# =========================
def read_logger_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,           # config lines
        sep=";",
        decimal=",",          # German decimal comma
        engine="python",
        on_bad_lines="skip",  # skip rare broken lines
        #low_memory=False
    )
    # header starts with "# timestamp" -> "timestamp"
    df = df.rename(columns=lambda c: c.lstrip("# ").strip())
    df["__source_file"] = path.name
    return df

# =========================
# LOAD ALL FILES
# =========================
files = sorted(DATA_DIR.glob(CSV_GLOB))
if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

dfs = []
bad = []
for f in files:
    try:
        d = read_logger_csv(f)
        dfs.append(d)
        print(f"Loaded {f.name}: {d.shape}")
    except Exception as e:
        bad.append((f.name, str(e)))

if bad:
    print("\nFiles that failed to load:")
    for name, err in bad:
        print("-", name, ":", err)

df = pd.concat(dfs, ignore_index=True)
print("\nCombined shape:", df.shape)

# =========================
# KEEP ONLY NEEDED COLUMNS
# =========================
need = [TS_COL, Y_COL] + X_RAW + ["__source_file"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError("Missing required columns:\n" + "\n".join(missing))

df = df[need].copy()

# Parse timestamp (format: 2026.03.02 06:07:31)
df[TS_COL] = pd.to_datetime(df[TS_COL], format="%Y.%m.%d %H:%M:%S", errors="coerce")
df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)

# Ensure numeric
for c in [Y_COL] + X_RAW:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows missing core signals
core = [Y_COL, "batt.current_A", "batt.voltage_V", "batt.soh_pct"]
df = df.dropna(subset=core).reset_index(drop=True)

# =========================
# FEATURE ENGINEERING
# =========================
I = "batt.current_A"
V = "batt.voltage_V"
P = "batt.power_W"

# dt
df["dt_s"] = df[TS_COL].diff().dt.total_seconds()
med_dt = df["dt_s"].dropna().median()
df["dt_s"] = df["dt_s"].fillna(med_dt)
df.loc[df["dt_s"] <= 0, "dt_s"] = med_dt

# derivatives + magnitudes
df["current_abs_A"] = df[I].abs()
df["current_sq_A2"] = df[I] ** 2

df["dI_A"] = df[I].diff()
df["dV_V"] = df[V].diff()

df["dI_dt_Aps"] = df["dI_A"] / df["dt_s"]
df["dV_dt_Vps"] = df["dV_V"] / df["dt_s"]

df["is_rest"] = (df["current_abs_A"] < REST_A).astype(int)

# rolling features (time-based)
df = df.set_index(TS_COL)
win = f"{ROLL_S}s"
df[f"I_mean_{ROLL_S}s"] = df[I].rolling(win, min_periods=3).mean()
df[f"V_mean_{ROLL_S}s"] = df[V].rolling(win, min_periods=3).mean()
df = df.reset_index()

# drop early NaNs from rolling
feature_cols = [
    "batt.soh_pct",
    I, V, P,
    "dt_s",
    "current_abs_A", "current_sq_A2",
    "dI_dt_Aps", "dV_dt_Vps",
    f"I_mean_{ROLL_S}s", f"V_mean_{ROLL_S}s",
    "is_rest",
]
df_feat = df.dropna(subset=feature_cols + [Y_COL]).copy()

print("Rows after feature eng:", len(df_feat))
print("Feature columns:", feature_cols)

# Save extracted dataset (CSV)
out_csv = OUT_DIR / "soc_features_extracted.csv"
df_feat.to_csv(out_csv, index=False)
print("Saved extracted dataset to:", out_csv)



print("Done.")

print("\nPreview of extracted dataset:\n")
print(df_feat.head(20))

print("\nColumn names:\n")
print(df_feat.columns)

print("\nDataset shape:", df_feat.shape)

# =========================
# VISUAL CHECKS
# =========================
# Plot a manageable sample (first 10k rows)
#n = min(10000, len(df_feat))
s = df_feat.iloc[::10]   # every 10th sample
t = s[TS_COL]



# 2) SOC / Current / Voltage over time
plt.figure()
plt.plot(t, s[Y_COL])
plt.title("Target SOC over time (sample)")
plt.xlabel("time")
plt.ylabel("soc_pct")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, s[I])
plt.title("Current over time (sample)")
plt.xlabel("time")
plt.ylabel("current_A")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, s[V])
plt.title("Voltage over time (sample)")
plt.xlabel("time")
plt.ylabel("voltage_V")
plt.tight_layout()
plt.show()

# 1) Sampling interval
plt.figure()
plt.hist(df_feat["dt_s"].clip(upper=df_feat["dt_s"].quantile(0.99)), bins=60)
plt.title("dt_s distribution (clipped at 99th percentile)")
plt.xlabel("dt_s")
plt.ylabel("count")
plt.tight_layout()
plt.show()

# 3) Voltage vs SOC during rest only
rest = df_feat[df_feat["is_rest"] == 1]
if len(rest) > 0:
    r = rest.sample(n=min(5000, len(rest)), random_state=0)
    plt.figure()
    plt.scatter(r[V], r[Y_COL], s=6)
    plt.title("Voltage vs SOC (rest points)")
    plt.xlabel("voltage_V")
    plt.ylabel("soc_pct")
    plt.tight_layout()
    plt.show()
else:
    print("No rest points found with REST_A =", REST_A)

# 4) Check redundancy: power vs V*I
if df_feat[P].notna().any():
    approx = df_feat[V] * df_feat[I]
    err = (df_feat[P] - approx)
    plt.figure()
    plt.hist(err.dropna().clip(err.quantile(0.01), err.quantile(0.99)), bins=80)
    plt.title("power_W - (V*I) error (clipped)")
    plt.xlabel("W")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()