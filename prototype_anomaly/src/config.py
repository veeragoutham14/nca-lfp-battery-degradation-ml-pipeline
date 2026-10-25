from pathlib import Path

# =========================
# PATHS
# =========================
DATA_DIR = Path(r"data/raw")

# Keep processed data persistent
OUT_DIR = Path(r"data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporary run outputs to archive + clear
ANOMALY_OUT_DIR = Path(r"data/models")
ANOMALY_OUT_DIR.mkdir(parents=True, exist_ok=True)

#EXCEL_VIEW_DIR = Path(r"data/Excel_view")
#EXCEL_VIEW_DIR.mkdir(parents=True, exist_ok=True)

# External archive location
ARCHIVE_BASE_DIR = Path(r"run_archive")
ARCHIVE_BASE_DIR.mkdir(parents=True, exist_ok=True)

CSV_GLOB = "*.csv"
FEATURE_OUTPUT_FILENAME = "anomaly_features_extracted.csv"
ANOMALY_ALL_OUTPUT_FILENAME = "anomaly_results_all.csv"
ANOMALY_ONLY_OUTPUT_FILENAME = "anomaly_results_only.csv"
ANOMALY_TOP20_OUTPUT_FILENAME = "anomaly_top20.csv"

# =========================
# CORE COLUMNS
# =========================
TS_COL = "timestamp"

# Context / operating features
SOC_COL = "batt.soc_pct"
SOH_COL = "batt.soh_pct"
CURRENT_COL = "batt.current_A"
VOLTAGE_COL = "batt.voltage_V"
POWER_COL = "batt.power_W" # optional 

# Voltage spread features
CELL_VOLT_MAX_COL = "batt.cellVoltageMax_V"
CELL_VOLT_MIN_COL = "batt.cellVoltageMin_V"
# COMP_CELL_VOLT_MAX_COL = "batt.compensatedCellVoltageMax_V" # only for soc (OCV)
# COMP_CELL_VOLT_MIN_COL = "batt.compensatedCellVoltageMin_V" # only for soc (OCV)

# =========================
# PACK TEMPERATURE SENSOR GROUPS
# 4 modules × 2 packs × 4 sensors = 8 packs, 32 sensors
# =========================
PACK_TEMP_GROUPS = {
    "mod1_pack1": [
        "batt.batteryModules.mod1.pack1.temperature0_C",
        "batt.batteryModules.mod1.pack1.temperature1_C",
        "batt.batteryModules.mod1.pack1.temperature2_C",
        "batt.batteryModules.mod1.pack1.temperature3_C",
    ],
    "mod1_pack2": [
        "batt.batteryModules.mod1.pack2.temperature0_C",
        "batt.batteryModules.mod1.pack2.temperature1_C",
        "batt.batteryModules.mod1.pack2.temperature2_C",
        "batt.batteryModules.mod1.pack2.temperature3_C",
    ],
    "mod2_pack1": [
        "batt.batteryModules.mod2.pack1.temperature0_C",
        "batt.batteryModules.mod2.pack1.temperature1_C",
        "batt.batteryModules.mod2.pack1.temperature2_C",
        "batt.batteryModules.mod2.pack1.temperature3_C",
    ],
    "mod2_pack2": [
        "batt.batteryModules.mod2.pack2.temperature0_C",
        "batt.batteryModules.mod2.pack2.temperature1_C",
        "batt.batteryModules.mod2.pack2.temperature2_C",
        "batt.batteryModules.mod2.pack2.temperature3_C",
    ],
    "mod3_pack1": [
        "batt.batteryModules.mod3.pack1.temperature0_C",
        "batt.batteryModules.mod3.pack1.temperature1_C",
        "batt.batteryModules.mod3.pack1.temperature2_C",
        "batt.batteryModules.mod3.pack1.temperature3_C",
    ],
    "mod3_pack2": [
        "batt.batteryModules.mod3.pack2.temperature0_C",
        "batt.batteryModules.mod3.pack2.temperature1_C",
        "batt.batteryModules.mod3.pack2.temperature2_C",
        "batt.batteryModules.mod3.pack2.temperature3_C",
    ],
    "mod4_pack1": [
        "batt.batteryModules.mod4.pack1.temperature0_C",
        "batt.batteryModules.mod4.pack1.temperature1_C",
        "batt.batteryModules.mod4.pack1.temperature2_C",
        "batt.batteryModules.mod4.pack1.temperature3_C",
    ],
    "mod4_pack2": [
        "batt.batteryModules.mod4.pack2.temperature0_C",
        "batt.batteryModules.mod4.pack2.temperature1_C",
        "batt.batteryModules.mod4.pack2.temperature2_C",
        "batt.batteryModules.mod4.pack2.temperature3_C",
    ],
}

MODULE_PACK_GROUPS = {
    "mod1": ["mod1_pack1", "mod1_pack2"],
    "mod2": ["mod2_pack1", "mod2_pack2"],
    "mod3": ["mod3_pack1", "mod3_pack2"],
    "mod4": ["mod4_pack1", "mod4_pack2"],
}

PACK_NAMES = list(PACK_TEMP_GROUPS.keys())
MODULE_NAMES = list(MODULE_PACK_GROUPS.keys())

# Flat list of all 32 raw temperature sensor columns
TEMP_SENSOR_COLS = [
    col
    for pack_cols in PACK_TEMP_GROUPS.values()
    for col in pack_cols
]

# =========================
# OPTIONAL DETAILED CELL VOLTAGE SENSORS
# =========================
# Start with the precomputed voltage spread features above.
# Add detailed per-cell voltages later if needed.

# =========================
# RAW FEATURE GROUPS
# =========================
CONTEXT_FEATURES = [
    CURRENT_COL,
    VOLTAGE_COL,
]



VOLTAGE_SPREAD_FEATURES = [
    CELL_VOLT_MAX_COL,
    CELL_VOLT_MIN_COL,
]

X_RAW = (
    CONTEXT_FEATURES
    + VOLTAGE_SPREAD_FEATURES
    + TEMP_SENSOR_COLS
)

# =========================
# FEATURE ENGINEERING PARAMS
# =========================
REST_A = 0.5
ROLL_S = 120  # seconds
TIMESTAMP_FORMAT = "%Y.%m.%d %H:%M:%S"

# =========================
# ANOMALY DETECTION PARAMS
# =========================
ANOMALY_CONTAMINATION = 0.01   # initial guess for Isolation Forest
RANDOM_STATE = 42


ANOMALY_FEATURES = [

    #mode awareness
    "is_rest",
    "is_charging",
    "is_discharging",

    # Pack-level thermal imbalance
    "pack_temp_spread_C",
    "pack_temp_std_C",
    "pack_max_dev_from_median_C",
    "max_intra_module_pack_delta_C",

    # Internal pack non-uniformity
    "max_pack_internal_spread_C",
    "mean_pack_internal_spread_C",
    "max_pack_internal_std_C",
    "mean_pack_internal_std_C",
    "max_pack_hotspot_delta_C",
    "mean_pack_hotspot_delta_C",

    # Thermal persistence
    f"pack_temp_spread_mean_{ROLL_S}s",
    f"pack_temp_std_mean_{ROLL_S}s",
    f"pack_max_dev_from_median_mean_{ROLL_S}s",
    f"max_intra_module_pack_delta_mean_{ROLL_S}s",

    # Electrical imbalance
    "cell_voltage_spread_V",                            # difference between max cell voltage and min cell voltage 
    f"Vspread_mean_{ROLL_S}s",

]