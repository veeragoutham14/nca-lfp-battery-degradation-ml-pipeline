import pandas as pd

from src.config import (
    TS_COL,
    X_RAW,
    CURRENT_COL,
    VOLTAGE_COL,
    TIMESTAMP_FORMAT,
    TEMP_SENSOR_COLS,
    VOLTAGE_SPREAD_FEATURES,
    PACK_TEMP_GROUPS,
    MODULE_PACK_GROUPS,
)


MIN_VALID_TEMP_SENSORS_PER_PACK = 4
MIN_REQUIRED_MODULES = 1


def get_complete_pack_groups(columns: pd.Index | list[str]) -> dict[str, list[str]]:
    """
    Ermittelt alle Pack-Gruppen, deren Temperatursensoren im Datensatz vollstaendig vorhanden sind.

    Ein Pack gilt hier nur dann als verfuegbar, wenn alle erwarteten
    Temperatursensor-Spalten im geladenen Datensatz existieren. Dadurch kann die
    Pipeline unterschiedliche Hardware-Layouts robust verarbeiten, ohne fuer
    nicht vorhandene Module oder Packs zu scheitern.
    """
    column_set = set(columns)
    complete_packs = {}

    for pack_name, sensor_cols in PACK_TEMP_GROUPS.items():
        if all(col in column_set for col in sensor_cols):
            complete_packs[pack_name] = sensor_cols

    return complete_packs


def get_complete_module_groups(complete_packs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Ermittelt alle Module, fuer die beide definierten Packs vollstaendig vorhanden sind.

    Ein Modul wird nur dann als verfuegbar betrachtet, wenn beide Pack-Namen aus
    `MODULE_PACK_GROUPS` im Datensatz als vollstaendige Pack-Gruppen erkannt
    wurden.
    """
    complete_modules = {}

    for module_name, pack_names in MODULE_PACK_GROUPS.items():
        if all(pack_name in complete_packs for pack_name in pack_names):
            complete_modules[module_name] = pack_names

    return complete_modules


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beschraenkt den Rohdatensatz auf die fuer die Pipeline benoetigten Spalten.

    Die Funktion baut aus Timestamp, definierten Rohmerkmalen und der
    Quelldatei-Spalte die Soll-Liste auf und prueft gleichzeitig, ob etwas
    Wesentliches fehlt. Fehlen Pflichtspalten, wird fruehzeitig ein klarer
    Fehler geworfen. Sind alle Spalten vorhanden, wird ein sauberer Ausschnitt
    des DataFrames zurueckgegeben.

    Aufrufkontext:
    Dies ist der erste Schritt in `prepare_base_dataframe`.
    """
    complete_packs = get_complete_pack_groups(df.columns)
    complete_modules = get_complete_module_groups(complete_packs)

    if len(complete_modules) < MIN_REQUIRED_MODULES:
        raise ValueError(
            "No supported battery module layout found in raw data.\n"
            f"Detected complete packs: {sorted(complete_packs.keys())}\n"
            "Need at least one full module with both packs present."
        )

    available_temp_cols = [
        col
        for pack_name in complete_packs
        for col in PACK_TEMP_GROUPS[pack_name]
    ]

    need = [TS_COL] + X_RAW[: len(X_RAW) - len(TEMP_SENSOR_COLS)] + available_temp_cols + ["__source_file"]
    missing = [c for c in need if c not in df.columns]

    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    return df[need].copy()


def parse_and_sort_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wandelt die Timestamp-Spalte in echte Zeitwerte um und sortiert chronologisch.

    Ungueltige Zeitstempel werden verworfen, damit nachfolgende Schritte nur mit
    konsistenten Zeitreihen arbeiten. Durch die Sortierung ist sichergestellt,
    dass Differenzen, Rolling-Windows und andere zeitbezogene Berechnungen auf
    der richtigen Reihenfolge basieren.

    Aufrufkontext:
    Wird in `prepare_base_dataframe` vor allen zeitbasierten Berechnungen
    ausgefuehrt.
    """
    df[TS_COL] = pd.to_datetime(
        df[TS_COL],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)
    return df


def get_numeric_columns() -> list[str]:
    """
    Liefert die Menge der Rohspalten, die numerisch interpretiert werden muessen.

    Dazu gehoeren insbesondere Temperatursensoren, Spannungs-Spreizungsinputs
    sowie die zentralen elektrischen Signale. Die Funktion enthaelt die
    Definition an einer Stelle, damit die eigentliche Typkonvertierung spaeter
    nicht mehrere verteilte Listen pflegen muss.

    Aufrufkontext:
    Interne Hilfsfunktion fuer `convert_numeric_columns`.
    """
    numeric_cols = (
        TEMP_SENSOR_COLS
        + VOLTAGE_SPREAD_FEATURES
        + [
            CURRENT_COL,
            VOLTAGE_COL,
        ]
    )

    seen = set()
    unique_numeric_cols = []
    for col in numeric_cols:
        if col not in seen:
            unique_numeric_cols.append(col)
            seen.add(col)

    return unique_numeric_cols


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert alle relevanten Rohsignalspalten in numerische Werte.

    Nicht interpretierbare Eintraege werden dabei bewusst zu `NaN`, damit sie in
    spaeteren Validierungs- oder Filter-Schritten sauber erkannt werden koennen.
    So wird verhindert, dass fehlerhafte String- oder Mischformate unbemerkt in
    die Feature-Berechnung gelangen.

    Aufrufkontext:
    Wird in `prepare_base_dataframe` nach der Spaltenauswahl ausgefuehrt.
    """
    numeric_cols = get_numeric_columns()

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_sufficient_pack_temperature_mask(
    df: pd.DataFrame,
    pack_groups: dict[str, list[str]],
) -> pd.Series:
    """
    Prueft fuer alle Zeilen gleichzeitig, ob jedes verfuegbare Pack genug gueltige Temperatursensoren hat.

    Die Logik entspricht der bisherigen zeilenweisen Pruefung: Fuer jedes Pack
    wird pro Zeile gezaehlt, wie viele Sensorwerte nicht fehlen. Eine Zeile
    bleibt nur dann erhalten, wenn in jedem beruecksichtigten Pack mindestens
    `MIN_VALID_TEMP_SENSORS_PER_PACK` gueltige Werte vorhanden sind.

    Die Umsetzung ist bewusst vektorisiert, damit auch sehr grosse Datensaetze
    performant verarbeitet werden koennen.
    """
    mask = pd.Series(True, index=df.index)

    for cols in pack_groups.values():
        pack_valid_mask = df[cols].notna().sum(axis=1) >= MIN_VALID_TEMP_SENSORS_PER_PACK
        mask &= pack_valid_mask

    return mask


def drop_rows_missing_core_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt alle Zeilen, die fuer verlaessliches Feature Engineering ungeeignet sind.

    Zunaechst werden Zeilen ohne zentrale elektrische Signale verworfen.
    Anschliessend werden Zeilen ohne die benoetigten Spannungs-Spreizungsinputs
    ausgeschlossen. Zum Schluss wird ueber `has_sufficient_pack_temperature`
    sichergestellt, dass auch die Temperaturabdeckung pro Pack hoch genug ist.
    Das Ergebnis ist ein deutlich robusterer Eingabedatensatz fuer die
    nachfolgenden Engineering-Schritte.

    Aufrufkontext:
    Wird in `prepare_base_dataframe` nach der numerischen Typkonvertierung
    ausgefuehrt.
    """
    pack_groups = get_complete_pack_groups(df.columns)
    module_groups = get_complete_module_groups(pack_groups)

    if len(module_groups) < MIN_REQUIRED_MODULES:
        raise ValueError(
            "No supported battery module layout remains after column selection.\n"
            f"Detected complete packs: {sorted(pack_groups.keys())}\n"
            "Need at least one full module with both packs present."
        )

    # Only keep what actually matters
    core = [CURRENT_COL, VOLTAGE_COL]

    df = df.dropna(subset=core).reset_index(drop=True)

    # Require voltage spread inputs
    df = df.dropna(subset=VOLTAGE_SPREAD_FEATURES).reset_index(drop=True)

    # Require proper temperature coverage (pack-aware)
    mask = get_sufficient_pack_temperature_mask(df, pack_groups=pack_groups)
    df = df[mask].reset_index(drop=True)

    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuehrt die komplette Vorverarbeitung fuer den zusammengefuehrten Rohdatensatz aus.

    Diese Funktion kapselt die einzelnen Preprocessing-Schritte in der richtigen
    Reihenfolge: Pflichtspalten auswaehlen, Zeitstempel parsen, numerische
    Konvertierung durchfuehren und unbrauchbare Zeilen entfernen. Das Ergebnis
    ist die stabile Ausgangsbasis fuer das gesamte nachfolgende Feature
    Engineering.

    Aufrufkontext:
    Wird in `src.main.extract` unmittelbar vor `build_feature_table` aufgerufen.
    """
    df = select_required_columns(df)
    df = parse_and_sort_timestamp(df)
    df = convert_numeric_columns(df)
    df = drop_rows_missing_core_signals(df)
    return df
