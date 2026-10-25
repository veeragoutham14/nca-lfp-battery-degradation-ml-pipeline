"""Synthetic nameplate values from config/battery_profiles.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "battery_profiles.json"


@lru_cache(maxsize=1)
def _profiles() -> dict:
    return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))


def _variant(key: str) -> dict:
    return _profiles()["variants"][key]


FIELD_NOMINAL_AH = float(_variant("nca_home_storage")["nominal_capacity_ah"])
FIELD_MODULE_AH = float(_variant("nca_home_storage")["module_capacity_ah"])
FIELD_MODULE_KWH = float(_variant("nca_home_storage")["module_energy_kwh"])
LAB_NOMINAL_AH = float(_variant("nca_full_cycle_reference")["nominal_capacity_ah"])
LAB_MODULE_AH = float(_variant("nca_full_cycle_reference")["module_capacity_ah"])
LAB_MODULE_KWH = float(_variant("nca_full_cycle_reference")["module_energy_kwh"])


def nominal_capacity_ah_for_serial(serial: str) -> float:
    """Return the configured nominal capacity for a serial, or the field nominal."""
    for variant in _profiles()["variants"].values():
        for device in variant.get("devices", []):
            if str(device.get("serial")) == str(serial):
                value = variant.get("nominal_capacity_ah")
                if value is not None:
                    return float(value)
    return FIELD_NOMINAL_AH
