from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = PROJECT_ROOT / "config" / "battery_profiles.json"
DEVICE_ROLES = ("field_tester", "full_cycle_test")


@dataclass(frozen=True)
class BatteryProfile:
    serial: str
    display_name: str
    variant_key: str
    chemistry: str
    device_role: str
    configured_module_count: int | None
    nominal_capacity_ah: float | None
    module_capacity_ah: float | None
    module_energy_kwh: float | None
    loss_slope_reference_module_count: int | None
    data_available_from: str | None
    default_loss_slopes_per_1000ah: tuple[float, ...]


def _optional_positive_float(value: Any, field: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _optional_positive_int(value: Any, field: str) -> int | None:
    number = _optional_positive_float(value, field)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _loss_slopes(value: Any, field: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    slopes = tuple(_optional_positive_float(item, field) for item in value)
    if any(item is None for item in slopes):
        raise ValueError(f"{field} cannot contain empty values")
    return tuple(float(item) for item in slopes if item is not None)


def _variant_devices(raw_variant: dict[str, Any], variant_key: str) -> list[dict[str, Any]]:
    raw_devices = raw_variant.get("devices")
    if isinstance(raw_devices, list) and raw_devices:
        devices: list[dict[str, Any]] = []
        for item in raw_devices:
            if not isinstance(item, dict):
                raise ValueError(f"{variant_key}.devices entries must be objects")
            devices.append(item)
        return devices

    raw_serials = raw_variant.get("serials")
    if isinstance(raw_serials, list) and raw_serials:
        return [{"serial": serial} for serial in raw_serials]
    raise ValueError(f"Battery variant {variant_key} needs devices")


def load_battery_profiles(path: Path = PROFILE_PATH) -> dict[str, BatteryProfile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("battery_profiles.json must contain a variants object")

    profiles: dict[str, BatteryProfile] = {}
    display_names: set[str] = set()
    for variant_key, raw_variant in variants.items():
        if not isinstance(raw_variant, dict):
            raise ValueError(f"Battery variant {variant_key} must be an object")
        chemistry = str(raw_variant.get("chemistry", "")).strip()
        device_role = str(raw_variant.get("device_role", "")).strip()
        nominal_capacity_ah = _optional_positive_float(
            raw_variant.get("nominal_capacity_ah"),
            f"{variant_key}.nominal_capacity_ah",
        )
        module_capacity_ah = _optional_positive_float(
            raw_variant.get("module_capacity_ah"),
            f"{variant_key}.module_capacity_ah",
        )
        module_energy_kwh = _optional_positive_float(
            raw_variant.get("module_energy_kwh"),
            f"{variant_key}.module_energy_kwh",
        )
        loss_slope_reference_module_count = _optional_positive_int(
            raw_variant.get("loss_slope_reference_module_count"),
            f"{variant_key}.loss_slope_reference_module_count",
        )
        slopes = _loss_slopes(
            raw_variant.get("default_loss_slopes_per_1000ah"),
            f"{variant_key}.default_loss_slopes_per_1000ah",
        )
        if not chemistry:
            raise ValueError(f"Battery variant {variant_key} needs chemistry")
        if device_role not in DEVICE_ROLES:
            raise ValueError(
                f"Battery variant {variant_key} device_role must be one of: "
                f"{', '.join(DEVICE_ROLES)}"
            )

        for raw_device in _variant_devices(raw_variant, str(variant_key)):
            serial = str(raw_device.get("serial", "")).strip()
            if not serial.isdigit():
                raise ValueError(f"Invalid battery serial in {variant_key}: {serial}")
            if serial in profiles:
                raise ValueError(f"Battery serial is assigned more than once: {serial}")
            display_name = str(raw_device.get("display_name") or f"Battery {serial}").strip()
            if not display_name:
                raise ValueError(f"Battery {serial} needs a display_name")
            if display_name.casefold() in display_names:
                raise ValueError(f"Battery display name is assigned more than once: {display_name}")
            display_names.add(display_name.casefold())
            available_from = raw_device.get("data_available_from")
            configured_module_count = _optional_positive_int(
                raw_device.get("module_count"),
                f"{variant_key}.devices[{serial}].module_count",
            )
            profiles[serial] = BatteryProfile(
                serial=serial,
                display_name=display_name,
                variant_key=str(variant_key),
                chemistry=chemistry,
                device_role=device_role,
                configured_module_count=configured_module_count,
                nominal_capacity_ah=nominal_capacity_ah,
                module_capacity_ah=module_capacity_ah,
                module_energy_kwh=module_energy_kwh,
                loss_slope_reference_module_count=loss_slope_reference_module_count,
                data_available_from=(
                    str(available_from).strip() if available_from is not None else None
                ),
                default_loss_slopes_per_1000ah=slopes,
            )
    return profiles


@lru_cache(maxsize=1)
def battery_profiles() -> dict[str, BatteryProfile]:
    return load_battery_profiles()


def battery_profile(serial: str | int) -> BatteryProfile | None:
    return battery_profiles().get(str(serial).strip())


def effective_nominal_capacity_ah(
    profile: BatteryProfile | None,
    module_count: int | None,
    module_count_status: str,
) -> float | None:
    if profile is None:
        return None
    if (
        module_count_status == "verified"
        and module_count is not None
        and module_count > 2
        and profile.module_capacity_ah is not None
    ):
        return module_count * profile.module_capacity_ah
    return profile.nominal_capacity_ah


def nominal_capacity_defaults() -> tuple[tuple[str, float], ...]:
    return tuple(
        (serial, profile.nominal_capacity_ah)
        for serial, profile in sorted(battery_profiles().items())
        if profile.nominal_capacity_ah is not None
    )


def loss_slope_defaults() -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            serial,
            ",".join(f"{slope:.12g}" for slope in profile.default_loss_slopes_per_1000ah),
        )
        for serial, profile in sorted(battery_profiles().items())
        if profile.default_loss_slopes_per_1000ah
    )
