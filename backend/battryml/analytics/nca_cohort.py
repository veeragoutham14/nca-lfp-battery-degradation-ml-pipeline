from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_ACTIVE_MODULE_COUNT = 2
DEFAULT_REFERENCE_MODULE_COUNT = 2
DEFAULT_REFERENCE_SERIAL = "100000017"

CohortRole = Literal["field_device", "reference_control"]
CohortStatus = Literal["included", "deferred", "reference"]


@dataclass(frozen=True)
class NcaCohortMembership:
    cohort_role: CohortRole
    cohort_status: CohortStatus
    in_active_cohort: bool
    module_count: int | None
    reason: str


def positive_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or not number.is_integer():
        return None
    return int(number)


def classify_nca_device(
    device: dict[str, Any],
    *,
    active_module_count: int = DEFAULT_ACTIVE_MODULE_COUNT,
    reference_serial: str = DEFAULT_REFERENCE_SERIAL,
    reference_module_count: int = DEFAULT_REFERENCE_MODULE_COUNT,
) -> NcaCohortMembership | None:
    """Classify visible NCA devices without discarding module cohorts."""
    return classify_chemistry_device(
        device,
        chemistry="NCA",
        reference_serial=reference_serial,
        reference_module_count=reference_module_count,
        active_module_count=active_module_count,
    )


def classify_chemistry_device(
    device: dict[str, Any],
    *,
    chemistry: str,
    reference_serial: str | None = None,
    reference_module_count: int = DEFAULT_REFERENCE_MODULE_COUNT,
    active_module_count: int = DEFAULT_ACTIVE_MODULE_COUNT,
) -> NcaCohortMembership | None:
    """Classify one chemistry while retaining every verified module cohort."""
    if active_module_count <= 0 or reference_module_count <= 0:
        raise ValueError("module counts must be positive")
    chemistry_key = chemistry.strip().upper()
    if not chemistry_key:
        raise ValueError("chemistry is required")
    if str(device.get("chemistry") or "").upper() != chemistry_key:
        return None

    serial = str(device.get("serial") or "")
    role = str(device.get("device_role") or "")
    module_count = positive_int(device.get("module_count"))
    module_status = str(device.get("module_count_status") or "missing")

    if role == "full_cycle_test" and (reference_serial is None or serial == reference_serial):
        return NcaCohortMembership(
            cohort_role="reference_control",
            cohort_status="reference",
            in_active_cohort=True,
            module_count=module_count or reference_module_count,
            reason=(
                f"{module_count or reference_module_count}-module full-cycle reference control"
            ),
        )
    if role != "field_tester":
        return None

    if module_status != "verified" or module_count is None:
        return NcaCohortMembership(
            cohort_role="field_device",
            cohort_status="deferred",
            in_active_cohort=False,
            module_count=module_count,
            reason="Module count must be verified before cohort assignment",
        )
    return NcaCohortMembership(
        cohort_role="field_device",
        cohort_status="included",
        in_active_cohort=True,
        module_count=module_count,
        reason=f"Included in the verified {module_count}-module {chemistry_key} cohort",
    )
