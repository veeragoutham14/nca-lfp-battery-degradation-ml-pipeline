from __future__ import annotations

from battryml.analytics.nca_cohort import classify_chemistry_device, classify_nca_device


def _device(
    serial: str,
    *,
    module_count: int | None,
    status: str = "verified",
    role: str = "field_tester",
) -> dict[str, object]:
    return {
        "serial": serial,
        "chemistry": "NCA",
        "device_role": role,
        "module_count": module_count,
        "module_count_status": status,
    }


def test_cohort_classifier_includes_all_verified_module_variants() -> None:
    active = classify_nca_device(_device("two", module_count=2))
    three_module = classify_nca_device(_device("three", module_count=3))
    four_module = classify_nca_device(_device("four", module_count=4))
    unverified = classify_nca_device(
        _device("unknown", module_count=None, status="missing")
    )
    reference = classify_nca_device(
        _device("100000017", module_count=None, status="missing", role="full_cycle_test")
    )

    assert active is not None and active.in_active_cohort is True
    assert active.cohort_status == "included"
    assert three_module is not None and three_module.in_active_cohort is True
    assert three_module.cohort_status == "included"
    assert "3-module" in three_module.reason
    assert four_module is not None and four_module.in_active_cohort is True
    assert four_module.cohort_status == "included"
    assert "4-module" in four_module.reason
    assert unverified is not None and unverified.cohort_status == "deferred"
    assert reference is not None and reference.cohort_role == "reference_control"
    assert reference.module_count == 2


def test_chemistry_classifier_keeps_nca_and_lfp_separate() -> None:
    nca = _device("nca", module_count=3)
    lfp = {**_device("lfp", module_count=4), "chemistry": "LFP"}

    assert classify_chemistry_device(nca, chemistry="LFP") is None
    membership = classify_chemistry_device(lfp, chemistry="LFP")
    assert membership is not None
    assert membership.in_active_cohort is True
    assert membership.module_count == 4
