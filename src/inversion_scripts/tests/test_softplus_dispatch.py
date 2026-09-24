"""Solver selection: ``SoftplusErrors: true`` selects the softplus positivity solver (the documented
flag, parallel to ``LognormalErrors: true``); ``InversionMethod: softplus`` remains honored for
back-compat; the boolean flag wins when both are set; and the default is the analytical solver.
An unrecognized method raises a clear error.
"""
import pytest

from src.inversion_scripts.invert import resolve_inversion_method


def test_default_is_analytical():
    assert resolve_inversion_method({}) == "analytical"


def test_softplus_errors_flag_selects_softplus():
    assert resolve_inversion_method({"SoftplusErrors": True}) == "softplus"


def test_softplus_errors_false_is_analytical():
    assert resolve_inversion_method({"SoftplusErrors": False}) == "analytical"


def test_inversion_method_backcompat():
    assert resolve_inversion_method({"InversionMethod": "softplus"}) == "softplus"


def test_flag_overrides_method_string():
    assert resolve_inversion_method(
        {"InversionMethod": "analytical", "SoftplusErrors": True}
    ) == "softplus"


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        resolve_inversion_method({"InversionMethod": "banana"})
