"""`SoftplusErrors: true` selects the softplus positivity solver (the documented flag, parallel to
`LognormalErrors: true`); `InversionMethod: softplus` remains honored for back-compat; and the
boolean flag wins when both are set. Default is the analytical solver."""
from invert import resolve_inversion_method


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
