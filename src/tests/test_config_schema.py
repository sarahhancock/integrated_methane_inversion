"""The softplus / correlated-Sa / correlated-So config keys must be registered in the input
sanitizer as OPTIONAL (type-checked if present) and NEVER required, so that (a) they are validated
and documented, and (b) existing configs that omit them still pass. Regression guard for the
easy mistake of dropping them into ``config_required``."""
import sanitize_input_yaml as s

# Keys introduced / surfaced for the productionized error methods.
NEW_KEYS = [
    # solver
    "SoftplusErrors", "InversionMethod", "SoftplusScale",
    # correlated observational-error covariance So
    "OffDiagonalObsCov", "OffDiagonalObsCovA1", "OffDiagonalObsCovL1",
    "OffDiagonalObsCovA2", "OffDiagonalObsCovL2", "OffDiagonalObsCovTemporalRho",
    "OffDiagonalObsCovCutoffKm", "UseResidualObsError",
    # correlated prior-error covariance Sa
    "NationalPriorGridNationalRatio", "NationalPriorMinUncertainty",
    "NationalPriorTwoComponent", "NationalPriorGlobalBackground",
    "SectorEnsembleTwoComponentSectors", "SectorEnsembleGenericSigma",
    "SectorEnsembleWetlandFile", "SectorEnsembleWetlandGlobalBackground",
    "SectorEnsembleGenericGlobalBackground",
]

BOOL_KEYS = [
    "SoftplusErrors", "OffDiagonalObsCov", "UseResidualObsError",
    "NationalPriorTwoComponent", "NationalPriorGlobalBackground",
    "SectorEnsembleWetlandGlobalBackground", "SectorEnsembleGenericGlobalBackground",
]


def test_new_keys_registered_as_optional():
    for k in NEW_KEYS:
        assert k in s.optional_rules, f"{k} is not registered in optional_rules"


def test_new_keys_not_required():
    # Must never be mandatory: a config that omits them has to keep validating.
    for k in NEW_KEYS:
        assert k not in s.config_required, f"{k} wrongly placed in config_required"


def test_boolean_keys_typed_bool():
    for k in BOOL_KEYS:
        assert s.optional_rules[k] is bool, f"{k} should be validated as bool"
