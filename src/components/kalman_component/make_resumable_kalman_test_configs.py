#!/usr/bin/env python3
"""Generate IMI config setups for testing Kalman workflow dispatch variants."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CASES = {
    "00_standard_non_kalman.yml": {
        "KalmanMode": False,
        "ResumableMonthlyKalman": False,
        "DoJacobian": True,
        "DoInversion": True,
        "DoPosterior": False,
    },
    "01_existing_period_kalman.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": False,
        "DoJacobian": True,
        "DoInversion": True,
        "DoPosterior": True,
    },
    "02_resumable_obs_products.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "obs_products",
    },
    "03_resumable_inversion_all_features.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanRequireResidualSo": True,
        "MonthlyKalmanPreserveResidualSo": True,
        "MonthlyKalmanNudgeFactor": 0.2,
        "MonthlyKalmanAdjustPriorWithScaleFactors": True,
        "MonthlyKalmanClipNegativeK": True,
    },
    "04_resumable_inversion_default_so.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanRequireResidualSo": False,
        "MonthlyKalmanPreserveResidualSo": False,
    },
    "05_resumable_inversion_no_prior_nudge.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanNudgeFactor": 0.0,
    },
    "06_resumable_inversion_no_prior_adjustment.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanAdjustPriorWithScaleFactors": False,
    },
    "07_resumable_inversion_no_negative_k_clip.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanClipNegativeK": False,
    },
    "08_resumable_inversion_no_so_preservation.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "MonthlyKalmanPreserveResidualSo": False,
    },
    "09_resumable_inversion_satdiagn_overpass.yml": {
        "KalmanMode": True,
        "ResumableMonthlyKalman": True,
        "MonthlyKalmanStage": "inversion",
        "UseSatDiagnOverpass": True,
    },
}


COMMON_OVERRIDES = {
    "SafeMode": False,
    "StartDate": 20190101,
    "EndDate": 20190201,
    "FinalDate": 20190401,
    "UpdateFreqDays": 31,
    "NudgeFactor": 0.2,
    "DynamicKFClustering": False,
    "MakePeriodsCSV": True,
    "RunSetup": False,
    "SetupTemplateRundir": False,
    "SetupSpinupRun": False,
    "SetupJacobianRuns": False,
    "SetupInversion": False,
    "SetupPosteriorRun": False,
    "DoHemcoPriorEmis": False,
    "DoSpinup": False,
    "ReDoJacobian": False,
    "HemcoPriorEmisDryRun": True,
    "SpinupDryrun": True,
    "ProductionDryRun": True,
    "PosteriorDryRun": True,
    "BCdryrun": True,
    "MonthlyKalmanEndDate": 20190401,
    "MonthlyKalmanStartDate": 20190101,
    "MonthlyKalmanNudgeFactor": 0.2,
    "MonthlyKalmanRequireResidualSo": True,
    "MonthlyKalmanPreserveResidualSo": True,
    "MonthlyKalmanAdjustPriorWithScaleFactors": True,
    "MonthlyKalmanClipNegativeK": True,
    "MonthlyKalmanDryRun": True,
    "UseSatDiagnOverpass": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_config",
        nargs="?",
        default="config.yml",
        help="Base IMI config to copy and override.",
    )
    parser.add_argument(
        "--outdir",
        default="src/components/kalman_component/test_setups/generated",
        help="Directory where generated test configs are written.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def make_config(base, case_name, overrides):
    config = deepcopy(base)
    config.update(COMMON_OVERRIDES)
    config.update(overrides)
    stem = Path(case_name).stem
    config["RunName"] = f"imi_{stem}"
    return config


def main():
    args = parse_args()
    base = load_yaml(args.base_config)
    outdir = Path(args.outdir)

    written = []
    for case_name, overrides in DEFAULT_CASES.items():
        config = make_config(base, case_name, overrides)
        outpath = outdir / case_name
        dump_yaml(config, outpath)
        written.append(outpath)

    manifest = {
        "base_config": str(Path(args.base_config).resolve()),
        "cases": [str(path) for path in written],
    }
    dump_yaml(manifest, outdir / "manifest.yml")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
