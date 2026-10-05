#!/usr/bin/env python3
"""Small YAML helpers for IMI monthly workflow scripts."""

import argparse
import calendar
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

IMI_SOURCE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(IMI_SOURCE_DIR))

from src.utilities.config_utils import load_config


def month_after(yyyymmdd):
    date = datetime.strptime(str(yyyymmdd), "%Y%m%d")
    year = date.year + (date.month == 12)
    month = 1 if date.month == 12 else date.month + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return f"{year:04d}{month:02d}{day:02d}"


def write_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def parse_value(value):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def set_key(config, key, value):
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def cmd_get(args):
    config = load_config(args.config)
    value = config.get(args.key, args.default)
    if value is None:
        return 1
    if isinstance(value, bool):
        print(str(value).lower())
    elif isinstance(value, (list, tuple)):
        print(" ".join(str(v) for v in value))
    else:
        print(os.path.expandvars(str(value)))
    return 0


def cmd_month_after(args):
    print(month_after(args.date))
    return 0


def cmd_write_temp(args):
    config = deepcopy(load_config(args.config, normalize=False))
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        set_key(config, key, parse_value(value))
    write_config(config, args.output)
    print(args.output)
    return 0


def cmd_count_elements(args):
    import numpy as np
    import xarray as xr

    config = load_config(args.config)
    state = xr.load_dataset(args.state_vector)["StateVector"].values
    labels = np.unique(state[np.isfinite(state)])
    n_elements = int(np.sum(labels > 0))
    if bool(config.get("OptimizeBCs", False)):
        n_elements += 4
    if bool(config.get("OptimizeOH", False)):
        n_elements += 1 if bool(config.get("isRegional", True)) else 2
    print(n_elements)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    get = subparsers.add_parser("get")
    get.add_argument("config")
    get.add_argument("key")
    get.add_argument("--default", default=None)
    get.set_defaults(func=cmd_get)

    next_month = subparsers.add_parser("month-after")
    next_month.add_argument("date")
    next_month.set_defaults(func=cmd_month_after)

    write_temp = subparsers.add_parser("write-temp")
    write_temp.add_argument("config")
    write_temp.add_argument("output")
    write_temp.add_argument("--set", action="append", default=[])
    write_temp.set_defaults(func=cmd_write_temp)

    count_elements = subparsers.add_parser("count-elements")
    count_elements.add_argument("config")
    count_elements.add_argument("state_vector")
    count_elements.set_defaults(func=cmd_count_elements)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
