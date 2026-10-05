#!/usr/bin/env python3
"""Limit Jacobian GEOS-Chem output to diagnostics required by the satellite operator."""

import re
import sys
from pathlib import Path

import yaml


def load_bool(config_path, key, default=False):
    if config_path is None or not Path(config_path).exists():
        return default
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "on")
    return bool(value)


def transported_species(config_text):
    species = []
    capture = False
    for line in config_text.splitlines():
        if line.startswith("    transported_species:"):
            capture = True
            continue
        if capture:
            if line.startswith("      - "):
                species.append(line.split("- ", 1)[1].strip())
                continue
            if line.startswith("  ") and not line.startswith("      "):
                break
    return species


def replace_collection_fields(text, collection, values):
    lines = text.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.startswith(f"  {collection}.fields:"):
            start = idx
            end = idx + 1
            while end < len(lines) and lines[end].startswith("                              "):
                end += 1
            break
    if start is None:
        return text

    replacement = [f"  {collection}.fields:      '{values[0]}',"]
    replacement.extend(f"                              '{value}'," for value in values[1:])
    lines[start:end] = replacement
    return "\n".join(lines) + "\n"


def rename_collection(text, old, new):
    text = text.replace(f"'{old}'", f"'{new}'")
    text = text.replace(f"THE {old} COLLECTION", f"THE {new} COLLECTION")
    for suffix in ("template", "frequency", "duration", "mode", "fields"):
        text = text.replace(f"  {old}.{suffix}:", f"  {new}.{suffix}:")
    return text


def set_collection_enabled(text, collection, enabled):
    lines = text.splitlines()
    quoted = f"'{collection}'"
    for idx, line in enumerate(lines):
        if quoted not in line:
            continue
        leading, rest = line.split(quoted, 1)
        if enabled:
            leading = re.sub(r"#+", "", leading)
        elif "#" not in leading:
            leading = leading + "#"
        lines[idx] = leading + quoted + rest
    return "\n".join(lines) + "\n"


def replace_collection_value(text, collection, key, value):
    pattern = rf"(^  {re.escape(collection)}\.{re.escape(key)}:\s*).*$"
    return re.sub(pattern, lambda match: f"{match.group(1)}{value}", text, flags=re.MULTILINE)


def satdiagn_field_name(species_name):
    if species_name == "CH4":
        return "SatDiagnConc_CH4               "
    if re.fullmatch(r"CH4_\d{4}", species_name):
        return f"SatDiagnConc_{species_name}          "
    return f"SatDiagnConc_{species_name}               "


def main(run_dir, config_file=None):
    run_path = Path(run_dir)
    history_path = run_path / "HISTORY.rc"
    config_path = run_path / "geoschem_config.yml"

    history_text = history_path.read_text()
    config_text = config_path.read_text()

    species_fields = ["SpeciesConcVV_CH4             "]
    satdiagn_fields = [satdiagn_field_name("CH4")]
    for spc in transported_species(config_text):
        if re.fullmatch(r"CH4_\d{4}", spc):
            species_fields.append(f"SpeciesConcVV_{spc}        ")
            satdiagn_fields.append(satdiagn_field_name(spc))

    use_satdiagn_overpass = load_bool(config_file, "UseSatDiagnOverpass", False)

    if "LevelEdgeDiags" in history_text and "StateMetLevEdge" not in history_text:
        history_text = rename_collection(history_text, "LevelEdgeDiags", "StateMetLevEdge")

    history_text = set_collection_enabled(history_text, "Restart", True)
    history_text = set_collection_enabled(history_text, "SpeciesConc", not use_satdiagn_overpass)
    history_text = set_collection_enabled(history_text, "StateMetLevEdge", not use_satdiagn_overpass)
    history_text = set_collection_enabled(history_text, "SatDiagn", use_satdiagn_overpass)
    history_text = set_collection_enabled(history_text, "SatDiagnEdge", use_satdiagn_overpass)

    history_text = replace_collection_fields(history_text, "SpeciesConc", species_fields)
    history_text = replace_collection_fields(history_text, "SatDiagn", satdiagn_fields)
    history_text = replace_collection_fields(history_text, "SatDiagnEdge", ["SatDiagnPEDGE                  "])

    for collection in ("SpeciesConc", "StateMetLevEdge"):
        history_text = replace_collection_value(collection=collection, key="frequency", value="00000000 010000", text=history_text)
        history_text = replace_collection_value(collection=collection, key="duration", value="00000001 000000", text=history_text)
        history_text = replace_collection_value(collection=collection, key="mode", value="'instantaneous'", text=history_text)

    if use_satdiagn_overpass:
        for collection in ("SatDiagn", "SatDiagnEdge"):
            history_text = replace_collection_value(collection=collection, key="frequency", value="00000100 000000", text=history_text)
            history_text = replace_collection_value(collection=collection, key="duration", value="00000100 000000", text=history_text)
            history_text = replace_collection_value(collection=collection, key="mode", value="'time-averaged'", text=history_text)

    history_path.write_text(history_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
