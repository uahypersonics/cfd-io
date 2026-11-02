"""Shared helpers for the cfd-io CLI commands."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import re
from pathlib import Path

import typer


# --------------------------------------------------
# helper: parse --attr key=value pairs
# --------------------------------------------------
def parse_attr_pairs(items: list[str]) -> dict[str, float | str]:
    """Parse ``key=value`` attribute pairs from the command line.

    Values are stored as float where they parse as a number, otherwise as
    the string given.

    Args:
        items: Pairs as typed, e.g. ``["mach=6.0", "case_name=blunt_cone"]``.

    Returns:
        Dict mapping attribute keys to float or string values.

    Raises:
        typer.BadParameter: If an item has no ``=``.
    """

    # split each pair on its first "=" and coerce numerics where possible
    attrs: dict[str, float | str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"expected key=value, got '{item}'")
        key, value = item.split("=", 1)
        try:
            attrs[key] = float(value)
        except ValueError:
            attrs[key] = value

    return attrs


# --------------------------------------------------
# helper: load attributes from a flow_conditions file
# --------------------------------------------------
def load_attrs_from_file(path: Path) -> dict[str, float | str]:
    """Load attributes from a flow conditions file (.dat legacy or .json).

    .json format: flat {"key": value} dict — values cast to float where possible.

    .dat legacy format: fixed-column lines like
        ``freestream Mach number, M      :    14.4  [-]``
    The short name after the last comma is used as the key (lowercased).
    For lines without a comma short-name the full description is snake_cased.
    The first numeric token before any ``[units]`` block is used as the value.
    String-valued lines (e.g. viscosity law name) are stored as strings.

    Args:
        path: Path to the attribute file (.dat or .json).

    Returns:
        Dict mapping attribute keys to float or string values.
    """
    # validate file exists
    if not path.exists():
        raise typer.BadParameter(f"attr-file not found: {path}")

    # read JSON format
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # cast values to float where possible
        attrs: dict[str, float | str] = {}
        for k, v in data.items():
            try:
                attrs[k] = float(v)
            except (ValueError, TypeError):
                attrs[k] = str(v)
        return attrs

    # read legacy .dat format
    if suffix == ".dat":
        attrs = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # skip blank lines and lines without a colon separator
                if not line or ":" not in line:
                    continue

                # split on first colon: left = description, right = value + units
                desc, _, value_part = line.partition(":")
                desc = desc.strip()
                value_part = value_part.strip()

                # extract key: last token after last comma in description (the short name)
                # example: "freestream Mach number, M" -> "m"
                # example: "Viscosity Option" -> "viscosity_option" (no comma)
                if "," in desc:
                    key = desc.rsplit(",", 1)[-1].strip().lower()
                else:
                    # snake_case the full description as fallback
                    key = re.sub(r"[^a-z0-9]+", "_", desc.lower()).strip("_")

                # skip if key is empty
                if not key:
                    continue

                # extract numeric value: strip unit block [...]  and dual-value slash
                # example: "14.4000 [-]"         -> "14.4000"
                # example: "119161873.55 / 17282.97 [Pa]/[psi]" -> "119161873.55"
                raw = value_part.split("[")[0].split("/")[0].strip()

                # store as float if numeric, otherwise as a stripped string
                try:
                    attrs[key] = float(raw)
                except ValueError:
                    attrs[key] = value_part.split("[")[0].strip()

        return attrs

    raise typer.BadParameter(
        f"unsupported attr-file format '{suffix}' — expected .dat or .json"
    )
