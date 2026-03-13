"""Canonical variable names and alias mappings.

Single source of truth for variable-name normalization across all
readers.  Every reader imports from here instead of defining its
own alias table.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

# --------------------------------------------------
# define grid coordinate names (frozen -> immutable)
# helps to split grid vs flow variables, and to check for valid grid names in readers
# --------------------------------------------------
GRID_NAMES: frozenset[str] = frozenset({"x", "y", "z"})

# --------------------------------------------------
# variable alias table
# to convert common variable name variants to a canonical name used internally
# --------------------------------------------------
VAR_ALIASES: dict[str, str] = {
    # grid coordinate aliases
    "x-grid": "x",
    "y-grid": "y",
    "z-grid": "z",
    # velocity
    "u": "uvel",
    "u-velocity": "uvel",
    "v": "vvel",
    "v-velocity": "vvel",
    "w": "wvel",
    "w-velocity": "wvel",
    # thermodynamic state
    "p": "pres",
    "pressure": "pres",
    "t": "temp",
    "temperature": "temp",
    "rho": "dens",
    "density": "dens",
    # mach number
    "m": "mach",
    # cgns standard names
    "coordinatex": "x",
    "coordinatey": "y",
    "coordinatez": "z",
    "velocitymagnitude": "vel_mag",
    "axial_velocity": "uvel",
    "radial_velocity": "vvel",
    "temperaturestagnation": "temp_stag",
    "coefpressure": "cp",
    # su2 variable names
    "velocity": "vel",
    "skin_friction_coefficient": "cf",
    "pressure_coefficient": "cp",
    "laminar_viscosity": "mu",
    "heat_flux": "qw",
}

# --------------------------------------------------
# vector component names
# when a reader splits a multi-component array, it looks here first;
# default fallback is base_x, base_y, base_z
# --------------------------------------------------
VECTOR_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "vel": ("uvel", "vvel", "wvel"),
}


# --------------------------------------------------
# main function: normalize
# --------------------------------------------------
def normalize(var_name: str) -> str:
    """Return the canonical variable name for *var_name*.

    Performs a case-insensitive lookup in ``VAR_ALIASES``.  If no alias
    is found the lower-cased *var_name* is returned as-is.
    """

    key = var_name.lower()
    return VAR_ALIASES.get(key, key)
