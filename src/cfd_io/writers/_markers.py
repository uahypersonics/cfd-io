"""Boundary markers shared by the mesh writers (SU2, Gmsh).

Structured grids carry their boundary names in
``dataset.attrs["markers"]["structured"]``, which maps logical side names
(``imin``, ``imax``, ``jmin``, ``jmax``, ``kmin``, ``kmax``) to the name of
that boundary in the output file.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

# --------------------------------------------------
# Canonical side ordering
# --------------------------------------------------
_SIDES_FOR_AXIS = {
    0: ("imin", "imax"),
    1: ("jmin", "jmax"),
    2: ("kmin", "kmax"),
}

# --------------------------------------------------
# helper function to validate markers and merge boundary elements by marker tag
# --------------------------------------------------
def _required_sides(active_axes: list[int]) -> list[str]:
    """Required marker side names in canonical order for *active_axes*."""
    sides: list[str] = []
    for ax in sorted(active_axes):
        lo, hi = _SIDES_FOR_AXIS[ax]
        sides.append(lo)
        sides.append(hi)
    return sides


# --------------------------------------------------
# helper function to validate that all required marker keys are present in the markers dict
# --------------------------------------------------
def _validate_markers(markers: dict, required: list[str]) -> None:
    """Check that *markers* contains every key in *required*.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [s for s in required if s not in markers]
    if missing:
        raise ValueError(
            f"missing required marker keys: {missing}; "
            f"required for this dimensionality: {required}"
        )


# --------------------------------------------------
# helper function to extract and validate markers from dataset.attrs
# --------------------------------------------------
def _extract_markers(attrs: dict, required: list[str]) -> dict[str, str]:
    """Extract ``attrs["markers"]["structured"]``.

    Raises:
        TypeError: ``markers`` is missing or malformed.
    """
    if "markers" not in attrs:
        example = {s: s for s in required}
        raise TypeError(
            "dataset.attrs is missing 'markers'; this writer requires "
            "boundary tag names. Provide them as e.g.\n"
            f"    dataset.attrs['markers'] = {{'structured': {example}}}\n"
            "where each value is the boundary name for that side (SU2 "
            "MARKER_TAG, Gmsh physical name)."
        )
    markers_top = attrs["markers"]
    if not isinstance(markers_top, dict):
        raise TypeError(
            f"attrs['markers'] must be a dict, "
            f"got {type(markers_top).__name__}"
        )
    if "structured" not in markers_top:
        raise TypeError(
            "attrs['markers'] must contain a 'structured' key "
            "for structured grids"
        )
    structured = markers_top["structured"]
    if not isinstance(structured, dict):
        raise TypeError(
            f"attrs['markers']['structured'] must be a dict, "
            f"got {type(structured).__name__}"
        )
    return structured
