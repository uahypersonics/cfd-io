"""Canonical-orientation helpers for structured CFD datasets.

After a reader returns a `Dataset`, the convention used inside cfd-io is:

* ``j = 0``        -> body / no-slip wall
* ``j = nj - 1``   -> outer / freestream boundary
* ``i = 0``        -> upstream / inflow edge
* ``i = ni - 1``   -> downstream / outflow edge
* ``k`` axis       -> spanwise (left untouched)

Different solvers and grid generators use different index conventions, so
the helper in this module looks at the velocity field on the four (i, j)
boundary edges and picks the canonical orientation.  The wall is taken to
be the boundary edge with the smallest mean ``|v|`` (no-slip), and the
``+i`` direction is flipped if the tangential velocity on the first
off-wall row points the wrong way.

If no velocity field is available the dataset is returned unchanged.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging

import numpy as np

from cfd_io.dataset import Dataset, Field, StructuredGrid

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 2-D structured array to 3-D by adding a unit k axis."""
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return arr


def _gather_velocity(flow: dict[str, Field]) -> np.ndarray | None:
    """Return a stacked velocity array of shape ``(ni, nj, nk, 3)`` if
    available in *flow*, otherwise ``None``.

    Looks for canonical names ``uvel`` / ``vvel`` / ``wvel``; missing
    components are treated as zero so 2-D solvers without ``wvel`` still
    work.
    """
    # check for at least the in-plane components -> required for wall detection
    if "uvel" not in flow or "vvel" not in flow:
        return None

    # gather components, defaulting missing ones to zeros
    u = _ensure_3d(np.asarray(flow["uvel"].data, dtype=float))
    v = _ensure_3d(np.asarray(flow["vvel"].data, dtype=float))
    if "wvel" in flow:
        w = _ensure_3d(np.asarray(flow["wvel"].data, dtype=float))
    else:
        w = np.zeros_like(u)

    return np.stack([u, v, w], axis=-1)


def _edge_mean_speed(vmag: np.ndarray, edge: str) -> float:
    """Return the mean velocity magnitude on a boundary edge.

    *vmag* has shape ``(ni, nj, nk)``; *edge* is one of
    ``"j0"``, ``"jN"``, ``"i0"``, ``"iN"``.
    """
    if edge == "j0":
        return float(np.mean(vmag[:, 0, :]))
    if edge == "jN":
        return float(np.mean(vmag[:, -1, :]))
    if edge == "i0":
        return float(np.mean(vmag[0, :, :]))
    if edge == "iN":
        return float(np.mean(vmag[-1, :, :]))
    raise ValueError(f"unknown edge: {edge}")


def _apply_swap(arr: np.ndarray) -> np.ndarray:
    """Swap the first two axes of a structured array (i <-> j)."""
    if arr.ndim == 2:
        return arr.T
    # 3-D: swap axes 0 and 1, leave k as-is
    return np.swapaxes(arr, 0, 1)


def _flip_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """Flip *arr* along *axis* without copying when possible."""
    return np.flip(arr, axis=axis)


# --------------------------------------------------
# main canonicalization routine
# --------------------------------------------------
def canonicalize_dataset(ds: Dataset) -> Dataset:
    """Reorient *ds* in place-style to the canonical (j=0 wall) layout.

    The function inspects the four ``(i, j)`` boundary edges, finds the
    one with the smallest mean ``|v|`` (treated as the no-slip wall),
    and applies the unique transpose / flip combination needed to move
    that edge to ``j = 0``.  It then ensures the ``+i`` direction is
    streamwise by checking the tangential velocity along the first
    off-wall row and flipping ``i`` when needed.

    Datasets that do not contain ``uvel`` / ``vvel`` are returned
    unchanged because the wall cannot be detected without a velocity
    field.

    Returns
    -------
    Dataset
        A new `Dataset` whose grid and flow arrays are reoriented.  The
        original dataset is not modified.
    """
    # gather velocity components -> required for wall detection
    # canonicalization is only meaningful on a structured grid
    if not isinstance(ds.grid, StructuredGrid):
        logger.debug("canonicalize_dataset: non-structured grid, skipping")
        return ds

    vel = _gather_velocity(ds.flow)
    if vel is None:
        # debug output for devs
        logger.debug("canonicalize_dataset: no uvel/vvel found, skipping")
        return ds

    # build a 3-D view of the grid x,y arrays for shape work
    x3 = _ensure_3d(np.asarray(ds.grid.x))
    ni, nj, nk = x3.shape

    # speed magnitude at every node
    vmag = np.linalg.norm(vel, axis=-1)

    # score each boundary edge by mean |v| -> wall = smallest
    scores = {
        "j0": _edge_mean_speed(vmag, "j0"),
        "jN": _edge_mean_speed(vmag, "jN"),
        "i0": _edge_mean_speed(vmag, "i0"),
        "iN": _edge_mean_speed(vmag, "iN"),
    }
    wall = min(scores, key=scores.get)
    # debug output for devs
    logger.debug("canonicalize_dataset: edge mean |v| = %s -> wall=%s", scores, wall)

    # decide what transform moves the wall edge to j=0
    # transforms are described as (swap_ij, flip_i, flip_j) booleans
    if wall == "j0":
        swap, flip_i, flip_j = False, False, False
    elif wall == "jN":
        swap, flip_i, flip_j = False, False, True
    elif wall == "i0":
        # transpose: old i=0 becomes new j=0
        swap, flip_i, flip_j = True, False, False
    elif wall == "iN":
        # transpose then flip new j (== old i) to bring old i=ni-1 to j=0
        swap, flip_i, flip_j = True, False, True
    else:  # pragma: no cover -- unreachable
        return ds

    # apply transform to a single array
    def _transform(arr: np.ndarray) -> np.ndarray:
        out = arr
        if swap:
            out = _apply_swap(out)
        if flip_i:
            out = _flip_axis(out, axis=0)
        if flip_j:
            out = _flip_axis(out, axis=1)
        return np.ascontiguousarray(out)

    # transform the grid coordinates
    new_x = _transform(np.asarray(ds.grid.x))
    new_y = _transform(np.asarray(ds.grid.y))
    new_z = _transform(np.asarray(ds.grid.z))

    # transform every flow field
    new_flow: dict[str, Field] = {}
    for name, fld in ds.flow.items():
        new_flow[name] = Field(_transform(np.asarray(fld.data)))

    # build a temporary dataset to compute the i-direction streamwise check
    # using the *transformed* uvel / vvel
    new_u = _ensure_3d(np.asarray(new_flow["uvel"].data))
    new_v = _ensure_3d(np.asarray(new_flow["vvel"].data))
    new_x3 = _ensure_3d(new_x)
    new_y3 = _ensure_3d(new_y)

    # streamwise check: look at the first off-wall slab j=1
    # average tangent (di) and average velocity (u,v) along i, take dot product
    if new_x3.shape[1] >= 2 and new_x3.shape[0] >= 2:
        # tangent vector along i, averaged across k
        dx = np.diff(new_x3[:, 1, :], axis=0)
        dy = np.diff(new_y3[:, 1, :], axis=0)
        # average direction over k
        dx_mean = float(np.mean(dx))
        dy_mean = float(np.mean(dy))
        # average velocity over the first off-wall row, midpoint between i nodes
        u_avg = 0.5 * (new_u[:-1, 1, :] + new_u[1:, 1, :])
        v_avg = 0.5 * (new_v[:-1, 1, :] + new_v[1:, 1, :])
        u_mean = float(np.mean(u_avg))
        v_mean = float(np.mean(v_avg))
        # tangential component of velocity (sign only matters)
        u_tan = u_mean * dx_mean + v_mean * dy_mean
        # debug output for devs
        logger.debug(
            "canonicalize_dataset: i-streamwise check u_tan=%g (dx=%g dy=%g)",
            u_tan, dx_mean, dy_mean,
        )
        if u_tan < 0.0:
            # +i is currently upstream -> flip i so it points downstream
            logger.debug("canonicalize_dataset: flipping i to make +i streamwise")
            new_x = np.ascontiguousarray(_flip_axis(new_x, axis=0))
            new_y = np.ascontiguousarray(_flip_axis(new_y, axis=0))
            new_z = np.ascontiguousarray(_flip_axis(new_z, axis=0))
            new_flow = {
                name: Field(np.ascontiguousarray(_flip_axis(np.asarray(fld.data), axis=0)))
                for name, fld in new_flow.items()
            }

    # update attrs to reflect the new (possibly swapped) shape
    new_attrs = dict(ds.attrs)
    final_shape = _ensure_3d(new_x).shape
    new_attrs["ni"] = int(final_shape[0])
    new_attrs["nj"] = int(final_shape[1])
    new_attrs["nk"] = int(final_shape[2])

    # promote 2-D arrays to 3-D so downstream writers/consumers see a
    # uniform (ni, nj, nk) layout regardless of the source format
    new_x = _ensure_3d(new_x)
    new_y = _ensure_3d(new_y)
    new_z = _ensure_3d(new_z)
    new_flow = {
        name: Field(_ensure_3d(np.asarray(fld.data)))
        for name, fld in new_flow.items()
    }

    return Dataset(
        grid=StructuredGrid(new_x, new_y, new_z),
        flow=new_flow,
        attrs=new_attrs,
    )
