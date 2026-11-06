"""Revolved test grid shared by the 3-D writer tests: a hemisphere nose with i = 0 on the axis."""

import numpy as np

from cfd_io.dataset import Dataset, StructuredGrid

# boundary names of the six sides
MARKERS_3D = {
    "imin": "axis",
    "imax": "outflow",
    "jmin": "wall",
    "jmax": "farfield",
    "kmin": "sweep_low",
    "kmax": "sweep_high",
}


def revolved_sector(ni: int, nj: int, nk: int, left_handed: bool = False) -> Dataset:
    """A hemisphere nose (radius 1 to 2) revolved through 10 degrees about the x-axis.

    Args:
        ni: Points from the axis upstream of the nose round to the shoulder.
        nj: Points outward, radius 1 to 2.
        nk: Points in the angle, 0 to 10 degrees.
        left_handed: Run k from 10 degrees down to 0, which makes (i, j, k) left-handed.

    Returns:
        Grid-only dataset with markers for all six sides; its face i = 0 lies on the axis.
    """

    # meridian plane: i from the axis upstream of the nose round to the shoulder, j outward
    phi = np.linspace(0.0, np.pi / 2, ni)
    radius = np.linspace(1.0, 2.0, nj)
    x_plane = -radius[None, :] * np.cos(phi[:, None])
    r_plane = radius[None, :] * np.sin(phi[:, None])
    r_plane[0, :] = 0.0

    # revolve about the x-axis (k reversed for a left-handed grid)
    theta = np.linspace(0.0, np.radians(10.0), nk)
    if left_handed:
        theta = theta[::-1]
    x = np.repeat(x_plane[:, :, None], nk, axis=2)
    y = r_plane[:, :, None] * np.cos(theta)[None, None, :]
    z = r_plane[:, :, None] * np.sin(theta)[None, None, :]

    grid = StructuredGrid(x, y, z)
    dataset = Dataset(grid=grid, attrs={"markers": {"structured": dict(MARKERS_3D)}})
    return dataset
