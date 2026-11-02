"""Tests for Tecplot ASCII finite-element quad zones and single-point guards."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cfd_io.cli.app import app
from cfd_io.convert_mod import do_convert
from cfd_io.readers.tecplot_ascii import read_tecplot_ascii

runner = CliRunner()

# -- shared test data --------------------------------------------------
# a curved structured grid (a quarter annulus), longer in i than in j
NI, NJ = 7, 4
radius = np.linspace(1.0, 1.6, NJ)
angle = np.linspace(0.0, 0.5 * np.pi, NI)
GRID_X = np.outer(np.cos(angle), radius)
GRID_Y = np.outer(np.sin(angle), radius)
# a field that identifies each node
GRID_P = GRID_X + 10.0 * GRID_Y


def _structured_quads(ni: int, nj: int) -> np.ndarray:
    """Quad connectivity (0-based, i-fastest node numbering) of an ni x nj grid."""

    # one quad per cell, corners counter-clockwise
    quads = []
    for j in range(nj - 1):
        for i in range(ni - 1):
            n00 = i + j * ni
            quads.append([n00, n00 + 1, n00 + 1 + ni, n00 + ni])

    return np.array(quads)


def _matches_up_to_flips(result: np.ndarray, expected: np.ndarray) -> bool:
    """Whether result equals expected, allowing the i and j directions to be reversed."""

    # the reconstruction picks its own start corner
    candidates = [
        expected,
        expected[::-1, :],
        expected[:, ::-1],
        expected[::-1, ::-1],
    ]
    matches = [np.allclose(result, candidate) for candidate in candidates]

    return any(matches)


def _write_cfdpp_style(path: Path) -> None:
    """FEPOINT zone as CFD++ writes it (N=, E=, SOLUTIONTIME=, F=FEPOINT, ET=QUADRILATERAL)."""

    # nodes in file order, i-fastest
    x = GRID_X.ravel(order="F")
    y = GRID_Y.ravel(order="F")
    p = GRID_P.ravel(order="F")
    quads = _structured_quads(NI, NJ) + 1

    # write header, one row per node, then the connectivity
    lines = [
        'TITLE = " results at NT=90000, TAU=0.00000"',
        'VARIABLES = "X",',
        ' "Y",',
        ' "P"',
        f'ZONE T="Zone_1", N={x.size}, E={len(quads)}, SOLUTIONTIME= 0.0000000e+00, STRANDID=1,',
        "F=FEPOINT, ET=QUADRILATERAL",
    ]
    lines += [f"{a:.12e} {b:.12e} {c:.12e}" for a, b, c in zip(x, y, p)]
    lines += [" ".join(str(n) for n in quad) for quad in quads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_su2_style_shuffled(path: Path) -> None:
    """FEQuadrilateral BLOCK zone (SU2 style) with the nodes stored in random order."""

    # shuffle the node numbering, so only the connectivity carries the structure
    rng = np.random.default_rng(7)
    n_nodes = NI * NJ
    order = rng.permutation(n_nodes)
    new_index = np.empty(n_nodes, dtype=int)
    new_index[order] = np.arange(n_nodes)
    x = GRID_X.ravel(order="F")[order]
    y = GRID_Y.ravel(order="F")[order]
    p = GRID_P.ravel(order="F")[order]
    quads = new_index[_structured_quads(NI, NJ)] + 1

    # write header, one variable block after another, then the connectivity
    lines = [
        'TITLE = ""',
        'VARIABLES = "x"',
        '"y"',
        '"p"',
        'ZONE T="sym1"',
        " STRANDID=0, SOLUTIONTIME=0.5",
        f" Nodes={n_nodes}, Elements={len(quads)}, ZONETYPE=FEQuadrilateral",
        " DATAPACKING=BLOCK",
    ]
    for block in (x, y, p):
        lines += [f"{value:.12e}" for value in block]
    lines += [" ".join(str(n) for n in quad) for quad in quads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ======================================================================
# finite-element quad zones
# ======================================================================


@pytest.mark.parametrize("writer", [_write_cfdpp_style, _write_su2_style_shuffled])
def test_fe_quad_zone_reads_as_structured_grid(tmp_path: Path, writer) -> None:
    """FEPOINT and BLOCK quad zones come back as the structured (ni, nj, 1) grid."""

    # write and read
    path = tmp_path / "fe.dat"
    writer(path)
    ds = read_tecplot_ascii(path)

    # check: the full grid, longer direction as i
    assert ds.grid.shape == (NI, NJ, 1)
    x = ds.grid.x[:, :, 0]
    y = ds.grid.y[:, :, 0]
    assert _matches_up_to_flips(x, GRID_X)
    assert _matches_up_to_flips(y, GRID_Y)

    # check: every node keeps its own field value
    pres = ds.flow["pres"].data[:, :, 0]
    np.testing.assert_allclose(pres, x + 10.0 * y)


# ======================================================================
# headers the reader can't size
# ======================================================================


@pytest.mark.parametrize(
    ("zone_line", "message"),
    [
        ('ZONE T="no sizes", F=POINT', "no point count"),
        ("ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE", "unsupported Tecplot zone type"),
        (
            "ZONE Nodes=4, Elements=1, ZONETYPE=FEQuadrilateral, DATAPACKING=BLOCK, "
            "VARLOCATION=([3]=CELLCENTERED)",
            "cell-centered",
        ),
    ],
)
def test_unreadable_zones_raise(tmp_path: Path, zone_line: str, message: str) -> None:
    """Zones without sizes, non-quad elements and cell-centered data fail loudly."""

    # write a small file with the given zone header
    path = tmp_path / "bad.dat"
    path.write_text(
        "\n".join(['VARIABLES = "x", "y", "p"', zone_line, "0.0 0.0 1.0", "1.0 0.0 2.0"]) + "\n",
        encoding="utf-8",
    )

    # check
    with pytest.raises(ValueError, match=message):
        read_tecplot_ascii(path)


# ======================================================================
# convert refuses a single grid point
# ======================================================================


def test_convert_rejects_a_single_grid_point(tmp_path: Path) -> None:
    """A grid read as one point is an error, and the CLI says so in one line."""

    # an ordered zone of one point with coordinates
    path = tmp_path / "one_point.dat"
    path.write_text(
        'VARIABLES = "x", "y", "p"\nZONE I=1, J=1, F=POINT\n0.0 0.0 1.0\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.h5"

    # check: the function raises and writes nothing
    with pytest.raises(ValueError, match="single grid point"):
        do_convert(path, out)
    assert not out.exists()

    # check: the CLI reports it without a traceback
    result = runner.invoke(app, ["convert", str(path), "-o", str(out)])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "single grid point" in result.output
