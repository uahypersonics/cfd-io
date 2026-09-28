"""Tests for Table dataclass, CSV reader/writer, and Tecplot ASCII table writer."""

from pathlib import Path

import numpy as np
import pytest

from cfd_io.readers.csv import read_table_csv
from cfd_io.table import Table
from cfd_io.writers.csv import write_table_csv
from cfd_io.writers.tecplot_ascii_table import write_table_tecplot_ascii

# ======================================================================
# shared fixtures
# ======================================================================

@pytest.fixture()
def simple_table() -> Table:
    """A small 3-column, 5-row table for reuse across tests."""
    return Table(
        columns={
            "eta": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "fp":  np.array([0.0, 0.5, 0.8, 0.95, 1.0]),
            "tau": np.array([1.68, 1.60, 1.30, 1.08, 1.0]),
        },
        metadata={"title": "FSC test case"},
    )


# ======================================================================
# Table validation
# ======================================================================

class TestTableValidation:
    def test_basic_construction(self, simple_table: Table) -> None:
        # check properties match input
        assert simple_table.n_rows == 5
        assert simple_table.n_columns == 3
        assert simple_table.shape == (5, 3)
        assert simple_table.names == ["eta", "fp", "tau"]

    def test_empty_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            Table(columns={})

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            Table(columns={"": np.array([1.0, 2.0])})

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            Table(columns={"  ": np.array([1.0, 2.0])})

    def test_non_array_raises(self) -> None:
        with pytest.raises(TypeError, match="numpy ndarray"):
            Table(columns={"x": [1.0, 2.0, 3.0]})  # type: ignore[dict-item]

    def test_2d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            Table(columns={"x": np.ones((3, 2))})

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            Table(
                columns={
                    "a": np.array([1.0, 2.0, 3.0]),
                    "b": np.array([1.0, 2.0]),
                }
            )

    def test_single_row_table(self) -> None:
        # edge case: single row is valid
        t = Table(columns={"x": np.array([42.0])})
        assert t.n_rows == 1
        assert t.shape == (1, 1)


# ======================================================================
# CSV round-trip
# ======================================================================

class TestCsvRoundTrip:
    def test_write_then_read_comma(self, simple_table: Table, tmp_path: Path) -> None:
        # write CSV and read it back; columns should match to float64 precision
        fpath = tmp_path / "table.csv"
        write_table_csv(fpath, simple_table, delimiter=",")
        recovered = read_table_csv(fpath, delimiter=",")

        assert recovered.names == simple_table.names
        for name in simple_table.names:
            assert np.allclose(recovered.columns[name], simple_table.columns[name])

    def test_write_then_read_whitespace(self, simple_table: Table, tmp_path: Path) -> None:
        # space-delimited round-trip
        fpath = tmp_path / "table.dat"
        write_table_csv(fpath, simple_table, delimiter=" ", fmt="%.6f")
        recovered = read_table_csv(fpath)

        assert recovered.names == simple_table.names
        for name in simple_table.names:
            assert np.allclose(
                recovered.columns[name], simple_table.columns[name], atol=1e-6
            )

    def test_read_csv_with_comment_lines(self, tmp_path: Path) -> None:
        # comment lines before header and between data rows should be ignored
        content = "# source: digitized from Liu 2021 Fig. 2a\neta,fp\n0.0,0.0\n1.0,0.5\n"
        fpath = tmp_path / "with_comments.csv"
        fpath.write_text(content)
        t = read_table_csv(fpath)
        assert t.names == ["eta", "fp"]
        assert t.n_rows == 2
        assert np.isclose(t.columns["fp"][1], 0.5)

    def test_read_single_data_row(self, tmp_path: Path) -> None:
        # single data row should not raise
        fpath = tmp_path / "one_row.csv"
        fpath.write_text("x,y\n1.0,2.0\n")
        t = read_table_csv(fpath)
        assert t.n_rows == 1

    def test_read_mismatched_columns_raises(self, tmp_path: Path) -> None:
        # header has 2 names but data has 3 values per row
        fpath = tmp_path / "bad.csv"
        fpath.write_text("a,b\n1.0,2.0,3.0\n")
        with pytest.raises(ValueError, match="2 columns but data has 3"):
            read_table_csv(fpath)

    def test_read_duplicate_names_raises(self, tmp_path: Path) -> None:
        fpath = tmp_path / "dup.csv"
        fpath.write_text("x,x\n1.0,2.0\n")
        with pytest.raises(ValueError, match="duplicate column names"):
            read_table_csv(fpath)


# ======================================================================
# Tecplot ASCII table writer
# ======================================================================

class TestWriteTableTecplotAscii:
    def test_header_structure(self, simple_table: Table, tmp_path: Path) -> None:
        # check TITLE, VARIABLES, and ZONE lines are present and correct
        fpath = tmp_path / "out.dat"
        write_table_tecplot_ascii(fpath, simple_table, title="Test", zone_title="Z1")
        lines = fpath.read_text().splitlines()

        assert lines[0] == 'TITLE = "Test"'
        assert lines[1] == 'VARIABLES = "eta", "fp", "tau"'
        assert lines[2] == 'ZONE T="Z1", I=5, J=1, K=1, F=POINT'

    def test_data_row_count(self, simple_table: Table, tmp_path: Path) -> None:
        # number of data lines should equal n_rows
        fpath = tmp_path / "out.dat"
        write_table_tecplot_ascii(fpath, simple_table)
        lines = fpath.read_text().splitlines()
        data_lines = lines[3:]  # skip TITLE, VARIABLES, ZONE
        assert len(data_lines) == simple_table.n_rows

    def test_data_values_parseable(self, simple_table: Table, tmp_path: Path) -> None:
        # each data line should parse to the correct number of floats
        fpath = tmp_path / "out.dat"
        write_table_tecplot_ascii(fpath, simple_table)
        lines = fpath.read_text().splitlines()
        for line in lines[3:]:
            vals = [float(v) for v in line.split()]
            assert len(vals) == simple_table.n_columns

    def test_metadata_title_used_as_default(self, tmp_path: Path) -> None:
        # metadata["title"] should supply TITLE when keyword arg is not given
        t = Table(
            columns={"x": np.array([1.0, 2.0])},
            metadata={"title": "From metadata"},
        )
        fpath = tmp_path / "out.dat"
        write_table_tecplot_ascii(fpath, t)
        first_line = fpath.read_text().splitlines()[0]
        assert "From metadata" in first_line

    def test_keyword_title_overrides_metadata(self, tmp_path: Path) -> None:
        # explicit keyword argument must win over metadata
        t = Table(
            columns={"x": np.array([1.0])},
            metadata={"title": "Metadata title"},
        )
        fpath = tmp_path / "out.dat"
        write_table_tecplot_ascii(fpath, t, title="Explicit title")
        first_line = fpath.read_text().splitlines()[0]
        assert "Explicit title" in first_line

    def test_returns_path(self, simple_table: Table, tmp_path: Path) -> None:
        fpath = tmp_path / "out.dat"
        result = write_table_tecplot_ascii(fpath, simple_table)
        assert result == fpath
        assert result.exists()

    def test_public_api_import(self) -> None:
        # all three functions and Table must be importable from top-level cfd_io
        import cfd_io
        assert hasattr(cfd_io, "Table")
        assert hasattr(cfd_io, "read_table_csv")
        assert hasattr(cfd_io, "write_table_csv")
        assert hasattr(cfd_io, "write_table_tecplot_ascii")
