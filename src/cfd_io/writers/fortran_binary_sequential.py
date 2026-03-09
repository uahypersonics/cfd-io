"""Generic Fortran unformatted binary writer.

Fortran's sequential unformatted I/O wraps every WRITE statement in
a pair of 4-byte integer "record markers":

    [nbytes]  <payload of nbytes bytes>  [nbytes]

where ``nbytes`` is the byte-length of the payload.  This writer
produces files that can be read directly by Fortran programs using
``FORM='UNFORMATTED', ACCESS='SEQUENTIAL'``.

Endianness defaults to little-endian (x86 / ARM / modern Linux and
macOS).  Pass ``endianness=">"`` for legacy big-endian files.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import os
import struct
from collections.abc import Iterable

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# FortranBinaryWriter main class
# --------------------------------------------------
class FortranBinaryWriter:
    """Write files containing Fortran unformatted sequential records.

    Args:
        fname: Path to the output file.
        endianness: Byte order (``"<"`` little-endian, ``">"`` big-endian).
        int_dtype: NumPy dtype for integer records.
        real_dtype: NumPy dtype for real-valued records.
    """

    # initialize writer with file path and data type settings
    def __init__(
        self,
        fname: str | os.PathLike[str],
        *,
        endianness: str = "<",
        int_dtype: np.dtype = np.int32,
        real_dtype: np.dtype = np.float64,
    ) -> None:
        self._fhandle = open(fname, "wb")
        self._endianness = endianness
        self._int = np.dtype(int_dtype)
        self._real = np.dtype(real_dtype)

    # -- low-level record writers -------------------------------------------

    # write one Fortran record: marker + payload + marker
    def _write_record_bytes(self, payload: bytes) -> None:
        """Write one Fortran record with 4-byte length markers around payload."""
        n = len(payload)

        # pack the byte-length as a 4-byte integer in the target endianness
        marker = struct.pack(self._endianness + "i", n)

        # write: [header marker] [payload] [trailer marker]
        self._fhandle.write(marker)
        self._fhandle.write(payload)
        self._fhandle.write(marker)

    # numpy-level record writer: serialize array bytes into a Fortran record
    def _write_numpy_record(self, arr: np.ndarray, *, order: str = "C") -> None:
        """Write a NumPy array as a single Fortran record."""
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        # serialize the array to raw bytes and wrap in record markers
        payload = arr.tobytes(order=order)
        self._write_record_bytes(payload)

    # -- typed record helpers -----------------------------------------------
    #
    # These build on _write_numpy_record() and _write_record_bytes() above
    # to expose the common Fortran data types: strings, integer vectors,
    # real-valued vectors, and shaped real arrays.
    # -------------------------------------------------------------------

    def write_string_fixed(
        self, s: str, length: int = 64, encoding: str = "ascii"
    ) -> None:
        """Write a fixed-length string as one record, padded with spaces.

        Args:
            s: String to write.
            length: Record length in bytes (padded or truncated).
            encoding: Character encoding.
        """
        # encode, then truncate or right-pad with spaces to exact length
        b = s.encode(encoding)[:length]
        if len(b) < length:
            b = b + b" " * (length - len(b))

        self._write_record_bytes(b)

    def write_ints(self, values: Iterable[int]) -> None:
        """Write an integer record.

        Args:
            values: Integers to write as a single record.
        """
        arr = np.asarray(list(values), dtype=self._int)
        self._write_numpy_record(arr)

    def write_reals(self, values: Iterable[float]) -> None:
        """Write a real-valued record.

        Args:
            values: Floats to write as a single record.
        """
        arr = np.asarray(list(values), dtype=self._real)
        self._write_numpy_record(arr)

    def write_array_real(self, arr: np.ndarray, *, fortran_order: bool = True) -> None:
        """Write a real-valued array as one record.

        Args:
            arr: Array to write.
            fortran_order: If True, write in column-major (Fortran) order.
        """
        a = np.asarray(arr, dtype=self._real)

        # Fortran stores arrays in column-major ("F") order by default
        order = "F" if fortran_order else "C"
        self._write_numpy_record(a, order=order)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying file."""
        self._fhandle.close()

    # -- context-manager support ------------------------------------------
    # These two methods enable the ``with`` statement:
    #
    #     with FortranBinaryWriter("grid.x") as writer:
    #         writer.write_ints([ni, nj, nk])
    #     # file is automatically closed here, even if an exception occurred
    #
    # __enter__ is called when entering the ``with`` block; returns self.
    # __exit__  is called when leaving the block; closes the file handle.

    def __enter__(self) -> FortranBinaryWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
