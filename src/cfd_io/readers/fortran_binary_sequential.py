"""Generic Fortran unformatted binary reader.

Fortran's sequential unformatted I/O wraps every WRITE statement in
a pair of 4-byte integer "record markers":

    [nbytes]  <payload of nbytes bytes>  [nbytes]

where ``nbytes`` is the byte-length of the payload.  This reader
walks through those records sequentially, unpacking each one into
raw bytes, integers, or floating-point arrays.

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

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# FortranBinaryReader main class
# --------------------------------------------------
class FortranBinaryReader:
    """Read files containing Fortran unformatted sequential records.

    Each record is bracketed by 4-byte integer length markers (header
    and trailer).  This is the default format produced by gfortran and
    most other Fortran compilers for ``FORM='UNFORMATTED', ACCESS='SEQUENTIAL'``.

    Args:
        fname: Path to the binary file.
        endianness: Byte order (``"<"`` little-endian, ``">"`` big-endian).
        int_dtype: NumPy dtype for integer records.
        real_dtype: NumPy dtype for real-valued records.

    Example:
        Read a double-precision, little-endian grid file:

        ```python
        with FortranBinaryReader("grid.x") as reader:
            dims = reader.read_ints(expected_count=3)
            coords = reader.read_array_real(shape=(dims[0], dims[1], dims[2]))
        ```

        Single-precision file with big-endian byte order:

        ```python
        with FortranBinaryReader(
            "grid.x",
            endianness=">",
            int_dtype=np.int32,
            real_dtype=np.float32,
        ) as reader:
            dims = reader.read_ints(expected_count=3)
        ```
    """

    # initialize reader with file path and data type settings
    def __init__(
        self,
        fname: str | os.PathLike[str],
        *,
        endianness: str = "<",
        int_dtype: np.dtype = np.int32,
        real_dtype: np.dtype = np.float64,
    ) -> None:

        # open the file and assign file handle
        self._fhandle = open(fname, "rb")
        # store the endianness for later use in struct unpacking
        self._endianness = endianness
        # store the integer and real data types for later use in numpy array construction
        self._int = np.dtype(int_dtype)
        self._real = np.dtype(real_dtype)


    # low-level record reader: read one record and return raw payload bytes
    def _read_record_bytes(self) -> bytes:
        """Read one Fortran record and return the raw payload bytes."""

        # read the leading 4-byte record length header
        head = self._fhandle.read(4)
        if len(head) == 0:
            raise EOFError("End of file while attempting to read record length header")
        if len(head) != 4:
            raise OSError("Truncated record header: expected 4 bytes")

        # unpack the record length from the header
        (n,) = struct.unpack(self._endianness + "i", head)
        if n < 0:
            raise OSError(f"Invalid record length: {n}")

        # read the actual data (payload) of the record
        payload = self._fhandle.read(n)
        if len(payload) != n:
            raise OSError("Truncated payload: fewer bytes than record length")

        # read and verify the trailing record length marker
        tail = self._fhandle.read(4)
        if len(tail) != 4:
            raise OSError("Truncated record trailer: expected 4 bytes")

        # unpack the trailing record length and verify it matches the header
        (n2,) = struct.unpack(self._endianness + "i", tail)
        if n2 != n:
            raise OSError(f"Record length mismatch: header={n}, trailer={n2}")

        # return the raw payload bytes for further processing by higher-level methods
        return payload


    # numpy-level record reader: interpret raw bytes as a typed 1-D array
    def _read_numpy_record(self, dtype: np.dtype) -> np.ndarray:
        """Read a record and interpret its bytes as a 1-D NumPy array."""

        # read the raw bytes of the record
        b = self._read_record_bytes()

        # sanity check: payload must be an exact multiple of the element size
        if len(b) % np.dtype(dtype).itemsize != 0:
            raise OSError("Record byte-length not divisible by dtype size")

        # frombuffer returns a read-only view
        arr = np.frombuffer(b, dtype=dtype)

        # return a copy to allow downstream mutation
        return arr.copy()


    # read a fixed-length string record, stripping trailing spaces and decoding
    def read_string_fixed(self, length: int = 64, encoding: str = "ascii") -> str:
        """Read a fixed-length string record, strip trailing spaces, decode.

        Args:
            length: Expected byte length of the string record.
            encoding: Character encoding.

        Returns:
            Decoded and right-stripped string.
        """

        # read the raw bytes of the record
        b = self._read_record_bytes()

        # Fortran writes character(len=N) as exactly N bytes
        if len(b) != length:
            raise OSError(
                f"Unexpected string record length: got {len(b)}, expected {length}"
            )

        # Fortran pads with trailing spaces; strip them before decoding
        return b.rstrip(b" ").decode(encoding, errors="strict")


    # read an integer record
    def read_ints(self, *, expected_count: int | None = None) -> np.ndarray:
        """Read an integer record.

        Args:
            expected_count: If given, verify the number of integers read.

        Returns:
            1-D integer array.
        """

        # read the record as a NumPy array of the configured integer dtype
        arr = self._read_numpy_record(self._int)

        # sanity check: verify the number of integers read if expected_count is given (optional)
        if expected_count is not None and arr.size != expected_count:
            raise OSError(
                f"Unexpected integer count: got {arr.size}, expected {expected_count}"
            )

        # return the array of integers
        return arr


    # read a real-valued record
    def read_reals(self, *, expected_count: int | None = None) -> np.ndarray:
        """Read a real-valued record.

        Args:
            expected_count: If given, verify the number of reals read.

        Returns:
            1-D real array.
        """

        # read the record as a NumPy array of the configured real dtype
        arr = self._read_numpy_record(self._real)

        # sanity check: verify the number of real values read if expected_count is given (optional)
        if expected_count is not None and arr.size != expected_count:
            raise OSError(
                f"Unexpected real count: got {arr.size}, expected {expected_count}"
            )

        # return the array of real values
        return arr

    # read a real-valued array record and reshape to the given shape
    def read_array_real(
        self, shape: tuple[int, ...], *, fortran_order: bool = True
    ) -> np.ndarray:
        """Read a real-valued array written as one record.

        Args:
            shape: Desired shape of the returned array.
            fortran_order: If True, interpret bytes as column-major (Fortran) order.

        Returns:
            Array reshaped to *shape*.
        """

        # read the record as a flat 1-D real array
        arr = self._read_numpy_record(self._real)

        # verify the total number of elements matches the requested shape
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise OSError(
                f"Unexpected array size: got {arr.size}, expected {expected} for shape {shape}"
            )

        # Fortran stores arrays in column-major ("F") order by default
        order = "F" if fortran_order else "C"
        return np.reshape(arr, shape, order=order)


    # skip the next n records without returning data
    def skip_records(self, n: int = 1) -> None:
        """Discard the next *n* Fortran records without returning data."""
        for _ in range(int(n)):
            self._read_record_bytes()


    # close the file when done
    def close(self) -> None:
        """Close the underlying file."""
        self._fhandle.close()


    # --------------------------------------------------
    # context-manager support
    # --------------------------------------------------
    # These two methods enable the ``with`` statement:
    #
    #     with FortranBinaryReader("grid.x") as reader:
    #         dims = reader.read_ints()
    #     # file is automatically closed here, even if an exception occurred
    #
    # __enter__ is called when entering the ``with`` block; returns self.
    # __exit__  is called when leaving the block; closes the file handle.
    def __enter__(self) -> FortranBinaryReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
