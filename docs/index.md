# cfd-io

I/O toolkit for structured CFD data: read, write, and convert between formats.

## Features

- **Unified API**: `read_file` / `write_file` dispatch by file extension
- **Format conversion**: `do_convert` reads one format, writes another
- **File inspection**: `get_info` returns structured metadata without loading data
- **CLI**: `cfd-io convert` and `cfd-io info` for quick terminal use

## Quick Start

```bash
pip install cfd-io
```

```python
from cfd_io import read_file, get_info

info = get_info("flow.h5")
grid, flow, attrs = read_file("flow.h5")
```

See the [User Guide](user-guide/index.md) to get started, or the
[API Reference](api/index.md) for full technical details.
