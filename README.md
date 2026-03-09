# cfd-io

I/O toolkit for structured CFD data: read, write, and convert between formats.

[![Test](https://github.com/uahypersonics/cfd-io/actions/workflows/test.yml/badge.svg)](https://github.com/uahypersonics/cfd-io/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/uahypersonics/cfd-io/branch/main/graph/badge.svg)](https://codecov.io/gh/uahypersonics/cfd-io)
[![PyPI](https://img.shields.io/pypi/v/cfd-io)](https://pypi.org/project/cfd-io/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://uahypersonics.github.io/cfd-io/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install cfd-io
```

## Quick Start

```python
from cfd_io import read_file, write_file, get_info

# inspect a file
info = get_info("flow.h5")
print(info.nx, info.ny, info.var_names)

# read data
grid, flow, attrs = read_file("flow.h5")

# write to a different format
write_file("flow.h5", grid, flow, attrs)
```

## Features

- **Unified API**: `read_file` / `write_file` dispatch by file extension
- **Format conversion**: `do_convert` reads one format, writes another
- **File inspection**: `get_info` returns structured metadata without loading data
- **CLI**: `cfd-io convert` and `cfd-io info` for quick terminal use

## CLI

```bash
# convert between formats
cfd-io convert grid.x grid.h5

# inspect a file
cfd-io info grid.h5
```

## Testing

```bash
pytest tests/ --cov --cov-report=term-missing \
    --ignore=tests/test_tecplot_ascii.py \
    --ignore=tests/test_tecplot_binary.py \
    --ignore=tests/test_plt.py -q
```

## Documentation

Full documentation: https://uahypersonics.github.io/cfd-io

## Code Style

This project follows established Python community conventions so that
contributors can focus on the physics rather than inventing formatting rules.

| Convention | What it covers | Reference |
|---|---|---|
| [PEP 8](https://peps.python.org/pep-0008/) | Code formatting, naming, whitespace | Python standard style guide |
| [PEP 257](https://peps.python.org/pep-0257/) | Docstring structure (triple-quoted, imperative mood) | Python standard docstring conventions |
| [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) | Docstring sections (`Args`, `Returns`, `Raises`) | Google Python style guide |
| [Ruff](https://docs.astral.sh/ruff/) | Automated linting and formatting | Enforces PEP 8 compliance automatically |
| [typing / TYPE_CHECKING](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING) | Type hints for IDE support and static analysis | Python standard library |

## Versioning & Releasing

This project uses [Semantic Versioning](https://semver.org/) (`vMAJOR.MINOR.PATCH`):

- **MAJOR** (`v1.0.0`, `v2.0.0`): Breaking API changes
- **MINOR** (`v0.3.0`, `v0.4.0`): New features, backward-compatible
- **PATCH** (`v0.3.1`, `v0.3.2`): Bug fixes, minor corrections

To publish a new version to [PyPI](https://pypi.org/project/cfd-io/):

1. Update the version in `pyproject.toml`
2. Regenerate the API architecture diagram:
   ```bash
   pydeps src/cfd_io --noshow --max-bacon=4 --cluster -o docs/assets/architecture.svg
   ```
3. Commit and push to `main`
4. Tag and push:
   ```bash
   git tag vMAJOR.MINOR.PATCH
   git push origin vMAJOR.MINOR.PATCH
   ```

The GitHub Actions workflow will automatically build and publish to PyPI via Trusted Publishing.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
