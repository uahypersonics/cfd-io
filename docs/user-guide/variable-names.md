# Variable Naming Convention

All readers normalize variable names to a **canonical form** on read.
Writers emit canonical names as-is.
This ensures consistent dictionary keys regardless of input format.

## Canonical Names

### Grid Coordinates

| Canonical | Description |
|-----------|-------------|
| `x`       | x-coordinate |
| `y`       | y-coordinate |
| `z`       | z-coordinate |

### Flow Variables

| Canonical | Description |
|-----------|-------------|
| `uvel`    | x-velocity |
| `vvel`    | y-velocity |
| `wvel`    | z-velocity |
| `pres`    | Pressure |
| `temp`    | Temperature |
| `dens`    | Density |
| `mach`    | Mach number |
| `xmom`    | x-momentum |
| `ymom`    | y-momentum |
| `zmom`    | z-momentum |
| `energy`  | Energy |

## Alias Mapping

Common shorthand and long-form names are mapped to the canonical form
automatically during read.  Matching is **case-insensitive**.

| Input name(s) | Canonical |
|---------------|-----------|
| `u`, `u-velocity` | `uvel` |
| `v`, `v-velocity` | `vvel` |
| `w`, `w-velocity` | `wvel` |
| `p`, `pressure` | `pres` |
| `t`, `temperature` | `temp` |
| `rho`, `density` | `dens` |
| `m` | `mach` |
| `x-grid` | `x` |
| `y-grid` | `y` |
| `z-grid` | `z` |

Variable names that don't match any alias pass through unchanged.

## Example

A Tecplot file with header `VARIABLES = "X" "Y" "Z" "RHO" "U" "V" "P"`
produces dictionaries with keys:

```python
grid_dict.keys()  # dict_keys(['x', 'y', 'z'])
flow_dict.keys()  # dict_keys(['dens', 'uvel', 'vvel', 'pres'])
```

!!! tip

    Writers always use the canonical names.  If you convert from
    Tecplot → HDF5, the HDF5 file will contain `dens`, `uvel`, etc.
    — not the original Tecplot names.
