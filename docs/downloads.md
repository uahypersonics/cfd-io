# Downloads

Sample data files for trying out `cfd-io`.

| File | Format | Description |
|------|--------|-------------|
| <a href="assets/samples/sample_flow.h5" download>sample_flow.h5</a> | HDF5 | Flow solution with grid and variables |
| <a href="assets/samples/sample_flow.dat" download>sample_flow.dat</a> | Tecplot ASCII | Flow solution in Tecplot format |
| <a href="assets/samples/sample_grid.x" download>sample_grid.x</a> | Plot3D grid | Grid file in Plot3D format |

After downloading, try:

```bash
cfd-io info sample_flow.h5
cfd-io convert sample_flow.h5 -o output.dat
```

See the [User Guide](user-guide/index.md) for more examples.
