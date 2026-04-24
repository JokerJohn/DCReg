# Parking-Lot DCReg Visualization Notes

This optional visualization demonstrates the real-scene parking-lot example
without adding Open3D to the C++ core. The C++ executable exports point clouds
and diagnostics, while the Python script only handles visualization.

## Data Meaning

- `source_initial.pcd`: the single LiDAR frame transformed by the initial
  prediction from `parkinglot_raw_1976_info.txt`.
- `source_registered.pcd`: the same LiDAR frame transformed by the final DCReg
  registration result.
- `target_map.pcd`: a `100 m` local crop of the prior map around the final
  pose, downsampled with a `0.5 m` ground-plane grid for visualization. The
  registration path still uses the full target map loaded by the C++ example.
- `diagnostics.json`: DCReg observability diagnostics, including Schur
  condition numbers, aligned eigenvalues, degenerate mask, contribution ratios,
  and the preconditioned linear-solve report.

## Run

```bash
cmake -S DCReg -B DCReg/build
cmake --build DCReg/build -j8 --target dcreg_parking_lot_example
./DCReg/build/dcreg_parking_lot_example
python3 scripts/visualize_parking_lot_example.py
```

By default, the C++ example exports visualization artifacts to:

```text
DCReg/data/Parking-Lot-example/visualization
```

Use `--export_vis <dir>` on the C++ executable and pass the same directory to
the Python script if you want a temporary export location.

For a non-GUI check, validate the export and print diagnostics only:

```bash
python3 scripts/visualize_parking_lot_example.py --dry_run
```

Install the optional Python dependency with:

```bash
python3 -m pip install open3d numpy pillow
```

If Pillow is not available, the script still saves the raw Open3D screenshot.

## Visual Encoding

- Gray-scale intensity points: `100 m` local prior-map crop sampled at `0.5 m`.
- Orange intensity points: source frame at the initial prediction.
- Blue intensity points: source frame after DCReg registration.
- Amber axes: weakest observable physical axes.
- Red axes: degenerate axes, if the DCReg mask marks any axis as degenerate.

The terminal output complements the 3D viewer by printing the rotation and
translation contribution-ratio matrices. These matrices show how the Schur
eigenvectors project onto the physical roll/pitch/yaw and x/y/z axes.

The viewer defaults to a compact presentation mode: one small 3D card shows
the color legend, degenerate axis, mask, and RMSE. This keeps the point-cloud
alignment visible instead of covering it with dense diagnostics. Use
`--verbose_labels` if you want the detailed per-axis lambda and contribution
ratio labels, and `--show_settings --show_open3d_helpers` for Open3D debugging.

A PNG screenshot is saved automatically as:

```text
DCReg/data/Parking-Lot-example/visualization/dcreg_parking_lot_view.png
```

The screenshot uses Open3D's stable static rendering path with a black
background, a top-down camera, intensity coloring, and a compact 2D information
card, so it is suitable for README or wiki figures. Add `--plain_screenshot`
if you need the raw point-cloud image.
Open the interactive viewer for the compact presentation labels, or add
`--verbose_labels` for the full diagnostic labels.
