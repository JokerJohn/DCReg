# Parking-Lot Example Data

This folder contains the small default source frame used by
`dcreg_parking_lot_example`:

- `parkinglot_raw_1976_frame.pcd`
- `parkinglot_raw_1976_info.txt`

The prior map is intentionally not committed because it is large. Download the
parking-lot prior map from the data link in the repository README and place it
here as:

```text
DCReg/dataset/Parking-Lot-example/prior_map.pcd
```

After that, run:

```bash
./DCReg/build/dcreg_parking_lot_example
python3 scripts/visualize_parking_lot_example.py
```

The C++ executable writes visualization artifacts to
`DCReg/dataset/Parking-Lot-example/visualization/`.
