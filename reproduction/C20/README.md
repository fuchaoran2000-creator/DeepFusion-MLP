# Primary C20 reproduction archive

This archive accompanies the revised manuscript locally. It is not a claim of a public repository release.
From this directory, run `python replay_c20.py` with NumPy installed. No ROS, GPU, SciPy or PyTorch is required for this frozen capture replay. The original Torch checkpoint is supplied as an optional artifact.

The script verifies features, readiness, fallback flags, exact raw fallback, corrected poses, and Table 4 metrics on the three archived Home test captures. Scoring references are opened only after inference. References are dataset-derived, not independent motion-capture truth. This replay checks the already reported executions; it is not a new independent test or a full ROS/SLAM rerun.

`C20_config.json` gives every normalization constant, clipping bound, fixed gate threshold, alpha and original checkpoint hash. `model/portable.json` and `model/portable_weights.npz` drive the shared NumPy runtime. `captures/` holds the original captured input pairs and expected outputs. `scoring/` contains the aligned reference and valid mask, isolated from inference. `expected_table4.csv` indexes original paired results. `indices/` includes all 12 scene-rotation conditions, count definitions, per-sequence ablations and source-based fault percentages.

The package does not reproduce training, source SLAM, sensor calibration, new hardware experiments or acquisition timing. Full raw bags are not included. All model/threshold choices remain frozen.
