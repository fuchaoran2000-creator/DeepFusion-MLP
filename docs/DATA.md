# Data and evaluation contracts

Commands run from the repository root. Original raw data are acquired separately from the authors:

- OpenLORIS: https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md
- NCLT: https://robots.engin.umich.edu/nclt/

The release contains only numeric experiment records, model parameters and derived references. No raw bag/image/laser datasets or private collection are included. Cite and respect the source dataset terms; the project software MIT license does not relicense their material.

## Three distinct reproduction levels

1. **Archived-score / fixed-inference verification:** `python tools/reproduce_paper.py`. The C20 model runs on three exact online captures; Table 7 is also recomputed from original score CSVs. This is independent of ROS/GPU.
2. **Retraining / new inference from recorded numeric inputs:** use `config/recorded/*.json` and `artifacts/quality_inputs`. The actual input-pair availability is preserved. The labels are parsed from losslessly compressed `.tum.gz` references. Fresh random seeds, Torch versions and numerical libraries may change learned parameters.
3. **Full raw-data source SLAM:** obtain bags/binaries, build the pinned SLAM tools, prepare `data/prepared`, then capture fresh ROS arrivals. Use the corresponding `config/*.json` manifests. New arrival scheduling and source SLAM can change the captured inputs; do not claim byte-identical scores from a fresh ROS run.

## Dataset split

Home1-1 trains, Home1-2 validates, and Home1-3/4/5 test the primary Home model. These are five sequences of one scene. Office1-1…7, Cafe1-1…2 and Corridor1-3…5 form the held-scene development evaluations. NCLT 2013-01-10 is a development/diagnostic session; NCLT 2012-04-29 and 2013-04-05 were subsequently evaluated under the frozen plan in `artifacts/studies/final_public_holdout_v1/plan.json`.

The fixed cross-scene training experiment uses Home+Office training, Cafe validation, Corridor/NCLT evaluation. Scene rotation instead chooses each test/validation scene pair and trains on the remaining two scenes: 12 folds × 3 objectives × 3 seeds = 108 models. Table 7 uses **one frozen Home C20**, not these retrained models.

## Causal contract

`causal_grid_v1` samples at 10 Hz anchored to the first Cartographer publication. It uses only received input poses with source times at or before the tick, age ≤0.15 s. Input origins stay fixed. Missing inputs clear history; the first recovery tick returns raw. References never control the inference clock or enter inference features.

Recorded manifests retain the original `cart`/`odom` identifiers for provenance. `load_causal_sequence` actually loads the `sampled_stream` and `gt` fields; those files are supplied. Historical quality descriptors use source-time causality because old scan callback arrival order was not recorded. Actual `causal_quality_v1` captures in `artifacts/quality_replay` record received scan callbacks and are distinct inputs.

## Scoring protocols

| Protocol | Input / pairing | Meaning |
|---|---|---|
| L | Final offline trajectory; chronological split within Home1-1 | Preliminary; 1 m RPE has sparse, unshared pairs |
| P | Fixed captured causal Home inputs | Shared-input method/feature comparison |
| U | Separate SLAM configurations on common valid output times | Extra-sensor configuration comparison |
| O | Source and corrected poses from the same online execution | Main paired Home C20 result |
| Frozen transfer | Same Home C20, fixed normalization/thresholds | Main Table 7 development transfer |
| Q44-F / Q44-N | Auxiliary 44D fault / frozen new-session studies | Separate model and input scope |

P/U/O use overlapping nearest 1 s relative-motion pairs, tolerance ±0.11 s, and exclude pairs with invalid interiors. Per-sequence translational RMSE is computed before an equal-sequence macro average. No complete-test trajectory fitting or scale adjustment is used. Initial coordinate alignment is shared by paired source/corrected outputs. Table 7 improvement is `100*(raw-corrected)/raw`, computed before rounding; negative means degradation.

Office references use OptiTrack. Home/Cafe/Corridor and NCLT references are SLAM-derived, so errors measure agreement with supplied references. Hardware, site and odometry changes in NCLT are joint changes, not controlled single-sensor factors. The two initial/adapted Table 7 NCLT rows are the same session; the baseline improvement after diagnostic front-end adjustment is not a C20 gain.

Model and captured-input hashes are recorded in `artifacts/source_provenance.json` and the frozen replay manifests. Do not rename a historical protocol as an independent new test.
