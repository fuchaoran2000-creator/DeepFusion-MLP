# Original raw-input ROS/SLAM reproduction

The recorded-input numerical paths do not require ROS. A complete new acquisition/replay uses Ubuntu 20.04, ROS Noetic, Cartographer ROS, `rosbag`, `rospy`, `tf2_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `cartographer_ros_msgs`, PyYAML, NumPy, SciPy, psutil and the project Torch stack. Use a ROS-compatible Python 3.8 environment; an unrelated virtualenv may not see the system ROS modules.

Exact upstream commits are in `third_party/versions.json`; local original source checkouts had no uncommitted source changes. Run:

```bash
python tools/fetch_external_sources.py
```

This fetches source only, using pinned commits and preserving upstream licenses. Build Cartographer/Cartographer ROS according to their upstream dependency/build instructions and the original ROS distribution. For the default checkout location, use an isolated install at `third_party/cartographer_ws/install_isolated`. Alternatively set:

```bash
export CARTOGRAPHER_INSTALL=/absolute/path/to/cartographer_ws/install_isolated
export CARTOGRAPHER_SOURCE=/absolute/path/to/cartographer_ws/src/cartographer
source /opt/ros/noetic/setup.bash
source "$CARTOGRAPHER_INSTALL/setup.bash"
```

2DLIW expects a catkin build of the pinned source at `third_party/2dliw_workspace/src/2DLIW-SLAM-main`; the runner uses `devel/lib/lvio_2d/lvio_2d_node` and upstream `config/params.yaml`. Build that workspace using its upstream instructions. No precompiled binaries, system packages or raw data are included.

## Prepare raw data

Download OpenLORIS bags into `data/raw/openloris/`, then:

```bash
python tools/prepare_rosbags.py data/raw/openloris/home1-1.bag data/raw/openloris/home1-2.bag \
  data/raw/openloris/home1-3.bag data/raw/openloris/home1-4.bag data/raw/openloris/home1-5.bag
python tools/prepare_nclt.py --root data/nclt --session 2013-01-10 --output data/prepared
```

The preparation tools extract reference poses separately. The SLAM input bag excludes `/gt`. Inspect `--help` and the source session conventions when preparing other dates. Odometry frame/units and sensor-to-base extrinsics must remain consistent.

## Capture causal inputs, train, replay

```bash
python tools/online_ros_benchmark.py data/prepared/home1-1 data/prepared/home1-2 \
  data/prepared/home1-3 data/prepared/home1-4 data/prepared/home1-5 \
  --algorithms cartographer_noimu --capture-stream --label unified_inputs --port 11446
python tools/verify_causal_stream.py --manifest config/benchmark_home_causal_v1.json \
  --output results/input_equivalence.json
python mentor_benchmark.py --manifest config/benchmark_home_causal_v1.json \
  --output results/mentor_benchmark/new_ros_training --configs full20 relative32 \
  --seeds 11 22 33 --no-extra-models --with-uncertainty
python tools/select_causal_deployment.py results/mentor_benchmark/new_ros_training
```

Read the resulting deployment selection before choosing a new checkpoint; do not automatically reuse seed 11 after retraining. For the frozen published C20:

```bash
python tools/online_ros_benchmark.py data/prepared/home1-3 data/prepared/home1-4 data/prepared/home1-5 \
  --algorithms cartographer_noimu --capture-stream \
  --checkpoint artifacts/checkpoints/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt \
  --label published_c20 --port 11447
```

The output is `/deepfusion_pose`. Source/reference scores must be computed on the same run and shared map anchor. Labels determine output filenames; fresh labels preserve previous executions.

## Auxiliary raw faults and SLAM configuration comparison

```bash
python tools/prepare_raw_fault_test.py --monotonic-clock
python tools/online_ros_benchmark.py data/prepared/home1-3_raw_fault_v2 \
  --algorithms cartographer_noimu --quality-stream \
  --checkpoint artifacts/checkpoints/home_trajectory_quality_v1/quality44_trajectory_seed22/model.pt \
  --label raw_fault_v2 --port 11460
python tools/online_ros_benchmark.py data/prepared/home1-3 data/prepared/home1-4 data/prepared/home1-5 \
  --algorithms cartographer_noimu cartographer_imu 2dliw --label configuration_compare --port 11461
```

Q44 requires `--quality-stream`; do not send a quality checkpoint through the old 20D/32D adapter. IMU/2DLIW receive additional measurements. Keep abnormal termination and output coverage in the results; playback completion alone does not prove final output completeness. Source SLAM diagnostics, final optimized trajectories, and current published trajectories are separate interfaces.

Rerunning source SLAM is sensitive to callback scheduling and software versions. The included recorded inputs are the reproducible evidence for the reported executions, not a promise of bit-identical fresh ROS timing or physical edge performance.
