# Learning-Joint-Optical-Flow-and-Depth-Estimation-from-Stereo-Events-and-Frames

Self-contained package for training and evaluating a dual-task network on the [MVSEC dataset](https://daniilidis-group.github.io/mvsec/). The network takes stereo event streams and stereo images as input and jointly predicts depth and optical flow.

## Package Structure

All paths are resolved relative to the directory containing the scripts, so the folder can be moved as a whole.

```text
joint_flow_depth_estimation/
├── train.py                    # training script (multi-scale loss, two-stage schedule)
├── test.py                     # evaluation script
├── EventDepthFlowNet.py        # network definition
├── README.md
├── requirements.txt            # environment dependencies
├── datasets/
│   ├── __init__.py
│   └── MVSEC/
│       ├── __init__.py
│       ├── MVSECProvider.py
│       ├── MVSECSequence_flow.py
│       ├── MVSECSequence_disparity.py
│       ├── MVSECSequence_flow_with_image.py   # dataset backend used by train.py and test.py
│       ├── preLoader/
│       │   ├── __init__.py
│       │   ├── mvsec.py
│       │   └── calibration.py
│       └── utils/
│           ├── __init__.py
│           └── EventToVoxel.py
├── Data/                       # user-provided, see Data Preparation
└── checkpoints_joint_flow_depth/   # created automatically by train.py
```

## Setup

Install the environment with the provided dependency file:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch build is recommended. The code automatically falls back to CPU if CUDA is unavailable.

## Data Preparation

Place the MVSEC HDF5 files under `Data/data_hdf5/` inside this package:

```text
Data/data_hdf5/
├── indoor_flying1_data.hdf5
├── indoor_flying1_gt.hdf5
├── indoor_flying2_data.hdf5
├── indoor_flying2_gt.hdf5
├── indoor_flying3_data.hdf5
├── indoor_flying3_gt.hdf5
├── outdoor_day1_data.hdf5
├── outdoor_day1_gt.hdf5
├── indoor_flying_calib.zip
└── outdoor_day_calib.zip
```

- The `data_hdf5` directory level is part of the path resolution inside `datasets/MVSEC/MVSECSequence_flow_with_image.py` and must not be flattened.
- The two calibration zip files are required by the data loader.

## Training

```bash
python train.py
```

Training uses a two-stage schedule. Stage 1 trains the RGB encoder and output heads for 10 epochs; Stage 2 fine-tunes all parameters with early stopping.

## Evaluation

```bash
python test.py                                  # full evaluation, indoor_flying1/2/3
python test.py --max_batches 25                 # first 25 batches per sequence
python test.py --ckpt PATH --seqs indoor_flying1
```

- Default checkpoint: `checkpoints_joint_flow_depth/best_model_rgb.pth`.
- Per-sequence metrics, sample counts, elapsed time and an `AVERAGE` entry are written to `test_joint_flow_depth_metrics.json`.
- A formatted report with the same numbers is written to `test_joint_flow_depth_report.txt`.
