# Dataset

This section introduces how to **prepare, organize, and convert datasets** into the unified [LeRobotDataset](https://github.com/huggingface/lerobot) format used by InternNav.
You’ll learn:

- 📁 [How to structure the dataset](#dataset-format)
- 🔁 [How to convert popular datasets like VLN-CE](#convert-to-lerobotdataset)
- 🎮 [How to collect your own demonstrations in InternUtopia](#collect-demonstration-dataset-in-internutopia)


These steps ensure compatibility with our training and evaluation framework across all supported benchmarks.



## Dataset Structure & Format Specification
InternNav adopts the [`LeRobotDataset`](https://github.com/huggingface/lerobot) format, which standardizes how videos, instructions, actions, and metadata are organized.
Each episode is stored in both `.parquet` format for structured access and `.mp4` for visualization.

The general directory structure looks like:

```bash
<datasets_root>
│
├── <sub_dataset_1> # Dataset for Env 1 (e.g., 3dfront_zed)
│   │
│   ├── <scene_dataset_1> # Dataset for Scene 1
│   │    │
│   │    ├── <traj_dataset_1> # Dataset for Trajectory 1
│   │    │   ├── data # Structured episode data in .parquet format
│   │    │   │   └── chunk-000
│   │    │   │       └──  episode_000000.parquet
│   │    │   │
│   │    │   ├── meta # Metadata and statistical information
│   │    │   │   ├── episodes_stats.jsonl # Per-episode stats
│   │    │   │   ├── episodes.jsonl # Per-episode metadata (subtask, instruction, etc.)
│   │    │   │   ├── info.json # Dataset-level information
│   │    │   │   └── tasks.jsonl # Task definitions
│   │    │   │
│   │    │   └── videos # Observation videos for each episode
│   │    │       │
│   │    │       └── chunk-000 # Videos for episodes 000000 - 000999
│   │    │            ├── observation.images.depth # Depth images for each trajectory
│   │    │            │   ├── 0.png # Depth image for each frame
│   │    │            │   ├── 1.png
│   │    │            │   └── ...
│   │    │            ├── observation.images.rgb # RGB images for each trajectory
│   │    │            │   ├── 0.jpg # RGB image for each frame
│   │    │            │   ├── 1.jpg
│   │    │            │   └── ...
│   │    │            ├── observation.video.depth # Depth video for each trajectory
│   │    │            │   └── episode_000000.mp4
│   │    │            └── observation.video.trajectory # RGB video for each trajectory
│   │    │                └── episode_000000.mp4
│   │    │
│   │    ├── <traj_dataset_2>
│   │    │
│   │    └── ...
│   │
│   ├── <scene_dataset_2>
│   │
│   └── ...
│
│
├── <sub_dataset_2>
│
└── ...

```


### Metadata Files (Inside `meta/`)

The `meta/` folder contains critical metadata and statistics that power training, evaluation, and debugging.



**1. episodes_stats.jsonl**

This file stores per-episode, per-feature statistical metadata for a specific dataset. Each line in the JSONL file corresponds to a single episode and contains the following:

- `episode_index`: The index of the episode within the dataset.
- `stats`: A nested dictionary mapping each feature (e.g., RGB images, joint positions, robot states, actions) to its statistical summary:
  - `min`: Minimum value for the feature across all timesteps in the episode.
  - `max`: Maximum value.
  - `mean`: Mean value.
  - `std`: Standard deviation.
  - `count`: Number of frames in the episode that the feature was observed.

> ⚠️ **Note**: The dimensions of `min`, `max`, `mean`, and `std` match the dimensionality of the original feature, while `count` is a scalar.

**Example entry:**

```json
{
  "episode_index": 0,
  "stats": {
    "observation.images.rgb": {
      "min": [[[x]], [[x]], [[x]]],
      "max": [[[x]], [[x]], [[x]]],
      "mean": [[[x]], [[x]], [[x]]],
      "std": [[[x]], [[x]], [[x]]],
      "count": [300]
    },
    "observation.images.depth": {
      "min": [x, x, ..., x],
      "max": [x, x, ..., x],
      "mean": [x, x, ..., x],
      "std": [x, x, ..., x],
      "count": [300]
    },
    ...
  }
}
```

---

**2. episodes.jsonl**

This file stores per-episode metadata for a specific task split (e.g., `task_0`, `task_1`, etc.). Each line represents one episode and includes basic information used by the training framework.

**Fields:**

- `episode_index`: A unique identifier for the episode.
- `tasks`: A list of high-level task descriptions. Each description defines the goal of the episode and should match the corresponding entry in `tasks.jsonl`.
- `length`: The total number of frames in this episode.

This file serves as the primary index of available episodes and their corresponding task goals.

**Example:**

```json
{
  "episode_index": 0,
  "tasks": [
    "Go straight down the hall and up the stairs. When you reach the door to the gym, go left into the gym and stop... "
  ],
  "length": 57
}
```


---

**3. info.json**

This file contains metadata shared across the entire `task_n` split. It summarizes key information about the dataset, including device specifications, recording configuration, and feature schemas.

**Fields:**

- `codebase_version`: The version of the data format used (e.g., `"v2.1"`).
- `robot_type`: The robot hardware platform used for data collection (e.g., `"a2d"`,`"unknown"`).
- `total_episodes`: Total number of episodes available in this task split.
- `total_frames`: Total number of frames across all episodes.
- `total_tasks`: Number of distinct high-level tasks (usually 1 for each task_n).
- `total_videos`: Total number of video files in this task split.
- `total_chunks`: Number of data chunks (each chunk typically contains ≤1000 episodes).
- `chunks_size`: The size of each chunk (usually 1000).
- `fps`: Frames per second used for both video and robot state collection.
- `splits`: Dataset split definitions (e.g., `"train": "0:503"`).
- `data_path`: Pattern for loading parquet files, where `episode_chunk` and `episode_index` are formatted.
- `video_path`: Pattern for loading video files corresponding to each camera.
- `features`: The structure and semantics of all recorded data streams.



Specifically, the `features` field specifies the structure and semantics of all recorded data streams. It includes four categories:

1. **Video / Image Features**
   Each entry includes:
   - `dtype`: `"video"`/ `"image"`
   - `shape`: Spatial resolution and channels
   - `names`: Axis names (e.g., `["height", "width", "rgb"]`)
   - `info`: Video-specific metadata such as codec, resolution, format, and whether it contains depth or audio.

2. **Observation Features**
   These include robot sensor readings (e.g., joint positions, effector poses) and follow this schema:
   - `dtype`: Data type (e.g., `"float32"`)
   - `shape`: Tensor shape
   - `names`: Dictionary describing physical meaning, usually under `"motors"`

3. **Action Features**
   These specify target actuator values or commands and have the same format as observation features.

4. **Other Features**
   These include auxiliary info such as timestamps, frame/episode indices, etc. Some of them (e.g., `timestamp`) are currently unused but included for completeness.

**Example Snippet (simplified)：**

```json
{
  "codebase_version": "v2.1",
  "robot_type": "unknown",
  "total_episodes": 1,
  "total_frames": 152,
  "total_tasks": 1,
  "total_videos": 2,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "features": {
    "observation.images.rgb": {
            "dtype": "image",
            "shape": [
                270,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
    "observation.images.depth": {
            "dtype": "image",
            "shape": [
                270,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
    "observation.video.trajectory": {
            "dtype": "video",
            "shape": [
                270,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 272,
                "video.width": 480,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.video.depth": {
            "dtype": "video",
            "shape": [
                270,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 272,
                "video.width": 480,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.camera_intrinsic": {
            "dtype": "float32",
            "shape": [
                3,
                3
            ],
            "names": [
                "intrinsic_0_0",
                "intrinsic_0_1",
                "intrinsic_0_2",
              ...
            ]
        },
        "observation.camera_extrinsic": {
            "dtype": "float32",
            "shape": [
                4,
                4
            ],
            "names": [
                "extrinsic_0_0",
                "extrinsic_0_1",
                "extrinsic_0_2",
             ...
            ]
        },
        "observation.path_points": {
            "dtype": "float64",
            "shape": [
                36555,
                3
            ],
            "names": [
                "x",
                "y",
                "z"
            ]
        },
        "observation.path_colors": {
            "dtype": "float32",
            "shape": [
                36555,
                3
            ],
            "names": [
                "r",
                "g",
                "b"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [
                4,
                4
            ],
            "names": [
                "action_0_0",
                "action_0_1",
                "action_0_2",
            ...
            ]
        },
        ...
}
```

---

**4. tasks.jsonl**

This file defines the unified high-level task associated with the current dataset.

**Fields:**

- `task_index`: The index of the task.
- `task`: A natural language description of the task scenario, including the environment setup and overall objective.

**Example:**

```json
{
  "task_index": 0,
  "task": "Go straight to the hallway and then turn left.  Go past the bed.  Veer to the right and go through the white door.  Stop when you're in the doorway."
}
```





## Convert to LeRobotDataset

InternNav adopts the [LeRobot](https://github.com/huggingface/lerobot) format for all supported datasets. This section explains how to convert popular datasets — **VLN-CE** — into this format using our provided [conversion scripts](#).

<!-- You’ll learn:

* How to convert **VLN-CE**
* How to quickly adapt **SimplerEnv** datasets
* How to adjust `.parquet` structure and metadata

--- -->

### VLN-CE → LeRobot

> VLN-CE (**V**ision and **L**anguage **N**avigation in **C**ontinuous **E**nvironments) is a benchmark for instruction-guided navigation task with crowdsourced instructions, realistic environments, and unconstrained agent navigation. [Download it here](https://github.com/jacobkrantz/VLN-CE).

1. **Download** our source code:

   ```shell
   # clone our repo
   git clone https://github.com/InternRobotics/InternNav.git

   # In trans_Lerobot env
   cd /scripts/dataset_converters/vlnce2lerobot

   # clone lerobot
   git clone -b user/michel-aractingi/2025-05-20-hil-rebase-robots https://github.com/huggingface/lerobot.git

   cd lerobot

   # Create a virtual environment with Python 3.10 and activate it
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   ```

2. **Install** `ffmpeg` in your environment:

   ```shell
   conda install ffmpeg -c conda-forge

   #Additionally, you can also install ffmpeg using sudo：
   sudo apt update
   sudo apt install ffmpeg
   ```


3. **Adapt** for InternNav and **Execute** the script:

   > To better accommodate the structure of navigation task datasets:

   > 1. Inherit from and extend the LeRobotDataset class by creating a new subclass called NavDataset.

   > 2. Inherit from and extend the LeRobotDatasetMetadata class by creating a new subclass called NavDatasetMetadata.

      ```shell
      python vlnce2lerobot.py \
         --data_dir /your/path/vln \               # Root folder
         --datasets RxR \                          # Which dataset split to process (RxR, R2R, …)
         --start_index 0 \
         --end_index 2000 \
         --repo_name vln_ce_lerobot \
         --num_threads 10
      ```



## Collect Demonstration Dataset in InternUtopia

Support for collecting demos via InternUtopia simulation is coming soon — stay tuned!
