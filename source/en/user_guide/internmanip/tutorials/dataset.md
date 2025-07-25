# Dataset

This section introduces how to **prepare, organize, and convert datasets** into the unified [LeRobotDataset](https://github.com/huggingface/lerobot) format used by InternManip.
You’ll learn:

- 📁 [How to structure the dataset](#data-structure--format-specification)
- 🔁 [How to convert popular datasets like CALVIN and SimplerEnv](#convert-to-lerobotdataset)
- 🎮 [How to collect your own demonstrations in InternUtopia](#collect-demonstration-dataset-in-internutopia)


These steps ensure compatibility with our training and evaluation framework across all supported benchmarks.

## Data Structure & Format Specification

### Dataset Layout
InternManip adopts the [`LeRobotDataset`](https://github.com/huggingface/lerobot) format, which standardizes how videos, instructions, actions, and metadata are organized.
Each episode is stored in both `.parquet` format for structured access and `.mp4` for visualization, and chunked in groups of 1000 episodes for efficient loading.

Directory Overview:

```bash
<datasets_root>
│
├── <sub_dataset_x>  # Each sub-dataset (e.g., GenManip-V1)
│   ├── data         # Episodes in .parquet format, split into chunks (1,000 per folder)
│   │   ├── chunk-000
│   │   │   ├── episode_000000.parquet
│   │   │   └── ...
│   │   └── chunk-00n
│   │
│   ├── meta         # Metadata and statistics
│   │   ├── episodes.jsonl
│   │   ├── info.json
│   │   ├── tasks.jsonl
│   │   ├── modality.json
│   │   ├── episode_modality.jsonl (optional)
│   │   └── stats.json
│   │
│   └── videos       # Multi-view videos aligned with episodes
│       ├── chunk-000
│       │   ├── observation.images.head
│       │   └── ...
│       └── chunk-00n
│
└── ...              # More sub-datasets
```

### Metadata Files (Inside `meta/`)

The `meta/` folder contains critical metadata and statistics that power training, evaluation, and debugging.





**1. `episodes.jsonl`**

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
    "Sort personal care products | A robot stands in front of a table, on which there is a 3-grid material frame and a target frame, with the target frame on the left and the 3-grid material frame on the right."
  ],
  "length": 1240
}
```



**2. `info.json`**

This file contains metadata shared across the entire `task_n` split. It summarizes key information about the dataset, including device specifications, recording configuration, and feature schemas.

**Fields:**

- `codebase_version`: The version of the data format used (e.g., `"v2.1"`).
- `robot_type`: The robot hardware platform used for data collection (e.g., `"a2d"`).
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


Specifically, The `features` includes four categories:

1. **Video / Image Features**
   Each entry includes:
   - `dtype`: `"video"`
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
  "robot_type": "a2d",
  "total_episodes": 503,
  "total_frames": 564350,
  "fps": 30,
  "features": {
    "observation.images.head": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "rgb"],
      "info": {
        "video.fps": 30.0,
        "video.codec": "av1"
      }
    },
    "observation.states.joint.position": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "motors": ["left_arm_0", ..., "right_arm_6"]
      }
    },
    "actions.end.position": {
      "dtype": "float32",
      "shape": [2, 3],
      "names": {
        "motors": ["left_xyz", "right_xyz"]
      }
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    }
  }
}
```

**3. `tasks.jsonl`**

This file defines the unified high-level task associated with the current dataset.

**Fields:**

- `task_index`: The index of the task.
- `task`: A natural language description of the task scenario, including the environment setup and overall objective.

**Example:**

```json
{
  "task_index": 0,
  "task": "Sort personal care products | A robot stands in front of a table, on which there is a 3-grid material frame and a target frame, with the target frame on the left and the 3-grid material frame on the right."
}
```





**4. `modality.json`**


The `modality.json` file defines the mapping between keys in the raw dataset and the keys used by the model or training pipeline.
This mapping allows the training and evaluation code to work with consistent and unified feature names across different datasets, even if the raw data schemas differ.

The top-level keys of modality.json are fixed and must include:

- `state`: Describes all observed robot states, such as joint positions, gripper status, or end-effector pose.
- `action`: Describes control commands or target values (e.g., delta joint values, delta end-effector positions).
- `video`: Maps video streams or camera views.
- `annotation` (optional): Contains human annotations or labels, such as task descriptions or validity flags.

Each sub-key under state or action defines:

- `start` (int, optional): The start index of this feature in its concatenated vector.
- `end` (int, optional): The end index (exclusive).
- `original_key` (string, optional): The corresponding key in the raw dataset. If omitted, no mapping is performed.

These indices are useful when packing multiple features (e.g., joint positions and gripper state) into a single tensor.

**Example:**
```json
{
    "state": {
        "joints": {
            "start": 0,
            "end": 7
        },
    },
    "action": {
        "joints": {
            "start": 0,
            "end": 7
        },
    },
    "video": {
        "ego_view": {
            "original_key": "video.ego_view"
        },
    },
    "annotation": {
        "human.action.task_description": {},
    }
}
```



**5. `episodes_stats.jsonl` (Optional)**

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
    "observation.images.head": {
      "min": [[[x]], [[x]], [[x]]],
      "max": [[[x]], [[x]], [[x]]],
      "mean": [[[x]], [[x]], [[x]]],
      "std": [[[x]], [[x]], [[x]]],
      "count": [300]
    },
    "observation.states.joint.position": {
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

<!-- For details about each file under the meta/ directory, please refer to the [format specification](./format_specification.md). -->


**6. `stats.json`**

This file stores global statistical information about all features in a dataset.
These statistics are primarily used for:

- Normalization – e.g., standardizing features to zero mean and unit variance.

- Outlier Detection – using min/max or percentile values (q01, q99) to identify anomalies.

**Example:**
```json
{
    "actions.end.position": {
        "mean": [ ... ],
        "std": [ ... ],
        "min": [ ... ],
        "max": [ ... ],
        "q01": [ ... ],
        "q99": [ ... ]
    },
```


## Convert to LeRobotDataset

InternManip adopts the [LeRobot](https://github.com/huggingface/lerobot) format for all supported datasets. This section explains how to convert popular datasets — **CALVIN** and **SimplerEnv** — into this format using our provided [conversion scripts](#).

You’ll learn:

* How to convert **CALVIN** using RLDS → LeRobot pipeline
* How to quickly adapt **SimplerEnv** datasets
* How to adjust `.parquet` structure and metadata

---

### CALVIN → RLDS → LeRobot

> CALVIN (**C**omposing **A**ctions from **L**anguage and **Vi**sio**n**) is a benchmark for long-horizon language-conditioned tasks. [Download it here](https://github.com/mees/calvin).
To reduce possible IO congestion and other issues during conversion, it is necessary to first convert the data to RLDS format, and then convert RLDS to Lerobot format.

#### Step 1: Convert CALVIN to RLDS

> RLDS is an episodic data format for sequential decision-making. See [RLDS repo](https://github.com/google-research/rlds).
This step refers to this [code repository](https://github.com/mees/calvin_rlds_converter/tree/master).


First create a conda environment using the provided `environment_convert.yml` .

```bash
cd grmanipulation/scripts/data_convert
conda env create -f environment_convert.yml
conda activate trans_Lerobot

# Clone converter
git clone https://github.com/mees/calvin_rlds_converter.git
```

Follow conversion steps in that repo to produce RLDS-format data.

#### Step 2: RLDS → LeRobot

This step refers to this [code repository](https://github.com/Tavish9/any4lerobot/tree/main).

1. **Download** our source code:

   ```shell
   # clone our repo
   git clone https://gitlab.pjlab.org.cn/EmbodiedAI/SimPlatform/grmanipulation.git

   # In trans_Lerobot env
   cd /grmanipulation/scripts/data_convert/calvin2lerobot

   # clone lerobot
   git clone -b user/michel-aractingi/2025-05-20-hil-rebase-robots https://github.com/huggingface/lerobot.git

   cd lerobot

   pip install -e .
   ```

2. **Install** `ffmpeg` in your environment:

   ```shell
   conda install ffmpeg -c conda-forge

   #Additionally, you can also install ffmpeg using sudo：
   sudo apt update
   sudo apt install ffmpeg
   ```

3. **Modify** `lerobot/lerobot/common/datasets/video_utils.py`  in line 245：

   ```python
   from collections import OrderedDict
   import subprocess
   from pathlib import Path

   FFMPEG_BIN = "/bin/ffmpeg"  # modify this to your FFmpeg installation path

   def encode_video_frames(
       imgs_dir: Path | str,
       video_path: Path | str,
       fps: int,
       vcodec: str = "libx264",  # default: libx264
       pix_fmt: str = "yuv420p",
       g: int | None = 2,
       crf: int | None = 30,
       fast_decode: int = 0,
       log_level: str | None = "error",
       overwrite: bool = False,
   ) -> None:
       """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
       video_path = Path(video_path)
       imgs_dir = Path(imgs_dir)
       video_path.parent.mkdir(parents=True, exist_ok=True)

       ffmpeg_args = OrderedDict(
           [
               ("-f", "image2"),
               ("-r", str(fps)),
               ("-i", str(imgs_dir / "frame_%06d.png")),
               ("-vcodec", vcodec),
               ("-pix_fmt", pix_fmt),
           ]
       )

       if g is not None:
           ffmpeg_args["-g"] = str(g)

       if crf is not None:
           ffmpeg_args["-crf"] = str(crf)

       if fast_decode:
           key = "-tune"
           value = "fastdecode"
           ffmpeg_args[key] = value

       if log_level is not None:
           ffmpeg_args["-loglevel"] = str(log_level)

       ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
       if overwrite:
           ffmpeg_args.append("-y")

       # ffempeg path
       ffmpeg_cmd = [FFMPEG_BIN] + ffmpeg_args + [str(video_path)]

       subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

       if not video_path.exists():
           raise OSError(
               f"Video encoding did not work. File not found: {video_path}. "
               f"Try running the command manually to debug: `{' '.join(ffmpeg_cmd)}`"
           )
   ```

4. **Modify** path in `convert.sh`

   ```shell
   cd ..
   cd /rlds2lerobot/openx2lerobot

   # Modify paths
   python openx_rlds.py \
       --raw-dir /path/to/droid/1.0.0 \
       --local-dir /path/to/LEROBOT_DATASET \
       --repo-id your_hf_id \
       --use-videos
   ```

5. **Execute** the script:

   ```
   bash convert.sh
   ```

#### Step 3: Adapt for InternManip:

> After completing the above two steps, a **standard Calvin lerobot dataset can be obtained**. However, if the training framework we provide is to be used, modifications need to be made to the names of some keys and file names.

1. **Rename** the file names in the video folder:

   ```shell
   cd /../../calvin2lerobot/src
   python trans_lerobot_train_name.py
   ```

2. **Modify** the Data structure in the `.parquet` :

   ```shell
   python trans_lerobot_train_data.py
   ```

3. This script can help you **calculate** the Global dataset statistics (mean, std, min, max, quantiles) :

   ```shell
   python trans_lerobot_train_name.py
   ```

📝 Final `meta/` folder should include:

```
meta/
├── episodes.jsonl
├── info.json
├── tasks.jsonl
├── modality.json
└── stats.json
```

---

### SimplerEnv → LeRobot

#### Step 1: Download pre-converted datasets

> We recommend directly downloading the dataset that has been converted. In this way, only simple processing of the dataset is needed to adapt it to our training framework.

```bash
# bridge_orig_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot

# fractal20220817_data_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot
```

#### Step 2: Adapt to InternManip

1. Due to the high decoding overhead of AV1-encoded videos, which may cause issues like NCCL communication blocking, it is recommended to **convert them to H.264** using the provided script：

   ```shell
   # Configure FFmpeg (as described above)
   conda activate trans_lerobot
   cd /grmanipulation/scripts/data_convert/simplerEnv2lerobot/utils

   # For the dataset bridge_orig_lerobot, run:
   python trans_widowx_video.py
   # To adapt to our training framework, the video needs to be renamed. However, if you want to follow the default naming convention used by the LeRobot community, you can choose not to do so.
   python trans_widowx_video_name.py


   # For the dataset fractal20220817_data_lerobot, run:
   python trans_google_video.py
   # To adapt to our training framework, the video needs to be renamed. However, if you want to follow the default naming convention used by the LeRobot community, you can choose not to do so.
   python trans_google_video_name.py
   ```

2. **Modify** the Data structure in the `.parquet` :

   ```shell
   # For the dataset bridge_orig_lerobot, run:
   python trans_widowx_data.py

   # For the dataset fractal20220817_data_lerobot, run:
   python trans_google_data.py
   ```

📝 Final `meta/` folder usage:

* `stats.json`, `tasks.json`, `episodes.jsonl`: ✅ ready-to-use
* `info.json`, `modality.json`: use the provided ones in `./simplerEnv2lerobot/meta`


## Collect Demonstration Dataset in InternUtopia

Support for collecting demos via InternUtopia simulation is coming soon — stay tuned!
