<style>
  .mytable {
    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 8px;        /* 四个角统一 8px 圆角 */
    border-collapse: separate; /* 必须设 separate，否则圆角会被 collapse 吃掉 */
    border-spacing: 0;
  }
  .mytable th, .mytable td {
    border: 1px solid rgba(128, 128, 128, 0.5);
    padding: 6px 12px;
  }
</style>

# 🛠️ Installation Guide

😄 Don’t worry — both [Quick Installation](#quick-installation) and [Dataset Preparation](#dataset-preparation) are beginner-friendly.


<!-- > 💡NOTE \
> 🙋 **[First-time users:](#-lightweight-installation-recommended-for-beginners)** Skip GenManip for now — it requires installing NVIDIA [⚙️ Isaac Sim](#), which can be complex.
Start with **CALVIN** or **SimplerEnv** for easy setup and full training/eval support.\
> 🧠 **[Advanced users:](#-full-installation-advanced-users)** Feel free to use all benchmarks, including **GenManip** with Isaac Sim support. -->

<!-- > For 🙋**first-time** users, we recommend skipping the GenManip benchmark, as it requires installing NVIDIA [⚙️ Isaac Sim](#) for simulation (which can be complex).
Instead, start with **CALVIN** or **SimplerEnv** — both are easy to set up and fully support training and evaluation. -->

<!-- This guide provides comprehensive instructions for installing and setting up the InternManip robot manipulation learning suite. Please read through the following prerequisites carefully before proceeding with the installation. -->

## Prerequisites

InternNav works across most hardware setups.
Just note the following exceptions:
- **Benchmark based on Isaac Sim** such as VN and VLN-PE benchmarks must run on **NVIDIA RTX series GPUs** (e.g., RTX 4090).

### Simulation Requirements
- **OS:** Ubuntu 20.04/22.04
- **GPU Compatibility**:
<table align="center" class="mytable">
  <tbody>
    <tr align="center" valign="middle">
      <td rowspan="2">
         <b>GPU</b>
      </td>
      <td rowspan="2">
         <b>Model Training & Inference</b>
      </td>
      <td colspan="3">
         <b>Simulation</b>
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         VLN-CE
      </td>
       <td>
         VN
      </td>
       <td>
         VLN-PE
      </td>

   </tr>
   <tr align="center" valign="middle">
      <td>
         NVIDIA RTX Series <br> (Driver: 535.216.01+ )
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         NVIDIA V/A/H100
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ❌
      </td>
      <td>
         ❌
      </td>
   </tr>
  </tbody>
</table>

```{note}
We provide a flexible installation tool for users who want to use InternNav for different purposes. Users can choose to install the training and inference environment, and the individual simulation environment independently.
```

<!-- Before installing InternManip, ensure your system meets the following requirements based on the specific models and benchmarks you plan to use. -->

### Model-Specific Requirements

<table align="center" class="mytable">
  <tbody>
    <tr align="center" valign="middle">
      <td rowspan="2">
         <b>Models</b>
      </td>
      <td colspan="2">
         <b>Minimum GPU Requirement</b>
      </td>
      <td rowspan="2">
         <b>System RAM<br>(Train/Inference)</b>
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         Training
      </td>
      <td>
         Inference
      </td>


   </tr>
   <tr align="center" valign="middle">
      <td>
         StreamVLN & InternVLA-N1
      </td>
      <td>
         A100
      </td>
      <td>
         RTX 4090 / A100
      </td>
      <td>
         80GB / 24GB
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         NavDP (VN Models)
      </td>
      <td>
        RTX 4090 / A100
      </td>
      <td>
         RTX 3060 / A100
      </td>
      <td>
         16GB / 2GB
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         CMA (VLN-PE Small Models)
      </td>
      <td>
         RTX 4090 / A100
      </td>
      <td>
         RTX 3060 / A100
      </td>
      <td>
         8GB / 1GB
      </td>
   </tr>
  </tbody>
</table>



## Quick Installation

We support two mainstream setups in this toolbox and the user can choose one given their requirements and devices.

### Basic Environment
For users that only need to run the model inference and visualize the planned trajectory results, we recommend run the simplest installation scripts:

```bash
    conda create -n internnav python=3.9
    pip install -r basic_requirements.txt
```

### Habitat Environment
For users that would like to train models or evaluate on Habitat without physics simulation, we recommend the following setup and easier environment installation.

#### Prerequisite
- Python 3.9
- Pytorch 2.1.2
- CUDA 12.4
- GPU: NVIDIA A100 or higher (optional for VLA training)

```bash
    conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
    git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines # install habitat_baselines
    pip install -r habitat_requirements.txt
```

### Isaac Sim Environment
For users that would like to evaluate the whole navigation system with VLN-PE on Isaac Sim, we recommend the following setup with NVIDIA RTX series GPUs for better rendering support.

#### Prerequisite
- Python 3.10 or higher
- CUDA 12.4
- GPU: NVIDIA RTX 4090 or higher (optional for VLA testing)

```bash
    # TODO: pull the docker with Isaac Sim and GRUtopia
    pip install -r isaac_requirements.txt --no-deps
```
```bash
# Make the script executable
chmod +x install.sh

# View available options
./install.sh --help
```


## Verification(TBD)

Please download our latest pretrained [checkpoint]() and run the following script to inference with visualization results.

```bash
    ./scripts/demo/internvla_n1.py --rgb_pkl ${SAVED_RGB_PKL} --depth_pkl ${SAVED_DEPTH_PKL} --output_path ${EXPECT_OUTPUT_PATH}
```

If it installed properly, you can find the evaluation results and the visualization in the `eval_results/internNav` directory:

🎉 Congratulations! You have successfully installed InternNav.

## Dataset Preparation

Please download the [InternData-N1]() and organize the data structure as follows.

```bash
data/
├── scene_datasets/
│   └── mp3d/
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
├── vln_n1/
│   ├── datasets/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
├── └── traj_data/
│       ├── images/
│       └── annotations.json
├── vln_ce/
│   ├── datasets/
│   └── traj_data/
└── vln_pe/
    ├── datasets/
    └── traj_data/
```
