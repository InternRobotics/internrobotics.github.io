<style>
  .mytable {
    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 8px;
    border-collapse: separate;
    border-spacing: 0;
  }
  .mytable th, .mytable td {
    border: 1px solid rgba(128, 128, 128, 0.5);
    padding: 6px 12px;
  }
</style>

# 🛠️ Installation Guide
```{important}
We are actively fixing mistakes in the document. If you find any errors in the documentation, please feel free to [open an issue](https://github.com/InternRobotics/InternRobotics-doc/issues). Your help in improving the document is greatly appreciated 🥰.
```

😄 Don’t worry — both [Quick Installation](#quick-installation) and [Dataset Preparation](#dataset-preparation) are beginner-friendly.


<!-- > 💡NOTE \
> 🙋 **[First-time users:](#-lightweight-installation-recommended-for-beginners)** Skip GenManip for now — it requires installing NVIDIA [⚙️ Isaac Sim](#), which can be complex.
Start with **CALVIN** or **SimplerEnv** for easy setup and full training/eval support.\
> 🧠 **[Advanced users:](#-full-installation-advanced-users)** Feel free to use all benchmarks, including **GenManip** with Isaac Sim support. -->

<!-- > For 🙋**first-time** users, we recommend skipping the GenManip benchmark, as it requires installing NVIDIA [⚙️ Isaac Sim](#) for simulation (which can be complex).
Instead, start with **CALVIN** or **SimplerEnv** — both are easy to set up and fully support training and evaluation. -->

<!-- This guide provides comprehensive instructions for installing and setting up the InternManip robot manipulation learning suite. Please read through the following prerequisites carefully before proceeding with the installation. -->

## Prerequisites

InternManip works across most hardware setups.
Just note the following exceptions:
- **GenManip Benchmark** must run on **NVIDIA RTX series GPUs** (e.g., RTX 4090).
- GR00T requires **CUDA 12.4 installed system-wide (not via Conda)**.
- **Miniconda**.

### Overall Requirements
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
         CALVIN
      </td>
       <td>
         Simpler-Env
      </td>
       <td>
         Genmanip
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
         ✅
      </td>
      <td>
         ❌
      </td>
   </tr>
  </tbody>
</table>

```{note}
We provide a flexible installation tool for users who want to use InternManip for different purposes. Users can choose to install the training and inference environment, and the individual simulation environment independently.
```

<!-- Before installing InternManip, ensure your system meets the following requirements based on the specific models and benchmarks you plan to use. -->

### Model-Specific Requirements

<table align="center" class="mytable">
  <tbody>
    <tr align="center" valign="middle">
      <td rowspan="2">
         <b>Models</b>
      </td>
      <td colspan="3">
         <b>Minimum GPU Requirement</b>
      </td>
      <td rowspan="2">
         <b>System RAM<br>(Train/Inference)</b>
      </td>
      <td rowspan="2">
         <b>CUDA</b>
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         Training (Full)
      </td>
      <td>
         Training (LoRA)
      </td>
       <td>
         Inference
      </td>


   </tr>
   <tr align="center" valign="middle">
      <td>
         GR00T-N1/1.5
      </td>
      <td>
        RTX 4090 / A100
      </td>
      <td>
         -
      </td>
      <td>
         RTX 3090 / A100
      </td>
      <td>
         24GB / 8GB
      </td>
      <td>
         ⚠️ 12.4
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         Pi-0
      </td>
      <td>
        RTX 4090 (48G) / A100
      </td>
      <td>
         RTX 4090
      </td>
      <td>
         RTX 3090 / A100
      </td>
      <td>
         70GB / 8GB
      </td>
      <td>
         -
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         Diffusion Policy (DP)
      </td>
      <td>
        RTX 2080
      </td>
      <td>
         -
      </td>
      <td>
         RTX 2070
      </td>
      <td>
         16GB / 8GB
      </td>
      <td>
         -
      </td>
   </tr>
  </tbody>
</table>



## Quick Installation

We provide a unified installation script that handles environment setup and dependency installation automatically.

### Step 1: Clone the Repository and Install uv
```bash
# Clone the main repository
git clone https://github.com/internrobotics/internmanip.git
cd internmanip

# Initialize and update submodules
git submodule update --init --recursive

# Skip the following commands if you have installed uv
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your shell or source the profile
source ~/.bashrc
```

### Step 2: Run the Installation Script
```bash
# Make the script executable
chmod +x install.sh

# View available options
./install.sh --help
```


#### 🙋 Lightweight Installation （Recommended for beginners）:
```bash
./install.sh --beginner
```


#### 🧠 Full Installation (Advanced users):
```bash
./install.sh --all
```
> 💡 Tips:
> Before installing genmanip, please ensure that Anaconda and Isaac Sim 4.5.0 are properly set up on your system. You can download the standalone version from [👉 Download Isaac Sim 4.5.0 (RC36)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip)
>
> After downloading, extract the archive to a suitable directory (e.g., ~/tools/isaac-sim-4.5.0). You should set the path to your local Isaac Sim installation during running `install.sh`.


#### Available Installation Options:
```bash
Usage:
  --calvin [NAME]         Create Calvin benchmark virtual environment and install dependencies
  --simpler-env [NAME]    Create SimplerEnv benchmark virtual environment and install dependencies
  --genmanip [NAME]       Create GenManip benchmark virtual environment and install dependencies
  --model [NAME]          Create model virtual environment and install dependencies
  --all                   Create all virtual environments and install dependencies (recommended for advanced users)
  --beginner              Create beginner virtual environments and install dependencies (without genmanip, recommended for beginners)

Customization Options:
  --venv-dir DIR          Set custom virtual environment root directory (default: .venv)
  --python-version V      Set default Python version (recommended default: 3.10)

Examples:
  ./install.sh --venv-dir ./my_envs --model
  ./install.sh --calvin calvin-test --model model-test
  ./install.sh --python-version 3.10 --calvin calvin-dev --simpler-env simpler-dev
  ./install.sh --all
  ./install.sh --beginner
  --help                  Show help information
```

#### Selective Installation:
```bash
# Install only specific components
./install.sh --model --genmanip
```

#### Activate Virtual Environment
After installation, virtual environments are created in the `.venv` directory by default.

```bash
# List available environments
ls .venv/

# Activate a specific environment
source .venv/{environment_name}/bin/activate

# Example: Activate model environment
source .venv/model/bin/activate

# Deactivate environment
deactivate
```

> 🟡 Note: Unlike other environments that use venv, genmanip relies on Conda for environment management. You should always activate the environment using:
> ```bash
> conda activate genmanip
> ```


Optionally, users can customize the virtual environments directory path by passing the `--venv-dir {path}` option when executing `install.sh`.

```bash
./install.sh --venv-dir ./my_envs --model
```


```{tips}
Since we install packages via `uv`, if there are network fluctuations causing some packages to fail installing, the process will skip them and still output `✅ install finished`. This does not guarantee that all packages were installed successfully. Therefore, please **check the logs carefully** after installation to see if there are any errors.
```


```{important}
If you encounter any issues during the installation, please first check the <a href="../troubleshooting.html">Troubleshooting</a> section.
```


## Verification

You can evaluate the pretrained **GR00t-N1** model on the `Simpler-Env` benchmark using a **client-server** architecture. This requires two separate terminal sessions:

**🖥 Terminal 1: Launch the policy server (model side)**

Activate the environment for the model, and start the policy server:
```bash
source .venv/model/bin/activate
python scripts/eval/start_agent_server.py
```
This will start the policy server that listens for observation inputs and sends back action predictions.

**🖥 Terminal 2: Launch the evaluator (benchmark side)**

Activate the environment for Simpler-Env, and run the evaluator:
```bash
source .venv/simpler-env/bin/activate
python scripts/eval/start_evaluator.py --config run_configs/examples/internmanip_demo.py --server
```

This will run the evaluation loop that sends observations to the model server and executes returned actions in the environment.

If it installed properly, you can find the evaluation results and the visualization in the `logs/demo/gr00t_n1_on_simpler` directory:

<!-- <p align="center">
<video width="640" height="480" controls>
    <source src="../../../_static/video/manip_verification.webm" type="video/webm">
</video>
</p> -->
<p align="center">
<video width="640" height="480" controls>
    <source src="../../../_static/video/widowx_bridge.webm" type="video/webm">
</video>
</p>
🎉 Congratulations! You have successfully installed InternManip.


> ⚠️ Note: The visualization results are only intended to verify the environment setup. You do not need to pay attention to the model’s grasp success rate shown in the videos.


## Dataset Preparation

### Automatic Download
Datasets, model weights, and benchmark assets are automatically downloaded when running the code for the first time. The default download location is `${repo_root}/data`. The system will prompt you to download required datasets.
```{warning}
Please ensure you have enough disk space available before starting the download. For cluster users, we recommend creating a symbolic link to a data storage for `${repo_root}/data` to avoid disk space issues.
```

### Manual Download
If you prefer manual dataset preparation:

1. **Visit our platform:** [Dataset Platform](https://huggingface.co/InternRobotics)
2. **Download datasets** based on your needs:
   - [IROS-2025-Challenge-Manip](https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Manip)
   - [GenManip-v1](https://huggingface.co/datasets/InternRobotics/InternData-GenmanipTest)
   - [CALVIN_ABC](https://huggingface.co/datasets/InternRobotics/InternData-Calvin_ABC)
   - [Google-Robot](https://huggingface.co/datasets/InternRobotics/InternData-fractal20220817_data)
   - [BridgeData-v2](https://huggingface.co/datasets/InternRobotics/InternData-BridgeV2)
