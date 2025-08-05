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

### ⚠️ Troubleshooting
**1. Tips for Slow or Unstable Networks**

If you encounter errors such as timeouts or incomplete downloads, especially in network-restricted or low-bandwidth environments, we recommend the following approaches.
- By default, `uv pip` uses relatively short HTTP timeouts. To extend the timeout, set the following environment variable before installation:
    ```bash
    export UV_HTTP_TIMEOUT=600  # Timeout in seconds (10 minutes)
    ```
- To ensure successful installation without network interruptions, you can download some large packages first and then install them locally:
    ```bash
    uv pip download -d ./wheelhouse "some-large-package"
    uv pip install --no-index --find-links=./wheelhouse "some-large-package"
    ```


**2. GCC Fails to Compile Due to Missing Dependencies**

When compiling C++ components (e.g., `building ManiSkill2_real2sim`), you might encounter errors related to GCC or missing shared libraries. This guide walks you through how to resolve them without root/sudo permissions.

- Step 1: Use a modern GCC (recommended ≥ 9.3.0). Older system compilers (e.g., GCC 5.x or 7.x) may not support required C++ standards. It's recommended to switch to GCC 9.3.0 or newer:
    ```bash
    export LD_LIBRARY_PATH=${PATH_TO}/gcc/gcc-9.3.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=${PATH_TO}/gcc/gcc-9.3.0/bin:$PATH
    ```
    > ⚠️ Note: Simply using a newer compiler might not be enough — it may depend on shared libraries that are not available on your system.
- Step 2: Manually install required libraries. If you encounter errors like: `error while loading shared libraries: libmpc.so.2 (libmpfr.so.1, libgmp.so.3)`. If you do have `sudo` privileges, the easiest way is to install the required libraries system-wide using your system package manager.
    ```bash
    sudo apt update
    sudo apt install gcc-9 g++-9
    ```
    Or, you need to manually compile and install the following dependencies locally:
    ```bash
    INSTALL_DIR=$HOME/local
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    wget https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz
    tar -xf gmp-6.2.1.tar.xz && cd gmp-6.2.1
    ./configure --prefix="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.xz
    tar -xf mpfr-4.2.1.tar.xz && cd mpfr-4.2.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    echo "📦 Installing MPC..."
    wget https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz
    tar -xf mpc-1.3.1.tar.gz && cd mpc-1.3.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR" --with-mpfr="$INSTALL_DIR"
    make -j$(nproc)
    make install
    ```
- Step 3 (Optional): Fix Missing `.so` Versions. Sometimes you have the correct library version (e.g., `libgmp.so.10`), but GCC expects an older symlink name (e.g., `libgmp.so.3`). You can fix missing library versions with symlinks.
- Step 4: Export the Library Path. Make sure the compiler can find your locally installed shared libraries:
    ```bash
    export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
    ```



## Verification (WIP)

You can evaluate the pretrained **GR00t-N1** model on the `Simpler-Env` benchmark using a **client-server** architecture. This requires two separate terminal sessions:

**🖥 Terminal 1: Launch the policy server (model side)**

Activate the environment for the model, and start the policy server:
```bash
source .venv/model/bin/activate
python scripts/eval/start_policy_server.py
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
   - [GenManip-v1](https://huggingface.co/datasets/InternRobotics/InternData-GenmanipTest)
   - [CALVIN_ABC](https://huggingface.co/datasets/InternRobotics/InternData-Calvin_ABC)
   - [Google-Robot](https://huggingface.co/datasets/InternRobotics/InternData-fractal20220817_data)
   - [BridgeData-v2](https://huggingface.co/datasets/InternRobotics/InternData-BridgeV2)



## ⚠️ Troubleshooting

### 1. Conda Terms of Service Acceptance Error

If you encounter the following error during the genmanip installation:
```bash
CondaToNonInteractiveError: Terms of Service have not been accepted for the following channels:
    • https://repo.anaconda.com/pkgs/main
    • https://repo.anaconda.com/pkgs/r
```
Manually accept the Terms of Service for each affected channel by running these commands:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```


### 2. Tips for Slow or Unstable Networks

If you encounter errors such as timeouts or incomplete downloads, especially in network-restricted or low-bandwidth environments, we recommend the following approaches.

- Step 1: By default, `uv pip` uses relatively short HTTP timeouts. To extend the timeout, set the following environment variable before installation:
    ```bash
    export UV_HTTP_TIMEOUT=600  # Timeout in seconds (10 minutes)
    ```

- Step 2: Locate the package that failed to install. When running the `install.sh` script, if a package fails to install, identify the failing line in the output. This usually indicates which package wasn't downloaded properly.
- Step 3: Dry-run to preview packages and versions. Use uv pip install --dry-run` to preview which packages (and exact versions) are going to be installed. For example:
    ```bash
    uv pip install "git+https://github.com/NVIDIA/Isaac-GR00T.git#egg=isaac-gr00t[base]" --dry-run
    ```
    This will list all packages along with their resolved versions, including those that might fail due to slow download.
- Step 4: Activate your environment. Before manually installing the package, make sure you're in the correct virtual environment.
    ```bash
    source .venv/{your_env_name}/bin/activate
    ```
- Step 5: Manually install the problematic package. After identifying the package and version, install it manually:
    ```bash
    uv pip install torch==2.5.1  # Replace with your actual package and version
    ```

<!---
- By default, `uv pip` uses relatively short HTTP timeouts. To extend the timeout, set the following environment variable before installation:
    ```bash
    export UV_HTTP_TIMEOUT=600  # Timeout in seconds (10 minutes)
    ```
- To ensure successful installation without network interruptions, you can download some large packages first and then install them locally:
    ```bash
    uv pip download -d ./wheelhouse "some-large-package"
    uv pip install --no-index --find-links=./wheelhouse "some-large-package"
    ```
--->

### 3. `import simpler_env` Fails Due to Missing Vulkan Library

If you encounter the following error when trying to import simpler_env:
```bash
>>> import simpler_env
Traceback (most recent call last):
  ...
ImportError: libvulkan.so.1: cannot open shared object file: No such file or directory
```
You can resolve this issue by installing the Vulkan runtime library via `apt`:
```bash
sudo apt update
sudo apt install libvulkan1
sudo ldconfig
```

### 4. GCC Fails to Compile Due to Missing Dependencies

When compiling C++ components (e.g., `building ManiSkill2_real2sim`), you might encounter errors related to GCC or missing shared libraries. This guide walks you through how to resolve them without root/sudo permissions.

- Step 1: Use a modern GCC (recommended ≥ 9.3.0). Older system compilers (e.g., GCC 5.x or 7.x) may not support required C++ standards. It's recommended to switch to GCC 9.3.0 or newer:
    ```bash
    export LD_LIBRARY_PATH=${PATH_TO}/gcc/gcc-9.3.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=${PATH_TO}/gcc/gcc-9.3.0/bin:$PATH
    ```
    > ⚠️ Note: Simply using a newer compiler might not be enough — it may depend on shared libraries that are not available on your system.
- Step 2: Manually install required libraries. If you encounter errors like: `error while loading shared libraries: libmpc.so.2 (libmpfr.so.1, libgmp.so.3)`. If you do have `sudo` privileges, the easiest way is to install the required libraries system-wide using your system package manager.
    ```bash
    sudo apt update
    sudo apt install gcc-9 g++-9
    ```
    Or, you need to manually compile and install the following dependencies locally:
    ```bash
    INSTALL_DIR=$HOME/local
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    wget https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz
    tar -xf gmp-6.2.1.tar.xz && cd gmp-6.2.1
    ./configure --prefix="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.xz
    tar -xf mpfr-4.2.1.tar.xz && cd mpfr-4.2.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR"
    make -j$(nproc)
    make install
    cd "$INSTALL_DIR"

    echo "📦 Installing MPC..."
    wget https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz
    tar -xf mpc-1.3.1.tar.gz && cd mpc-1.3.1
    ./configure --prefix="$INSTALL_DIR" --with-gmp="$INSTALL_DIR" --with-mpfr="$INSTALL_DIR"
    make -j$(nproc)
    make install
    ```
- Step 3 (Optional): Fix Missing `.so` Versions. Sometimes you have the correct library version (e.g., `libgmp.so.10`), but GCC expects an older symlink name (e.g., `libgmp.so.3`). You can fix missing library versions with symlinks.
- Step 4: Export the Library Path. Make sure the compiler can find your locally installed shared libraries:
    ```bash
    export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
    ```
