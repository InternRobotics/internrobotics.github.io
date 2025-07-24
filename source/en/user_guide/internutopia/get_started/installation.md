# 🚀Installation

## Prerequisites

- OS: Ubuntu 20.04/22.04
- RAM: 32GB+
- GPU: NVIDIA RTX 2070+ (must with RTX cores)
- NVIDIA Driver: 535.216.01+

> For complete requirements, please see [Isaac Sim's Requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html).
>
> InternUtopia is built upon NVIDIA's [Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Isaac Sim](https://developer.nvidia.com/isaac/sim) platforms, inheriting their dependencies. InternUtopia 2.2 specifically requires **Isaac Sim 4.5.0**. To ensure optimal performance and avoid any potential issues, it is essential to use this version rather than any other releases.

## Installation

Three ways of installation are provided:

- [Install from source (Linux)](#install-from-source-linux): recommended for users who want to thoroughly explore InternUtopia with Isaac Sim as a GUI application on Linux workstation with a NVIDIA GPU.
- [Install from PyPI (Linux)](#install-from-pypi-linux): recommended for users who want to use InternUtopia as a handy toolbox with Isaac Sim as a GUI application on Linux workstation with a NVIDIA GPU.
- [Install with Docker (Linux)](#install-with-docker-linux): recommended for users who prefer a stable and predictable environment, or deployment on remote servers or the Cloud

See more: [Differences Between Workstation And Docker](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_faq.html#isaac-sim-setup-differences).

Windows support is in our roadmap. Contributions are welcome!


### Install from source (Linux)

Before proceeding with the installation, ensure that you have [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) and [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

1. Clone the InternUtopia repository with [git](https://git-scm.com).
   ```bash
   $ git clone git@github.com:InternRobotics/InternUtopia.git
   ```

2. Navigate to InternUtopia root path and configure the conda environment.

   ```bash
   $ cd PATH/TO/INTERNUTOPIA/ROOT

   # Conda environment will be created and configured automatically with prompt.
   $ ./setup_conda.sh

   $ cd .. && conda activate internutopia  # or your conda env name
   ```

### Install from PyPI (Linux)

Before proceeding with the installation, ensure that you have [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) and [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

1. Create conda env with `python=3.10` specified.
    ```bash
   $ conda create -n <env> python=3.10 libxcb=1.14
   ```
2. Install InternUtopia through pip.

   **NOTE**: Ensure you have [git](https://git-scm.com) installed.

   ```bash
   $ conda activate <env>
   $ pip install internutopia
   ```
3. Configure the conda environment.

   ```bash
   $ python -m internutopia.setup_conda_pypi

   $ conda deactivate && conda activate <env>
   ```

### Install with Docker (Linux)

Make sure you have [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed. You can refer to the [container installation doc](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) of Isaac Sim for detailed instructions.

1. Clone the InternUtopia repository to any desired location.

   ```bash
   $ git clone git@github.com:InternRobotics/InternUtopia.git
   ```

1. Pull the InternUtopia docker image.

   ```bash
   $ docker pull registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0
   ```

1. Start docker container, replacing <your tag> with the above tag:

   ```bash
   $ xhost +local:root # Allow the container to access the display

   $ cd PATH/TO/INTERNUTOPIA/ROOT

   $ docker run --name internutopia -it --rm --gpus all --network host \
     -e "ACCEPT_EULA=Y" \
     -e "PRIVACY_CONSENT=Y" \
     -e "DISPLAY=${DISPLAY}" \
     -v /tmp/.X11-unix/:/tmp/.X11-unix \
     -v ${PWD}:/isaac-sim/InternUtopia \
     -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw \
     registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0
   ```

   You are now ready to use InternUtopia in this container.

   **NOTE**: If you are using a remote server without display, you can use the [WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html) to stream the simulation UI.

## Prepare Assets

```{note}
📝First of all you **MUST** complete the [User Agreement for GRScenes-100 Dataset Access](https://docs.google.com/forms/d/e/1FAIpQLSccX4pMb57eZbjXpH12Jz6WUBmCfeyc2t0s98k_u4Z-GD3Org/viewform?fbzx=8256642192244696391).
```

Then you can one of the following methods to get the assets:

- Download the assets automatically with [InternUtopia](#installation) installed:

  ```shell
  $ python -m internutopia.download_assets
  ```

  During the script execution, you can choose to download full assets (~80GB) or a minimum set (~500MB), and you will be asked to specify the local path to store the downloaded assets.

  > **NOTE**: If InternUtopia is installed with Docker, We recommend downloading the assets to a location under `/isaac-sim/InternUtopia/` in container (which is mounted from a host path) so that it can be retained across container recreations.

- Download the assets manually from [HuggingFace](https://huggingface.co/datasets/OpenRobotLab/GRScenes)/[ModelScope](https://www.modelscope.cn/datasets/Shanghai_AI_Laboratory/GRScenes/summary)/[OpenDataLab](https://openxlab.org.cn/datasets/OpenRobotLab/GRScenes), and then use the following command to tell InternUtopia where the assets locate if you are meant to use it with InternUtopia:

  ```shell
  $ python -m internutopia.set_assets_path
  ```

## Verify Installation

```shell
$ python -m internutopia.demo.h1_locomotion  # start simulation
```

If properly installed, Isaac Sim GUI window would pop up and you can see a humanoid robot (Unitree H1) walking following a pre-defined trajectory in Isaac Sim.

<video width="720" height="405" controls>
    <source src="../../../_static/video/h1_locomotion.webm" type="video/webm">
</video>

> **NOTE**: A slowdown is expected during first execution.
> Isaac sim requires some one-time startup setup the first time you start it.
> The process could take up to 5 minutes. This is expected behavior, and should only occur once!
