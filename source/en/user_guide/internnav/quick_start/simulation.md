# Simulation Environments Setup

Our toolchain provides two Python environment solutions to accommodate different usage scenarios with the InternNav-N1 series model:

- For quick trials and evaluations of the InternNav-N1 model, we recommend using the [Habitat environment](#habitat-environment). This option offer allowing you to quickly test and eval the InternVLA-N1 models with minimal configuration.
- If you require high-fidelity rendering, training capabilities, and physical property evaluations within the environment, we suggest using the [Isaac Sim](#isaac-sim-environment) environment. This solution provides enhanced graphical rendering and more accurate physics simulations for comprehensive testing.

Choose the environment that best fits your specific needs to optimize your experience with the InternNav-N1 model. Note that both environments support the training of the system1 model NavDP.

## Install with Isaac Sim Environment

#### Install from Docker Image
To help you get started quickly, we've prepared a **Docker image** pre-configured with Isaac Sim 4.5, InternUtopia and models. A detailed guideline can be found at [challenge](https://github.com/InternRobotics/InternNav/tree/main/scripts/iros_challenge#-environment-setup) page.

You can pull the image (~17GB) and run evaluations in the container using the following command:
```bash
docker pull crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.2
```

Run the container by:
```bash
xhost +local:root # Allow the container to access the display

cd PATH/TO/INTERNNAV/  # where the latest code pulled

docker run --name internnav -it --rm --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "DISPLAY=${DISPLAY}" \
  --entrypoint /bin/bash \
  -w /root/InternNav \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ${PWD}:/root/InternNav \
  -v ${HOME}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${HOME}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${HOME}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${HOME}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${HOME}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${HOME}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${HOME}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${HOME}/docker/isaac-sim/documents:/root/Documents:rw \
  -v ${PWD}/data/scene_data/mp3d_pe:/isaac-sim/Matterport3D/data/v1/scans:rw \
  crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.2
```
After the container started, you can quickly start the env and install the InternNav:
```bash
conda activate internutopia
pip install -e .[isaac,model]
```
<!-- To help you get started quickly, we've prepared a Docker image pre-configured with Isaac Sim 4.5 and InternUtopia. You can pull the image and run evaluations in the container using the following command:
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0
docker run -it --name internutopia-container registry.cn-hangzhou.aliyuncs.com/internutopia/internutopia:2.2.0
``` -->
#### Conda Installation from Scratch
**Prerequisite**
- Ubuntu 20.04, 22.04
- Python 3.10.16 (3.10.* should be ok)
- NVIDIA Omniverse Isaac Sim 4.5.0
- NVIDIA GPU (RTX 2070 or higher)
- NVIDIA GPU Driver (recommended version 535.216.01+)
- PyTorch 2.5.1, 2.6.0 (recommended)
- CUDA 11.8, 12.4 (recommended)

Before proceeding with the installation, ensure that you have [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.


```bash
conda create -n <env> python=3.10 libxcb=1.14

# Install InternUtopia through pip.(2.1.1 and 2.2.0 recommended)
conda activate <env>
pip install internutopia

# Configure the conda environment.
python -m internutopia.setup_conda_pypi
conda deactivate && conda activate <env>
```
For InternUtopia installation, you can find more detailed [docs](https://internrobotics.github.io/user_guide/internutopia/get_started/installation.html) in [InternUtopia](https://github.com/InternRobotics/InternUtopia?tab=readme-ov-file).
```bash
# Install PyTorch based on your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Install other deps
cd Path/to/InternNav/
pip install -e .[isaac]
```

## Install with Habitat Environment
If you need to train or evaluate models on [Habitat](#optional-habitat-environment) without physics simulation, we recommend the following setup and easier environment installation.

#### Prerequisite
- Python 3.9
- Pytorch 2.6.0
- CUDA 12.4
- GPU: NVIDIA A100 or higher (optional for VLA training)

```bash
conda create -n <env> python=3.9
conda activate <env>
```
Install habitat sim and habitat lab:
```bash
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines
```
Install pytorch and other requirements:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cd Path/to/InternNav/
pip install -e .[habitat,internvla_n1]
```
