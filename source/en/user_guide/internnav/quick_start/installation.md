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

# Installation Guide

This page provides detailed instructions for installing **InternNav** in inference-only mode, such as when deploying **InternVLA-N1** on your own robot or with a custom dataset.
Follow the steps below to set up the environment and run inference with the model.

If you want to **reproduce the results** presented in the [technical report](https://internrobotics.github.io/internvla-n1.github.io/), please follow this page, and also complete the following sections on [Simulation Environments Setup](./simulation.md), [Dataset Preparation](./interndata.md) and [Training and Evaluation](./train_eval.md). 

For more advanced examples, refer to these demos:

-  [**InternVLA-N1 Inference-only Demo**](https://githubtocolab.com/InternRobotics/InternNav/blob/main/scripts/notebooks/inference_only_demo.ipynb)
-  [**Real-World Unitree Go2 Deploy Script**](https://github.com/kew6688/InternNav/tree/main/scripts/realworld)


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
### Install InternNav
Clone the **InternNav** repository:
```bash
git clone https://github.com/InternRobotics/InternNav.git --recursive
```
After pull the latest code, install InternNav with models:
```bash
# create a new isolated environment for model server
conda create -n <internnav> python=3.10 libxcb=1.14
conda activate <internnav>

# install PyTorch (CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# install InternNav with model dependencies
pip install -e .[model] --no-build-isolation

```

To enable additional functionalities, several install flags are available:

| Flag             | Description                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `.`              | Install the core only for the InternNav framework.                                                                                          |
| `[model]`        | Install all dependencies for training and evaluating models (CMA, RDP, NavDP, InternVLA-N1).                                                |
| `[isaac]`        | Install dependencies for the [Isaac environment](./simualtion.md).                                                                        |
| `[habitat]`      | Install dependencies for the [Habitat environment](./simualtion.md).                                                                      |



### Download Checkpoints
1. **InternVLA-N1 pretrained Checkpoints**
- Download our latest pretrained [checkpoint](https://huggingface.co/InternRobotics/InternVLA-N1) of InternVLA-N1 and run the following script to inference with visualization results. Move the checkpoint to the `checkpoints` directory.
2. **DepthAnything v2 Checkpoints**
- Download the DepthAnything v2 pretrained [checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth). Move the checkpoint to the `checkpoints` directory.

## Verification

InternNav adopts a **client–server design** to simplify model deployment and prediction. 

To verify the installation of **InternNav**, start the model server first.
```bash
python scripts/eval/start_server.py --port 8087
```
The output should be:
```
Starting Agent Server...
Registering agents...
INFO:     Started server process [18877]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8087 (Press CTRL+C to quit)
```

To verify the installation of **internvla-n1**. Initialize the internvla-n1 agent by
```bash
from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient

agent=AgentCfg(
      server_host='localhost',
      server_port=8087,
      model_name='internvla_n1',
      ckpt_path='',
      model_settings={
            'policy_name': "InternVLAN1_Policy",
            'state_encoder': None,
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
      }
)
agent = AgentClient(cfg.agent)
```
The output should be something like:
```
Loading navdp model: NavDP_Policy_DPT_CriticSum_DAT
Pretrained: None
No pretrained weights provided, initializing randomly.
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.06it/s]
INFO:     ::1:38332 - "POST /agent/init HTTP/1.1" 201 Created
```

Load a capture frame from RealSense DS455 camera:
```
from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta
rs_meta_path = '/root/InternNav/scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

fake_obs_640 = load_obs_from_meta(rs_meta_path)
fake_obs_640['instruction'] = 'go to the red car'
print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)
```
The output should be:
```
(480, 640, 3) (480, 640)
```

Test model inference
```
action = agent.step([obs])[0]['action'][0]
print(f"Action taken: {action}")
```

The output should be:
```
============ output 1  ←←←←
s2 infer finish!!
get s2 output lock
=============== [2, 2, 2, 2] =================
Output discretized traj: [2] 0
INFO:     ::1:46114 - "POST /agent/internvla_n1/step HTTP/1.1" 200 OK
Action taken: 2
```

Congrats, now you have made one prediction. In this task, the agent convert the trajectory output to discrete action. Apply this action "turn left" (2) to real robot controller by using `internnav.env.real_world_env`. 

Checkout the real deploy demo video:

<video width="720" height="405" controls>
    <source src="../../../_static/video/nav_demo.webm" type="video/webm">
</video>

for more details, check out the [**Internvla_n1 Inference-only Demo**](https://githubtocolab.com/InternRobotics/InternNav/blob/main/scripts/notebooks/inference_only_demo.ipynb).