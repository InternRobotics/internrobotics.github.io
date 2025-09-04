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


# 🏃🏻‍♂️ Training and Evaluation


This document guides you through:
- **[Minimal validation](#minimal-validation-training).** Verify that your environment and setup can successfully train a model on a small dataset.
- **[Large-scale multi-node training](#large-scale-finetuning).** Learn how to finetune models on multiple GPUs and nodes.
- **[Supported models and datasets](#available-models-and-datasets).** Get an overview of the built-in policies and datasets you can use.
- **[Evaluate your trained models](#evaluation-and-benchmarking)** using **closed-loop benchmarking**, allowing you to measure the **success rate (SR)** on various tasks.
- **[Extend the framework](#available-benchmarks)** by adding your own **custom benchmarks**.


## Minimal Validation Training

Before running any script, make sure to activate your virtual environment and correctly set the `PYTHONPATH`. This ensures that all local modules can be correctly discovered and executed:
```bash
source .venv/{environment_name}/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```


We provide several built-in policies such as **GR00T-N1**, **GR00T-N1.5**, **Pi-0**, **DP-CLIP**, and **ACT-CLIP**.
To quickly verify your setup, you can train the **Pi-0** model on the `genmanip-demo` dataset (300 demonstrations of the instruction *"Move the milk carton to the top of the ceramic bowl"*).
This requires **1 GPU with at least 24GB memory**:

```bash
torchrun --nnodes 1 --nproc_per_node 1 \       # number of processes per node, e.g., 1
   scripts/train/train.py \
   --config run_configs/train/pi0_genmanip_v1.yaml # Config file that specifies which model to train on which dataset, along with hyperparameters
```

> 😄 When you run the script, it will prompt you to log in to Weights & Biases (WandB). This integration allows you to monitor your training process in real time via the WandB dashboard.


The script will also automatically download all required models and datasets from Hugging Face into the Hugging Face cache directory (by default located at `~/.cache/huggingface/`). If you're concerned about storage space or want to customize the cache location, you can now specify it directly in the YAML configuration file using the `hf_cache_dir` field:

```bash
hf_cache_dir: /your/custom/cache/path
```
> 💡 Note! The download process may take some time depending on your network speed—please be patient.

### ⚠️ Common Issues



1. **Authentication Required:** If you see an error related to missing access rights, make sure you’ve logged into Hugging Face CLI:

   ```bash
   huggingface-cli login
   ```

2. **403 Forbidden: Gated Repository Access:** If you encounter the following error:

   ```pgsql
   403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings to view this repository.
   ```
   Then ensure that your Hugging Face access token has the correct fine-grained permissions enabled for accessing gated repositories. You can verify and adjust these in your Hugging Face account's [Access Tokens settings](https://huggingface.co/settings/tokens).




<!-- Once training is finished, you can evaluate the trained model with:
```bash
python scripts/eval/start_evaluator.py \
   --config run_configs/examples/dp_clip_on_genmanip.py
``` -->
<!-- python scripts/eval/eval.py \
   --benchmark genmanip-demo \         # name of the dataset or benchmark
   --model-name dp-clip \
   --model-path ./Checkpoints/runs/genmanip-demo/dp-clip \  # model checkpoint path
   --results-dir eval_results/genmanip-demo/dp-clip \       # directory for evaluation results
   --visualization                     # enable visualization -->


## Large-Scale Finetuning

### Single Node (Multi-GPU)
To finetune a built-in model such as **Pi-0** on the **GenManip** dataset using **8 GPUs**, you can use the following srun command:

```bash
srun --job-name=pi0_genmanip --gres=gpu:8 --ntasks-per-node=1 \
torchrun \
   --nnodes 1 \
   --nproc_per_node 8 \
   scripts/train/train.py \
   --config run_configs/train/pi0_genmanip_v1.yaml
```
### Multi-Node Multi-GPU (Slurm)
We also provide Slurm scripts for multi-node training.

**Step 1:** Create `train_pi0_genmanip_slurm.sh`:
```bash
#!/bin/bash
set -e

master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export NCCL_SOCKET_IFNAME=$1
export NCCL_IB_HCA=$2

torchrun \
   --nnodes=$SLURM_NNODES \
   --nproc_per_node=8 \
   --node_rank=$SLURM_PROCID \
   --master_port=29500 --master_addr=$master_addr \
   scripts/train/train.py \
   --config run_configs/train/pi0_genmanip_v1.yaml
```
**Step 2:** Create `multinode_submit.slurm`:
```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1   # 1 task per node
#SBATCH --gpus-per-task=8     # 8 GPUs per task
srun bash train_pi0_genmanip_slurm.sh
```
**Step 3:** Start training:
```bash
sbatch multinode_submit.slurm
```
> 💡 Tips: The recommended training setup is a global batch size of 2048 for 50,000 steps, which typically takes approximately 500 GPU hours (Assuming each node has 8 GPUs).

## Customizing Training with Your Own YAML Config


If you would like to train with your own choice of model and dataset, you can simply create a custom YAML configuration file and pass it to the `--config` argument in the training script.

For example, to train the pre-registered **Pi-0** model on the **GenManip** dataset, a minimal YAML configuration might look like this:
```yaml
model_type: pi0                        # Name of a pre-registered model
dataset_path: InternRobotics/InternData-GenmanipTest  # Can be a HuggingFace Hub ID or local path
data_config: genmanip_v1              # Pre-registered dataset configuration
base_model_path: lerobot/pi0          # (Optional) Overrides the model checkpoint path; will default to HF if omitted
```

**💡 Notes:**

- `model_type`: Must match the name of a model that has already been registered within InternManip.
- `dataset_path`: Can be a HuggingFace ID (e.g., `InternRobotics/InternData-GenmanipTest`) or a local directory where the dataset is downloaded.
- `data_config`: Refers to a dataset configuration preset (e.g., for preprocessing or loading behavior), also pre-registered in the codebase.
- `base_model_path`: This is optional. If the selected `model_type` is supported and known, InternManip will automatically resolve and download the correct checkpoint from HuggingFace. If you’ve already downloaded a model locally or want to use a custom one, you can specify the path here directly.

By editing or extending this YAML file, you can quickly try different models, datasets, or training setups — all without modifying the training script.


## Available Models and Datasets

<!-- At present, we have the following built-in models and datasets that can be used for finetuning. -->

When creating your own YAML config file for training or evaluation, you can directly refer to the following officially supported values:

- Use values from the `${model_type}` and `${base_model_path}` columns below to populate the corresponding fields in your YAML.
- Similarly, values from the `${data_config}` and `${dataset_path}` columns can be used to specify the dataset configuration and loading path.

<!-- This ensures consistency with the models and datasets that have been pre-registered within InternManip. -->

The following are the supported models along with their HuggingFace IDs:



<table align="center" class="mytable">
  <thead>
    <tr align="center" valign="middle">
      <td><b><code>${model_type}</code></b></td>
      <td><b><code>${base_model_path}</code></b></td>
    </tr>
  </thead>
  <tbody>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/Physical-Intelligence/openpi"><code>pi0</code></a></td>
      <td><code>lerobot/pi0</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><code>pi0fast</code></td>
      <td><code>pi0fast_base</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/NVIDIA/Isaac-GR00T"><code>gr00t_n1</code></a></td>
      <td><code>nvidia/GR00T-N1-2B</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/NVIDIA/Isaac-GR00T"><code>gr00t_n1_5</code></a></td>
      <td><code>nvidia/GR00T-N1.5-3B</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/real-stanford/diffusion_policy"><code>dp_clip</code></a></td>
      <td><code>None</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/Shaka-Labs/ACT"><code>act_clip</code></a></td>
      <td><code>None</code></td>
    </tr>
  </tbody>
</table>


Below are the datasets officially integrated into InternManip:

<table align="center" class="mytable">
  <thead>
    <tr align="center" valign="middle">
      <td><b><code>${data_config}</code></b></td>
      <td><b><code>${dataset_path}</code></b></td>
    </tr>
  </thead>
  <tbody>
    <tr align="center" valign="middle">
      <td><a href="https://arxiv.org/abs/2506.10966"><code>genmanip_v1</code></a></td>
      <td><code>InternRobotics/InternData-GenmanipTest</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/mees/calvin"><code>calvin_abc</code></a></td>
      <td><code>InternRobotics/InternData-Calvin_ABC</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/simpler-env/SimplerEnv"><code>google_robot</code></a></td>
      <td><code>InternRobotics/InternData-fractal20220817_data</code></td>
    </tr>
    <tr align="center" valign="middle">
      <td><a href="https://github.com/simpler-env/SimplerEnv"><code>bridgedata_v2</code></a></td>
      <td><code>InternRobotics/InternData-BridgeV2</code></td>
    </tr>
  </tbody>
</table>




<!-- - **Models**:
  - [`pi0`](https://github.com/Physical-Intelligence/openpi): A pre-trained model for general-purpose manipulation tasks.
  - [`gr00t-n1/1.5`](https://github.com/NVIDIA/Isaac-GR00T): An advanced model with additional capabilities.
  - [`dp-clip`](https://github.com/real-stanford/diffusion_policy): A model that combines diffusion policy with CLIP for instruction-guided manipulation.
  - [`act-clip`](https://github.com/Shaka-Labs/ACT): A model that integrates Action-chunking Transformer with CLIP for instruction-guided manipulation.
- **Datasets**:
  - [`genmanip-v1`](https://arxiv.org/abs/2506.10966): A LLM-driven tabletop simulation platform tailored for policy generalization studies.
  - [`calvin-abcd`](https://github.com/mees/calvin): An open-source simulated benchmark to learn long-horizon language-conditioned tasks.
  - [`google-robot`](https://github.com/simpler-env/SimplerEnv): Google-Robot dataset built upon Simpler-Env environments.
  - [`bridgedata-v2`](https://github.com/simpler-env/SimplerEnv): BridgeData-v2 dataset built upon Widowx embodiment. -->

<!-- To finetune your own model, you can follow the same command structure as above, replacing the `--model_name` with your registered custom model path and adjusting other parameters as needed. Please refer to [`How to customize your dataset`](../tutorials/dataset.md) and [`How to add a model`](../tutorials/model.md) to learn how to prepare your dataset and model for finetuning. -->



<!-- This document will guide you to:



Below we use **GenManip** as an example to demonstrate the evaluation process. -->


<!-- This section provides a quick start guide for evaluating models in the GRManipulation framework. The evaluation process is designed to be straightforward, allowing users to quickly assess the performance of their models on various tasks.  -->

<!-- ## Evaluation in a single process
By default, the inference of model will be running in the main loop sharing the same process with the `env`. You can evaluate `pi0` on the `Genmanip` benchmark in a single process using the following command: -->

<!-- ```bash
python scripts/eval/start_evaluator.py \
   --config scripts/eval/config/pi0_on_genmanip.py
``` -->

## Evaluation and Benchmarking (WIP)


The default evaluation setup adopts a client-server architecture where the policy (model) and the environment run in separate processes. This improves compatibility and modularity for large-scale benchmarks.
You can evaluate `pi0` on the `genmanip` benchmark in a single process using the following command:

**Configuring Evaluation: Key Setup and Model Checkpoint**

The evaluation requires properly configuring the evaluation config file. Below is an example of a config instance.
Please pay special attention to modifying the `base_model_path` field, which should point to your finetuned model checkpoint.

```python
from internmanip.configs import *
from pathlib import Path

eval_cfg = EvalCfg(
    eval_type='genmanip',
    agent=AgentCfg(
        agent_type='pi0',
        base_model_path='/PATH/TO/YOUR/CHECKPOINT',  # <--- MODIFY THIS PATH
        agent_settings={...},
        model_kwargs={...},
        server_cfg=ServerCfg(...),
    ),
    env=EnvCfg(...),
)
```


```{important}
You must modify the `base_model_path` to the path of your own finetuned checkpoint, which is different from the HuggingFace loaded checkpoint — you should **NOT** use the unfinetuned checkpoint directly for evaluation!
```

Also, note that the evaluation data for `genmanip` is different from the training data, so please be careful to distinguish between them when running evaluations.


**🖥 Terminal 1: Launch the Policy Server (Model Side)**

Activate the environment for the model and start the policy server:
```bash
source .venv/model/bin/activate
python scripts/eval/start_agent_server.py
```
This server listens for observation inputs from the environment and responds with action predictions from the model.

**🖥 Terminal 2: Launch the Evaluator (Environment Side)**
```bash
source .venv/simpler_env/bin/activate
python scripts/eval/start_evaluator.py --config run_configs/eval/pi0_on_genmanip.py --server
```

This client sends observations to the model server, receives actions, and executes them in the environment.


<!-- ### 1. Client-Server Setup

The hardware (e.g., RTX series v.s. A100) or package requirements for model and benchmark simulator can potentially be conflicting. To solve this, we use a **client-server** architecture to separate the model inference from the simulation. The **server** will run the model inference and the **client** will run the simulator. The communication between them is done through a socket connection.

Specifically, you should first launch the model server with the following command:


```bash
cd path/to/internmanip
conda activate {model_env}
python -m scripts.eval.start_policy_server
```


Then you should run the benchmark client with the following command:

```bash
conda activate {benchmark_env}
python scripts/eval/start_evaluator.py \
   --config scripts/eval/config/pi0_on_genmanip.py \
   --server
```

### 2. Evaluation Output -->

The terminal prints SR (Success Rate) information for each episode and task:

```json
{
    "success_episodes": [
        {"task_name": "tasks/...", "episode_name": "010", "episode_sr": 1.0, ...}
    ],
    "failure_episodes": [],
    "success_rate": 1.0
}
```
<!-- ### 3. Log & Result Saving -->

<!-- When `env_setting.res_save_path` in `scripts/eval/configs/config.py` is set to a valid directory, the evaluation process will automatically store:
- **Intermediate results**: RGB images (if `is_save_img=True`), robot state information.
- **Result summary**: A `result.json` file containing task-level and episode-level success rates (same as terminal output). -->

You can view the images generated during evaluation in the `logs/demo/gr00t_n1_on_simpler` directory.



<p align="center">
<video width="640" height="480" controls>
    <source src="../../../_static/video/manip_eval.webm" type="video/webm">
</video>
</p>



<!-- We provide an example bash script `bash_scripts/eval_genmanip.sh` for running the evaluation on the `Genmanip` benchmark. You can run it with the following command:

```bash
bash bash_scripts/eval_genmanip.sh
``` -->

> You can modify the bash script according to your resource availability and requirements.

## Available Benchmarks
The following benchmarks are currently available for evaluation:
- **[GenManip](https://arxiv.org/abs/2506.10966)**
- **[CALVIN](https://github.com/mees/calvin)**
- **[Simpler-Env](https://github.com/simpler-env/SimplerEnv)**


InternManip offers implementations of multiple manipulation policy models—**GR00T-N1**, **GR00T-N1.5**, **Pi-0**, **DP-CLIP**, and **ACT-CLIP**—as well as curated datasets including **GenManip**, **Simpler-Env**, and **CALVIN**, all organized in the standardized **LeRobot** format.

The available `${MODEL}`, `${DATASET}`, `${BENCHMARK}` and their results are summarized in the following tables:

### CALVIN (ABC-D) Benchmark
| Model  | Dataset/Benchmark | Score (Main Metric) | Model Weights |
| ------------ | ---- | ------------- | ------- |
| `gr00t_n1` | `calvin_abcd` | | |
| `gr00t_n1_5` | `calvin_abcd` | | |
| `pi0` | `calvin_abcd` | | |
| `dp_clip` | `calvin_abcd` | | |
| `act_clip` | `calvin_abcd` | | |

### Simpler-Env Benchmark
| Model  | Dataset/Benchmark | Success Rate | Model Weights |
| ------------ | ------------- | ------------- | ------- |
| `gr00t_n1` | `google_robot` | | |
| `gr00t_n1_5` | `google_robot` | | |
| `pi0` | `google_robot` | | |
| `dp_clip` | `google_robot` | | |
| `act_clip` | `google_robot` | | |
| `gr00t_n1` | `bridgedata_v2` | | |
| `gr00t_n1_5` | `bridgedata_v2` | | |
| `pi0` | `bridgedata_v2` | | |
| `dp_clip` | `bridgedata_v2` | | |
| `act_clip` | `bridgedata_v2` | | |

### Genmanip Benchmark
| Model  | Dataset/Benchmark | Success Rate | Model Weights |
| ------------ | ------------- | ------------- | ------- |
| `gr00t_n1` | `genmanip_v1` | | |
| `gr00t_n1_5` | `genmanip_v1` | | |
| `pi0` | `genmanip_v1` | | |
| `dp_clip` | `genmanip_v1` | | |
| `act_clip` | `genmanip_v1` | | |



<!---
We have the finetuned weights for the following built-in models that can be used for evaluation:
- `pi0`: A pre-trained model for general-purpose manipulation tasks.
- `gr00t-n1/1.5`: An advanced model with additional capabilities.
- `dp-clip`: A model that combines diffusion policy with CLIP for instruction-guided manipulation.
- `act-clip`: A model that integrates Action-chunking Transformer with CLIP for instruction-guided manipulation.
--->

<!-- ## [Optional] Add a Custom Benchmark

To evaluate models on your own benchmark, you should first implement your benchmark and register it into InternManip, then follow the same command structure as above, replacing the `EvalCfg` with your custom benchmark name and adjusting other parameters as needed. Please refer to [`How to add your benchmark`](../tutorials/evaluation.md) to learn how to prepare your model and benchmark for evaluation. -->

## What's Next?


Now that you’ve completed the training and evaluation process, you may want to incorporate your **own dataset**, **model**, or **benchmark**. To do so, please refer to the following guides:

* 📁 **[How to customize your dataset](../quick_start/add_dataset.md)** – Learn how to prepare and register your dataset for training or evaluation.
* 🧠 **[How to add a model](../quick_start/add_model.md)** – Learn how to integrate your own model into InternManip’s training pipeline.
* 🧪 **[How to add your benchmark](../quick_start/add_benchmark.md)** – Learn how to implement and register a new evaluation benchmark.

Once you’ve set them up, you can follow the same command structures used above—just replace the relevant configuration entries (e.g., `--config`) with your custom definitions.
