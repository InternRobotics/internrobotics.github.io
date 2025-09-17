# Create Your Model and Agent

## Development Overview
The main architecture of the evaluation code adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then make model to predict and response to the client.

The InternNav project adopts a modular design, allowing developers to easily add new navigation algorithms.
The main components include:

- **Model**: Implements the specific neural network architecture and inference logic

- **Agent**: Serves as a wrapper for the Model, handling environment interaction and data preprocessing

- **Config**: Defines configuration parameters for the model and training

## Custom Model
A Model is the concrete implementation of your algorithm. Implement model under `baselines/models`. A model ideally would inherit from the base model and implement the following key methods:

- `forward(train_batch) -> dict(output, loss)`
- `inference(obs_batch, state) -> output_for_agent`

## Create a Custom Config Class

In the model file, define a `Config` class that inherits from `PretrainedConfig`.
A reference implementation is `CMAModelConfig` in [`cma_model.py`](../internnav/model/cma/cma_policy.py).

## Registration and Integration

In [`internnav/model/__init__.py`](../internnav/model/__init__.py):
- Add the new model to `get_policy`.
- Add the new model's configuration to `get_config`.

## Create a Custom Agent

The Agent handles interaction with the environment, data preprocessing/postprocessing, and calls the Model for inference.
A custom Agent usually inherits from [`Agent`](../internnav/agent/base.py) and implements the following key methods:

- `reset()`: Resets the Agent's internal state (e.g., RNN states, action history). Called at the start of each episode.
- `inference(obs)`: Receives environment observations `obs`, performs preprocessing (e.g., tokenizing instructions, padding), calls the model for inference, and returns an action.
- `step(obs)`: The external interface, usually calls `inference`, and can include logging or timing.

Example: [`CMAAgent`](../internnav/agent/cma_agent.py)

For each step, the agent should expect an observation from environment.

For the vln benchmark under internutopia:

```
action = self.agent.step(obs)
```
**obs** has format:
```
obs = [{
    'globalgps': [X, Y, Z]              # robot location
    'globalrotation': [X, Y, Z, W]      # robot orientation in quaternion
    'rgb': np.array(256, 256, 3)        # rgb camera image
    'depth': np.array(256, 256, 1)      # depth image
}]
```
**action** has format:
```
action = List[int]                      # action for each environments
# 0: stop
# 1: move forward
# 2: turn left
# 3: turn right
```

## Create a Trainer

The Trainer manages the training loop, including data loading, forward pass, loss calculation, and backpropagation.
A custom trainer usually inherits from the [`Base Trainer`](../internnav/trainer/base.py) and implements:

- `train_epoch()`: Runs one training epoch (batch iteration, forward pass, loss calculation, parameter update).
- `eval_epoch()`: Evaluates the model on the validation set and records metrics.
- `save_checkpoint()`: Saves model weights, optimizer state, and training progress.
- `load_checkpoint()`: Loads pretrained models or resumes training.

Example: [`CMATrainer`](../internnav/trainer/cma_trainer.py) shows how to handle sequence data, compute action loss, and implement imitation learning.

## Training Data

The training data is under `data/vln_pe/traj_data`. Our dataset provides trajectory data collected from the H1 robot as it navigates through the task environment.
Each observation in the trajectory is paired with its corresponding action.

You may also incorporate external datasets to improve model generalization.

## Evaluation Data
In `raw_data/val`, for each task, the model should guide the robot at the start position and rotation to the target position with language instruction.

## Set the Corresponding Configuration

Refer to existing **training** configuration files for customization:

- **CMA Model Config**: [`cma_exp_cfg`](../scripts/train/configs/cma.py)

Configuration files should define:
- `ExpCfg` (experiment config)
- `EvalCfg` (evaluation config)
- `IlCfg` (imitation learning config)

Ensure your configuration is imported and registered in [`__init__.py`](../scripts/train/configs/__init__.py).

Key parameters include:
- `name`: Experiment name
- `model_name`: Must match the name used during model registration
- `batch_size`: Batch size
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `dataset_*_root_dir`: Dataset paths
- `lmdb_features_dir`: Feature storage path

Refer to existing **evaluation** config files for customization:

- **CMA Model Evaluation Config**: [`h1_cma_cfg.py`](../scripts/eval/configs/h1_cma_cfg.py)

Main fields:
- `name`: Evaluation experiment name
- `model_name`: Must match the name used during training
- `ckpt_to_load`: Path to the model checkpoint
- `task`: Define the tasks settings, number of env, scene, robots
- `dataset`: Load r2r or interiornav dataset
- `split`: Dataset split (`val_seen`, `val_unseen`, `test`, etc.)
