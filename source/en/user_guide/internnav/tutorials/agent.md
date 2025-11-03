# Defining Custom Models and Agents in InternNav

This tutorial provides a detailed guide for registering new agent and model within the InternNav framework

---

## Development Overview
The main architecture of the evaluation code adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then make model to predict and response to the client.

The InternNav project adopts a modular design, allowing developers to easily add new navigation algorithms.
The main components include:

- **Model**: Implements the specific neural network architecture and inference logic

- **Agent**: Serves as a wrapper for the Model, handling environment interaction and data preprocessing

- **Config**: Defines configuration parameters for the model and training

## Supported Models
- InternVLA-N1
- CMA (Cross-Modal Attention)
- RDP (Recurrent Diffusion Policy)
- Navid (RSS2023)
- Seq2Seq Policy

## Custom Model
A Model is the concrete implementation of your algorithm. Implement model under `baselines/models`. A model ideally would inherit from the base model and implement the following key methods:

- `forward(train_batch) -> dict(output, loss)`
- `inference(obs_batch, state) -> output_for_agent`

## Create a Custom Config Class

In the model file, define a `Config` class that inherits from `PretrainedConfig`.
A reference implementation is `CMAModelConfig` in [`cma_model.py`](https://github.com/InternRobotics/InternNav/blob/main/internnav/model/cma/cma_policy.py).

## Registration and Integration

In [`internnav/model/__init__.py`](https://github.com/InternRobotics/InternNav/blob/main/internnav/model/__init__.py):
- Add the new model to `get_policy`.
- Add the new model's configuration to `get_config`.

## Create a Custom Agent

The Agent handles interaction with the environment, data preprocessing/postprocessing, and calls the Model for inference.
A custom Agent usually inherits from [`Agent`](https://github.com/InternRobotics/InternNav/blob/main/internnav/agent/base.py) and implements the following key methods:

- `reset()`: Resets the Agent's internal state (e.g., RNN states, action history). Called at the start of each episode.
- `inference(obs)`: Receives environment observations `obs`, performs preprocessing (e.g., tokenizing instructions, padding), calls the model for inference, and returns an action.
- `step(obs)`: The external interface, usually calls `inference`, and can include logging or timing.

Example: [`CMAAgent`](https://github.com/InternRobotics/InternNav/blob/main/internnav/agent/cma_agent.py)

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
    'instruction': str                  # language instruction for the navigation task
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
## Registration
The agent should be registered to internnav.agent, so it can be used by the name through configs.
```
from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

@Agent.register('cma')
class NewAgent(Agent):
    def __init__(self, agent_config: AgentCfg):
        ...
```
Make sure you also import it inside `internnav/agent/__init__.py`
```
# make the register decorator taking effect
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
```

## Agent and Model Initialization

Refer to existing **evaluation** config files for customization:
```
agent_cfg=AgentCfg(
    server_host='localhost',
    server_port=8023,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        policy_name='InternVLAN1_Policy',
        state_encoder=None,
    },
)
```

## Typical Usage Example
```
from internnav.configs.agent import AgentCfg

cfg = AgentCfg(server_host="127.0.0.1", server_port=8087)
client = AgentClient(cfg)

# step once
obs = [{"rgb": ..., "depth": ..., "instruction": "go to kitchen"}]
action = client.step(obs)
print("Predicted action:", action)

# reset agent
client.reset()
```