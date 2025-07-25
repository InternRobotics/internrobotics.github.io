# 🥇 Add a New Benchmark


This guide walks you through adding a custom agent and custom evaluation benchmark to the InternManip framework.

### 1. Implement Your Model Agent


To support a new model in InternManip, define a subclass of [`BaseAgent`](../../internmanip/agent/base.py). You must implement two core methods:

- `step()`: given an observation, returns an action.
- `reset()`: resets internal states, if needed.

**Example: Define a Custom Agent**
```python
from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg

class MyCustomAgent(BaseAgent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        # Custom model initialization here

    def step(self, obs):
        # Implement forward logic here
        return action

    def reset(self):
        # Optional: reset internal state
        pass
```

**Register Your Agent**

In `internmanip/agent/base.py`, register your agent in the `AgentRegistry`:
```python
class AgentRegistry(Enum):
    ...
    CUSTOM = "MyCustomAgent"

    @property
    def value(self):
        if self.name == "CUSTOM":
            from internmanip.agent.my_custom_agent import MyCustomAgent
            return MyCustomAgent
        ...
 ```

<!---
Define a subclass of [`BaseAgent`](../../internmanip/agent/base.py) to implement two essential methods for model reset and step functionality. An [example](../../internmanip/agent/openvla_agent.py) based on the OpenVLA policy model is provided for reference.--->

### 2. Creating a New Evaluator

To add support for a new evaluation environment, inherit from the `Evaluator` base class and implement required methods:

```python
from internmanip.evaluator.base import Evaluator
from internmanip.configs import EvalCfg

class CustomEvaluator(Evaluator):

    def __init__(self, config: EvalCfg):
        super().__init__(config)
        # Custom initialization logic
        ...

    @classmethod
    def _get_all_episodes_setting_data(cls, episodes_config_path) -> List[Any]:
        """Get all episodes setting data from the given path."""
        ...

    def eval(self):
        """The default entrypoint of the evaluation pipeline."""
        ...
```

### 3. Registering the Evaluator

Register the new evaluator in `EvaluatorRegistry` under `internmanip/evaluator/base.py`:

```python
# In internmanip/evaluator/base.py
class EvaluatorRegistry(Enum):
    ...
    CUSTOM = "CustomEvaluator"  # Add new evaluator

    @property
    def value(self):
        if self.name == "CUSTOM":
            from internmanip.evaluator.custom_evaluator import CustomEvaluator
            return CustomEvaluator
    ...
```

### 4. Creating Configuration Files

Create configuration files for the new evaluator:

```python
# scripts/eval/configs/custom_agent_on_custom_bench.py
from internmanip.configs import *
from pathlib import Path

eval_cfg = EvalCfg(
    eval_type="custom_bench",  # Corresponds to the name registered in EvaluatorRegistry
    agent=AgentCfg(
        agent_type="custom_agent", # Corresponds to the name registered in AgentRegistry
        model_name_or_path="path/to/model",
        model_kwargs={...},
        server_cfg=ServerCfg(  # Optional server configuration
            server_host="localhost",
            server_port=5000,
        ),
    ),
    env=EnvCfg(
        env_type="custom_env", # Corresponds to the name registered in EnvWrapperRegistry
        config_path="path/to/env_config.yaml",
        env_settings=CustomEnvSettings(...)
    ),
    logging_dir="logs/eval/custom",
    distributed_cfg=DistributedCfg( # Optional distributed configuration
        num_workers=4,
        ray_head_ip="auto",  # Use "auto" for local machine
        include_dashboard=True,
        dashboard_port=8265,
    )
)
```

## 5. Launch the Evaluator
```python
python scripts/eval/start_evaluator.py \
  --config scripts/eval/configs/custom_on_custom.py
```
Use `--distributed` for Ray-based multi-GPU, and `--server` for client-server mode.
