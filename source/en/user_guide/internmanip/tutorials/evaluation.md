# Evaluation (WIP)

This document provides the guide of evaluation workflow and custom development.

## Table of Contents
1. [Overview](#overview)
2. [Evaluation Workflow](#evaluation-workflow)
3. [API Specifications](#api-specifications)
4. [Configuration Guide](#configuration-guide)
5. [Distributed Evaluation](#distributed-evaluation)
6. [Examples and Best Practices](#examples-and-best-practices)
7. [Troubleshooting](#troubleshooting)
<!---8. [Development Guide](#development-guide)--->

## Overview

The InterManip Evaluator is a modular robotic manipulation task evaluation system that supports multiple evaluation environments and agent types. The system adopts a plugin-based architecture, making it easy to extend with new evaluation environments and agents.

### Key Features
- **Multi-Environment Support**: Supports SimplerEnv, Calvin, GenManip, and other evaluation environments
- **Distributed Evaluation**: Ray-based distributed evaluation framework supporting multi-GPU parallel evaluation
- **Client-Server Mode**: Supports agents running in server mode to avoid conflicts from software and hardware environments
- **Flexible Configuration**: Pydantic-based configuration system with type checking and validation
- **Extensible Architecture**: Easy to add new evaluators and environments

### System Architecture
```
scripts/eval/start_evaluator.py (Entry Point)
    ↓
Evaluator (Base Class, managing the EnvWrapper and BaseAgent)
    ↓
Specific Evaluators (SimplerEvaluator, CalvinEvaluator, GenManipEvaluator)
    ↓
Specific Environments (SimplerEnv, CalvinEnv, GenmanipEnv) + Specific Agents (Pi0Agent, Gr00t_N1_Agent, DPAgent, ...)
```

### Supported benchmarks source codebase
- **SimplerEnv**: internmanip/benchmarks/SimplerEnv
- **Calvin**: internmanip/benchmarks/calvin
- **GenManip**: internmanip/benchmarks/genmanip

## Evaluation Workflow

### 1. Starting Evaluation

The evaluation system is launched through `start_evaluator.py`:

#### 🖥 Terminal 1: Launch the Policy Server (Model Side)

Activate the environment for the model and start the policy server:
```bash
# Client-server mode
source .venv/model/bin/activate
python scripts/eval/start_agent_server.py
```
This server listens for observation inputs from the environment and responds with action predictions from the model.
#### 🖥 Terminal 2: Launch the Evaluator (Environment Side)
```bash
source .venv/simpler_env/bin/activate
python scripts/eval/start_evaluator.py --config run_configs/eval/pi0_on_genmanip.py --server
```

Alternative usage scenarios apart from the client-server model:

```bash
# Basic evaluation
python scripts/eval/start_evaluator.py --config <config>

# Distributed evaluation
python scripts/eval/start_evaluator.py --config <config> --distributed

# Combined modes
python scripts/eval/start_evaluator.py --config <config> --distributed --server
```

Each evaluator follows this workflow:

1. **Initialization**: Load configuration, initialize environment and agent
2. **Data Loading**: Load evaluation tasks from configuration files or data sources
3. **Task Execution**: Execute evaluation tasks sequentially or in parallel
4. **Result Collection**: Collect execution results for each task
5. **Statistical Analysis**: Calculate success rate, or other metrics
6. **Result Persistence**: Save results to specified directory with timestamps


## API Specifications

### 1. Evaluator Base Class Interface

#### Required Methods

```python
class Evaluator:
    @classmethod
    def _get_all_episodes_setting_data(cls, episodes_config_path) -> List[Any]:
        """
        Load all evaluation task configurations

        Args:
            episodes_config_path: Path to evaluation task configuration file(s)

        Returns:
            List[Any]: List of evaluation task configurations
        """
        raise NotImplementedError

    def eval(self):
        """
        Main evaluation method
        """
        raise NotImplementedError
```

#### Optional Methods

```python
def __init__(self, config: EvalCfg):
    """
    Initialize evaluator

    Args:
        config: Evaluation configuration
    """
    super().__init__(config)
    # Custom initialization logic

    @classmethod
    def _update_results(cls, result):
        """
        Update evaluation results

        Args:
            result: Single task evaluation result
        """
        raise NotImplementedError

    @classmethod
    def _print_and_save_results(cls):
        """
        Print and save evaluation results
        """
        raise NotImplementedError

    def _eval_single_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single evaluation task

        Args:
            episode_data: Configuration data for a single task

        Returns:
            Dict[str, Any]: Task execution result
        """
        raise NotImplementedError
```

### 2. Configuration Interface Specification

#### EvalCfg Configuration Structure

```python
class EvalCfg(BaseModel):
    eval_type: str                                    # Evaluator type
    agent: AgentCfg                                   # Agent configuration
    env: EnvCfg                                       # Environment configuration
    logging_dir: Optional[str] = None                 # Logging directory
    distributed_cfg: Optional[DistributedCfg] = None  # Distributed configuration
```

## Configuration Guide

### 1. Basic Configuration

```python
eval_cfg = EvalCfg(
    eval_type="simpler",             # Evaluator type
    agent=AgentCfg(...),             # Agent configuration
    env=EnvCfg(...),                 # Environment configuration
    logging_dir="logs/eval/simpler", # Logging directory
    distributed_cfg=DistributedCfg(...),
)
```

### 2. Agent Configuration

```python
agent=AgentCfg(
    agent_type="gr00t_n1",                # Agent type
    base_model_path="path/to/model",      # Model path
    agent_settings={                      # Model parameters
        'policy_setup': 'bridgedata_v2',
        'action_scale': 1.0,
        'exec_horizon': 1,
        'action_ensemble_temp': -0.8,
        'embodiment_tag': 'new_embodiment',
        'denoising_steps': 16,
        # ...
    },
    model_kwargs={
        'HF_cache_dir': None,
    },
    server_cfg=ServerCfg(                 # Server configuration (optional)
        server_host="localhost",
        server_port=5000,
        timeout=30,
    ),
)
```

### 3. Environment Configuration

```python
env=EnvCfg(
    env_type="simpler",                    # Environment type
    device_id=0,                           # Device ID
    episodes_config_path=[...]
)
```

### 4. Distributed Configuration

```python
distributed_cfg=DistributedCfg(
    num_workers=4,                        # Number of worker processes
    ray_head_ip="10.150.91.18",          # Ray cluster head node IP
    include_dashboard=True,               # Include dashboard
    dashboard_port=8265,                  # Dashboard port
)
```

## [Optional] Distributed Evaluation

### Ray Cluster Setup

Distributed evaluation is based on the Ray framework, supporting the following deployment modes:

- **Single Machine Multi-Process**: `ray_head_ip="auto"`
- **Multi-Machine Cluster**: `ray_head_ip="10.150.91.18"`

<!---
### 2. Workflow
[WIP]--->


## [Optional] Use Multiple Evaluators to Speed Up
If you have sufficient resources, we also provide multi-process parallelization to speed up the evaluation. This feature is enabled by the [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) distributed framework, so it requires starting up a Ray cluster as the distributed backend.

- Start up a Ray cluster on your machine(s):
```bash
# Call `ray start --head` on your main machine. It will start a Ray cluster which includes all available CPUs/GPUs/memory by default.
ray start --head [--include-dashboard=true] [--num-gpus=?]
```

- [Optional] Scale up your Ray cluster:
```bash
# When the Ray cluster is ready, it will print the Ray cluster head IP address on the terminal.
# If you have more than one machine and want to scale up your Ray cluster, execute the following command on the other machines:
ray start --address='{your_ray_cluster_head_ip}:6379'
```

- Customize your [`DistributedCfg`](../../internmanip/configs/evaluator/distributed_cfg.py):
```python
# Example configuration, `DistributedCfg` should be defined in the `EvalCfg`
from internmanip.configs import *

eval_cfg = EvalCfg(
    eval_type="calvin",
    agent=AgentCfg(
        ...
    ),
    env=EnvCfg(
        ...
    ),
    distributed_cfg=DistributedCfg(
        num_workers=4, # Usually equals to the number of GPUs
        ray_head_ip="10.150.91.18", # or "auto" if you are located at the Ray head node machine
        include_dashboard=True, # By default
        dashboard_port=8265, # By default
    )
)
```

- Enable distributed evaluation mode before starting the evaluator pipeline:
```bash
python scripts/eval/start_evaluator.py --config scripts/eval/configs/seer_on_calvin.py --distributed
```

- [Optional] View the task progress or resource monitor:

The Ray framework provides a dashboard to view its task scheduling progress and resource usage. Access it on this address `{ray_head_ip}:8265`.
