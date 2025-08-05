# ✍🏻 Create a New Model


This guide shows you, step by step, how to plug a **new end‑to‑end policy model** into the InternManip framework. Follow the checklist below and you will be able to train your custom model with the stock training script (`scripts/train/train.py`)—no core code edits required.



## File Structure and Why

Currently, leading manipulation models strive to leverage existing pretrained large models for better generalization. For example, **GR00T-N1** and **Pi-0** typically consist of a pretrained VLM backbone and a compact downstream action expert that maps extracted context to the action space. Reflecting this design, InternManip organizes model files into three main components:

- **Backbone**: The pretrained VLM backbone responsible for understanding visual or textual inputs.
- **Action Head**: The downstream expert that consumes backbone features and predicts actions.
- **Policy Model**: The wrapper that integrates the backbone and action head into a single end-to-end policy.

Model definitions reside in the `internmanip/model` directory, which contains three sub-folders:

```text
internmanip
├── model
│   ├── action_head        # task‑specific experts
│   ├── backbone           # pretrained encoders (ViT, CLIP, …)
│   └── basemodel          # full end‑to‑end policies
│       └── base.py        # <‑‑ universal interface
...
└── configs
    └── model              # config classes (inherits PretrainedConfig)
scripts
    └── train              # trainers, entry points
```

## 1. Outline
To integrate a new model into the framework, you need to create the following files:

1. A **Config** that stores architecture related hyper‑parameters.
2. A **Model** class that inherits `BasePolicyModel` and implements the model structure.
3. A **data\_collator** that shapes raw samples into model‑ready tensors.

Finally, you need to **register** the model with the framework and you can start training your model. We will guide you through the process step by step.


## 2. Create the Model Configuration File

Each model in our framework should define its architecture-related hyperparameters in a **configuration file**. 
These configuration classes inherit from `transformers.PretrainedConfig`, which provides serialization, deserialization, and compatibility with HuggingFace’s model loading utilities.

You should place your model’s config file in:
```bash
internmanip/configs/model/{model_name}_cfg.py
```

**🧱 About transformers.PretrainedConfig**

[`PretrainedConfig`](https://huggingface.co/docs/transformers/main_classes/configuration) is the base class for all HuggingFace model configurations. It supports:
- Loading/saving config files via .from_pretrained() and .save_pretrained()
- Managing default values
- Providing shared arguments across training, inference, and serialization


<!-- The config file is used to store the architecture related hyper-parameters. Here is some basic information you need to know:
You shall add the model configuration file in `internmanip/configs/model/{model_name}_cfg.py`, which should inherit `transformers.PretrainedConfig`. -->


The following is **an example** of a model configuration file:

```python
from transformers import PretrainedConfig

class CustomPolicyConfig(PretrainedConfig):
    """Configuration for CustomPolicy."""
    model_type = "custom_model"

    """Model-specific parameters"""
    vit_name = "google/vit-base-patch16-224-in21k"
    freeze_vit = True
    hidden_dim = 256
    output_dim = 8
    dropout = 0.0
    n_obs_steps = 1
    horizon = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def transform(self) -> Tuple[List[Transform], List[int], List[int]]:
        """
        This method defines the input processing logic for the model.
        
        It must return a 3-tuple:
        - `transforms`: A list of preprocessing or augmentation operations applied to raw inputs.
        - `obs_indices`: A list of time step indices used as observation input.
        - `action_indices`: A list of time step indices the model needs to predict (action horizon).

        You can customize `transforms` to include resizing, normalization, cropping, etc.
        """
        transforms = None
        return transforms, list(range(self.n_obs_steps)), list(range(self.horizon))
```
> 🔧 Important: All config classes must implement a transform() method that returns a 3-tuple.

As shown in the example above, the config class defines key architectural hyperparameters—such as the backbone model name, whether to freeze the backbone, the hidden/output dimensions of the action head, and more. You are free to extend this config with any additional parameters required by your custom model.

**🔧 About `transforms`**

Additionally, you can implement a **model-specific `transform` method** within the config class. This method allows you to apply custom data transformations that are ***not*** included in the dataset-specific transform list defined in `internmanip/configs/dataset/data_config.py`.

During training, the script `scripts/train/train.py` will automatically call this method and apply your custom transform alongside the default ones. Your `transform` method should follow the same input/output format as dataset-specific transform. For implementation guidance, refer to examples in the `internmanip/dataset/transform` directory.


## 3. Implement the Model

In this class to implement the model, you need to inherit `BasePolicyModel` and register it with `@BasePolicyModel.register("custom_model")`.

The model configuration file will be passed to the `__init__` method of the model class to initialize the model. With in the `__init__` method, you should define the model structure and initialize the model.

You should also implement the `forward` method to define the model forward pass. The `forward` method should return a dictionary of tensors, which will be used to compute the loss. The `inference` method is used to generate the action from the model.

```python
from internmanip.model.basemodel.base import BasePolicyModel
from transformers import ViTModel, ViTConfig
import torch.nn as nn, torch.nn.functional as F, torch
from typing import Dict
from internmanip.configs.model.custom_policy_cfg import CustomPolicyConfig

class CustomPolicyModel(BasePolicyModel):
    """ViT backbone + 2‑layer MLP head."""

    config_class = CustomPolicyConfig
    name = "custom_model"

    def __init__(
        self, 
        config: Optional[CustomPolicyConfig] = None,
        *args,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        

        # 1 Backbone
        vit_conf = ViTConfig.from_pretrained(config.vit_name)
        self.vit = ViTModel.from_pretrained(config.vit_name, config=vit_conf)
        if config.freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 2 Action Head
        self.mlp = nn.Sequential(
            nn.Linear(vit_conf.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    # —— Training / Inference ——
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        imgs, tgt = batch["images"], batch.get("actions")
        feats = self.vit(imgs).last_hidden_state[:, 0]  # CLS token
        pred = self.mlp(feats)
        out = {"prediction": pred}
        if train and tgt is not None:
            out["loss"] = F.mse_loss(pred, tgt.view_as(pred))
        return out

    def inference(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        actions = self.forward(batch, noise=None, time=None)["prediction"]
        return actions
```

In the example above, the model is composed of a ViT backbone and a simple 2-layer MLP action head. The `forward` method handles loss computation during training, while the `inference` method generates actions during evaluation.

When designing your own model, you can follow this backbone–head pattern or adopt a completely different architecture. If needed, you can define custom `backbone` and `action_head` modules—typically by subclassing `nn.Module`. Just ensure that your model's `inference` output has the shape `(n_actions, action_dim)`.


## 4. Write a Data Collator

You need to define a data_collator function that converts a list of raw samples from default data loader into a single batch dictionary that is compatible with the model's `forward` method.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from internmanip.configs.model.custom_cfg import CustomPolicyConfig

@DataCollatorRegistry.register(CustomPolicyConfig.model_type)
def custom_data_collator(samples):
    imgs = torch.stack([s["image"] for s in samples])
    acts = torch.stack([s["action"] for s in samples])
    return {"images": imgs, "actions": acts}
```

> **Why?** The built‑in `BaseTrainer` accepts any callable named `data_collator` so long as it returns a dictionary of tensors compatible with your model’s `forward` signature.


## 5. Register Everything

Add the following **one-time** registration lines (typically at the end of your model file) to enable seamless dynamic loading with `AutoConfig` and `AutoModel`:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("custom_model", CustomPolicyConfig)
AutoModel.register(CustomPolicyConfig, CustomPolicyModel)
```

Make sure the string `"custom_model"` passed to `AutoConfig.register` matches the model name used in both your `CustomPolicyModel` definition and the data collator registration.

Don't forget to register the module in your `__init__.py`, so that your custom model gets imported and initialized properly during runtime. For example:

```python
# In internmanip/model/basemodel/__init__.py
from internmanip.model.basemodel.base import BasePolicyModel

__all__ = ["BasePolicyModel"]
# Import all model modules to ensure registration logic is executed
from internmanip.model.basemodel.custom import custom_model  # <- Your custom model module
```

Once registered, InternManip’s trainer can instantiate your model and you can start training.

📚 For more details related to training and evaluation, please refer to [train_eval.md](./train_eval.md) and [training.md](../tutorials/training.md).
