# ✍🏻 Create a New Model (WIP, Adjusting according to the new architecture)

This section guides you through the process of adding a new end2end model to the InternManip framework.

## File Structure and Why

Currently, the leading manipulation models try to leverage the existing pretrained large models for better generalization. They (for instance, **GR00T-N1** and **Pi-0**) often consist of a pretrained VLM backbone and a small downstream action expert that maps extracted hidden context to action space. In this way, we organize the model files into three main components:
- **Backbone**: The pretrained VLM backbone, which is responsible for understanding the visual and textual inputs.
- **Action Head**: The downstream action expert that takes the context from the backbone and maps it to the action space.
- **Policy Model**: The base model that integrates the backbone and action head into a single end-to-end model.

Specifically, the model definitions are located in the `internmanip/model` directory, there are three subfolders under this directory:
```plaintext
internmanip
├── model
│   ├── action_head
│   ├── backbone
│   ├── basemodel
│   │   ├── base.py
│   │   ├── ...
│   ├── ...
├── ...
```

To create a new model, you need to implement a new model class derived from the `BasePolicyModel` class in `internmanip/model/basemodel/base.py`. It looks like this:
```python
from transformers import PreTrainedModel

from internmanip.configs.model.model_cfg import ModelCfg

class BasePolicyModel(PreTrainedModel):
    policy_models = {}

    def __init__(self, config: ModelCfg):
        super().__init__(config)
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference method not implemented.")

```
where you need to implement the `__init__`, `forward`, and `inference` methods. The `forward` method is used for training, while the `inference` method is used for inference.

## Implementation Steps
As a quick start, we will use a very simple model with a ViT visual encoder and two layers of MLP as an example.

1. Create a new file for your model in the `internmanip/model/basemodel` directory, for example `custom_model.py`.
2. Import the necessary modules and classes, implement `__init__`, `forward`, and `inference` methods, and register your model class with the `BasePolicyModel` class:
```python
from pydantic import BaseModel
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig   # pip install transformers


from internmanip.model.basemodel.base import BasePolicyModel


class CustomPolicyConfig(BaseModel):
    """Configuration for Custom Policy Model."""
    vit_name: str = "google/vit-base-patch16-224-in21k"  # or any HF ViT
    freeze_vit: bool = True
    input_dim: int
    hidden_dim: int = 256
    output_dim: int
    dropout: float = 0.0


@BasePolicyModel.register("custom_model")
class CustomModel(BasePolicyModel):
    """Two-layer MLP policy."""

    def __init__(self, config: CustomPolicyConfig):
        super().__init__()
        self.config = config

        # 1. ViT visual encoder
        vit_conf = ViTConfig.from_pretrained(config.vit_name)
        self.vit = ViTModel.from_pretrained(config.vit_name, config=vit_conf)
        if config.freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 2. Two-layer MLP head
        vit_out_dim = vit_conf.hidden_size   # 768 for base
        self.mlp = nn.Sequential(
            nn.Linear(vit_out_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )


    def forward(self, batch: Dict[str, torch.Tensor], train: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for both training and inference.
        When train=True we also return the loss.
        """
        images = batch["images"]         # (B, 3, 224, 224)
        vit_out = self.vit(images).last_hidden_state[:, 0] # (B, 768) - CLS token output
        pred = self.mlp(vit_out)

        outputs = {"prediction": pred}

        if train:
            # Assume the batch contains a key named "actions" that holds the GT
            if pred.shape != targets.shape:
                targets = targets.view_as(pred)
            loss = F.mse_loss(pred, targets)
            outputs["loss"] = loss

        return outputs

    def inference(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Inference-specific forward pass (no loss)."""
        return self.forward(batch, train=False, **kwargs)

```
3. Now you can train your just customized model on `genmanip-demo` dataset with the following command:
```bash
torchrun --nnodes 1 --nproc_per_node 1 \       # number of processes per node, e.g., 1
   scripts/train/train.py \
   --model_name custom_model \     # model name
   --dataset-path genmanip-demo \  # registered dataset name or custom path
   --data-config genmanip-v1       # registered data config
```

For more advanced tutorials, please refer to the [Model](../tutorials/model.md) section.
