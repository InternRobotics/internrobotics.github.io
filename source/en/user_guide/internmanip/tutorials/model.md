# Model (WIP)

This document provides a overview of the modular model structure used in our robotic manipulation framework. It explains each component’s purpose and configuration, enabling users to:

- [**✍️ Make customizations:**](#contents) Adjust parameters such as select_layer, tune_llm, or change the action horizon.
- [**🆕 Add custom models:**](#add-a-custom-model) Add new backbones, action heads, or policy models by following the provided templates.


**Overview:**
- **Backbone**: Processes camera observations and natural language instructions into contextual embeddings.
- **Action Head**: Converts backbone features into executable robot actions, often via diffusion-based generative modeling.
- **Policy Model**: Orchestrates the backbone and action head, managing data flow and inference to produce final robot commands.

> Minor customizations or full replacements are supported. For developing new policy models that combine or extend these modules, follow the guidance below.


<!-- ## Model Structure

We provide a modular model structure that consists of the following components:

- **VLM Backbone**: A vision-language model backbone that processes visual and language inputs.
- **Action Head**: A head that generates actions based on the processed features from the preceding layers.
- **Policy Model**: A policy model that aggregates the above modules and drives the robot to perform tasks. -->


## Backbone

We provide some kinds of the backbone: `EagleBackbone`, `EagleBackbone1_5`, `DiffusionRgbEncoder`, `PaliGemmaWithExpertModel`, and `ACTBackbone`, .



### EagleBackbone

The Eagle Backbone serves as the foundational component that processes multimodal inputs (vision and language) and generates unified feature representations suitable for downstream tasks.

It has several steps to process the input:

1. **Input Preparation**: Converts raw inputs into model-compatible format
2. **Feature Extraction**: Processes visual and textual inputs separately
3. **Multimodal Fusion**: Combines visual and textual features using attention mechanisms
4. **Output Generation**: Produces unified feature representations

#### Key Features

- Configurable layer selection for LLM truncation
- Visual and language model fine-tuning control
- Built-in image preprocessing with EagleProcessor
- Support for vision reprojection and resolution scaling

#### Usage

##### Basic Initialization

```python
from internmanip.model.backbone.eagle_backbone import EagleBackbone

backbone = EagleBackbone(
    select_layer=12,          # Use first 12 LLM layers
    tune_llm=False,           # Freeze LLM parameters
    tune_visual=True,         # Fine-tune vision model
    processor_cfg={
        "model_path": "/path/to/eagle/model",
        "max_input_tiles": 4,
        "model_spec": {...}
    }
)
```

##### Forward Pass

```python
batch_features = backbone.prepare_input(batch_dict)
output = backbone(batch_features)
# Returns: {"backbone_features": embeddings, "backbone_attention_mask": mask}
```

### EagleBackbone1_5

A simplified version designed for Eagle 1.5 models with streamlined configuration.

#### Key Features

- Simplified initialization with fewer parameters
- Automatic parameter tracking and warnings
- Prefix-based input filtering for multi-modal inputs
- Fixed 2048 → 1536 dimension projection

#### Usage

##### Basic Initialization

```python
from internmanip.model.backbone.eagle_backbone import EagleBackbone1_5

# Basic usage
backbone = EagleBackbone1_5(
    select_layer=16,          # Use first 16 LLM layers
    tune_llm=True,           # Fine-tune LLM
    tune_visual=False,       # Freeze vision model
    project_to_dim=1536      # Output dimension
)
```

##### Forward Pass

```python
# Forward pass expects eagle_ prefixed inputs
vl_input = {
    "eagle_input_ids": tensor,
    "eagle_attention_mask": tensor,
    "eagle_pixel_values": tensor,
    "eagle_image_sizes": tensor
}
output = backbone(vl_input)
```

### DiffusionRgbEncoder

The Diffusion Vision Backbone is designed specifically for diffusion policy implementations. It provides efficient visual feature extraction optimized for robotic manipulation tasks.

It has several steps to process the input:

1. **Image Preprocessing**: Optional cropping and normalization
2. **Backbone Feature Extraction**: Pretrained CNN feature extraction
3. **Spatial Softmax Pooling**: Converts feature maps to spatial keypoints
4. **Final Linear Projection**: Maps keypoints to output feature dimensions

#### Key Features

- Automatically adapts to different input image dimensions
- Leverages torchvision pretrained models
- Generates focused attention points for manipulation tasks

#### Usage

##### Basic Initialization

```python
from internmanip.configs.model.dp_cfg import DiffusionConfig
from internmanip.model.backbone.diffusion_vision_backbone import DiffusionRgbEncoder

config = DiffusionConfig(
    vision_backbone='resnet50',
    pretrained_backbone_weights=None,  # Train from scratch
    crop_shape=(256, 256),
    crop_is_random=True,
    spatial_softmax_num_keypoints=64,
    use_group_norm=True,  # Replace BatchNorm with GroupNorm
    image_features=(3, 480, 640)  # Original image dimensions
)

encoder = DiffusionRgbEncoder(config)
```

##### Forward Pass

```python
images = torch.randn(8, 3, 480, 640)
features = encoder(images)
# Output shape: [8, 64] where 64 = spatial_softmax_num_keypoints * 2
```

### PaliGemmaWithExpertModel

A hybrid backbone combining PaliGemma and Gemma Expert for enhanced multimodal reasoning. This model merges visual grounding with language modeling using shared and expert-specific transformer layers.


#### Key Features

1. Dual Transformer Architecture: Combines `PaliGemma` (VLM) and `Gemma Expert` (LLM).
2. Expert-tuned Attention Layers: Joint attention computation across base and expert models.
3. Vision-Language Embedding Support: Includes `embed_image()` and `embed_language_tokens()` interfaces.
4. Custom bfloat16 Precision Control: Converts selected submodules to bfloat16 with compatibility for physical intelligence models.
5. Selective Freezing: Fine-grained control over training of visual encoder and/or base model.

#### Usage

##### Basic Initialization

```python
from internmanip.model.paligemma_with_expert import PaliGemmaWithExpertModel
from internmanip.configs.paligemma_with_expert_config import PaliGemmaWithExpertConfig

config = PaliGemmaWithExpertConfig(
    paligemma_config=...,
    gemma_expert_config=...,
    freeze_vision_encoder=True,
    train_expert_only=False,
    attention_implementation='eager',  # or 'fa2', 'flex'
)

model = PaliGemmaWithExpertModel(config)
```

##### Forward Pass
```python
# input_embeds: List of tensors from vision and language encoders
outputs, cache = model(
    attention_mask=...,
    position_ids=...,
    past_key_values=...,  # optional for autoregressive decoding
    inputs_embeds=[vision_embeds, language_embeds],
    use_cache=True,
    fill_kv_cache=False,
)
```


### ACTBackbone

The ACT (Action Chunking Transformer) Backbone is designed specifically for the ACT framework, which processes multimodal inputs including robot states, environment states, and visual observations to generate contextual embeddings for action prediction.

It has several steps to process the input:

1. **Image Feature Extraction**: Uses pretrained CNN backbones (e.g., ResNet) to extract visual features from camera observations
2. **State Processing**: Processes robot state and environment state through linear projections
3. **Latent Encoding**: Handles latent space encoding for VAE-based training (optional)
4. **Transformer Encoding**: Uses a transformer encoder to fuse all features into unified representations
5. **Positional Embedding**: Applies 2D sinusoidal positional embeddings for visual features and learned embeddings for state features

#### Key Features

- **Modular Design**: Separates image processing, state processing, and transformer encoding
- **Multi-modal Fusion**: Combines visual, state, and latent features through attention mechanisms
- **Flexible Input Handling**: Supports various input tensor shapes and dimensions
- **VAE Integration**: Optional variational autoencoder for latent space modeling
- **Configurable Architecture**: Adjustable transformer layers, attention heads, and feature dimensions

#### Architecture Components

- **Image Backbone**: Pretrained CNN (ResNet variants) for visual feature extraction
- **State Projections**: Linear layers for robot and environment state processing
- **Transformer Encoder**: Multi-layer transformer for feature fusion
- **Positional Embeddings**: 2D sinusoidal embeddings for visual features, learned embeddings for states

#### Usage

##### Basic Initialization

```python
from internmanip.model.backbone.act_backbone import ACTBackbone
from internmanip.model.basemodel.act_detr.configuration_act import ACTConfig

config = ACTConfig()
config.dim_model = 256
config.latent_dim = 64
config.n_encoder_layers = 4
config.n_heads = 8
config.dim_feedforward = 1024
config.dropout = 0.1
config.pre_norm = True
config.feedforward_activation = "relu"
config.use_vae = False

# Set up input/output features
from internmanip.model.types import PolicyFeature, FeatureType
input_features = {
    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    'observation.environment_state': PolicyFeature(type=FeatureType.ENV, shape=(10,))
}
output_features = {
    'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,))
}
config.set_input_output_features(input_features, output_features)

backbone = ACTBackbone(config)
```

##### Forward Pass

```python
batch = {
    "observation.state": torch.randn(batch_size, 8),
    "observation.environment_state": torch.randn(batch_size, 10),
    "observation.images": [torch.randn(batch_size, 3, 224, 224)],  # Optional
    "action": torch.randn(batch_size, 16, 8),
    "action_is_pad": torch.zeros(batch_size, 16, dtype=torch.bool)
}

latent_sample = torch.randn(batch_size, config.latent_dim)
encoder_out = backbone.encode_features(batch, latent_sample)
# Returns: (seq_len, batch_size, dim_model) tensor
```

## Action Head

Action Head is responsible for converting the high-level contextual features from the VLM backbone into executable robot actions. It uses a flow-matching based approach for robust action generation.

We provides some kinds of action head: `FlowmatchingActionHead`, `FlowmatchingActionHead_1_5`, `DiffusionActionHead`, `PI0FlowMatching` and `ACTActionHead`.



### FlowmatchingActionHead

The main flow matching action head that uses a DiT (Diffusion Transformer) model to generate robot actions through denoising diffusion.

#### Key Features

- Flow matching training with Beta distribution noise scheduling
- Multi-embodiment support with category-specific encoders/decoders
- Sinusoidal timestep encoding for diffusion process
- Cross-attention between state-action embeddings and vision-language features
- Configurable projector and diffusion model fine-tuning

#### Usage

##### Basic Initialization

```python
from internmanip.model.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig

config = FlowmatchingActionHeadConfig(
    action_dim=7,                    # Robot action dimension
    action_horizon=16,               # Action sequence length
    max_num_embodiments=32,          # Number of robot types
    num_inference_timesteps=10,      # Denoising steps
    tune_projector=True,             # Fine-tune encoders/decoders
    tune_diffusion_model=True,       # Fine-tune DiT model
    diffusion_model_cfg={...}        # DiT configuration
)

action_head = FlowmatchingActionHead(config)
```

##### Forward Pass

```python
backbone_output = {"backbone_features": vl_embeddings, "backbone_attention_mask": mask}
action_input = {"action": actions, "state": states, "embodiment_id": robot_ids, "action_mask": mask}
output = action_head(backbone_output, action_input)
# Returns: {"loss": mse_loss, "predictions": predicted_velocity}
```

##### Inference

```python
with torch.no_grad():
    result = action_head.inference(backbone_output, action_input)
    # Returns: {"action_pred": generated_actions}
```

### FlowmatchingActionHead_1_5

Enhanced version with additional backbone processing capabilities and batch expansion support.

#### Key Features

- Vision-language layer normalization and self-attention processing
- Batch expansion for data augmentation during training
- Improved backbone feature processing pipeline

#### Usage

##### Basic Initialization

```python
from internmanip.model.action_head.flow_matching_action_head import FlowmatchingActionHead_1_5, FlowmatchingActionHeadConfig_1_5

# Configuration with additional features
config = FlowmatchingActionHeadConfig_1_5(
    action_dim=7,
    action_horizon=16,
    use_vlln=True,                   # Use vision-language layer norm
    expand_batch=2,                  # Expand batch by 2x for training
    vl_self_attention_cfg={...},     # Self-attention configuration
    **standard_config_params
)

# Initialize and use similarly to base version
action_head = FlowmatchingActionHead_1_5(config)
```

##### Forward pass

```python
output = action_head(backbone_output, action_input)
# Returns: {"loss": mse_loss, "predictions": predicted_velocity}
```

##### Inference

```python
with torch.no_grad():
    result = action_head.get_action(backbone_output, action_input)
    # Returns: {"action_pred": generated_actions}
```

### DiffusionActionHead

The `DiffusionActionHead` class uses a 1D convolutional UNet to generate robot actions through denoising diffusion, following the Diffusion Policy framework.

#### Key Features

- DDPM/DDIM noise scheduling with configurable timesteps and beta parameters
- Multi-modal conditioning supporting vision, state, and language inputs
- 1D Convolutional UNet with FiLM modulation for temporal action sequence generation
- Language conditioning via CLIP embeddings with trainable projection layers
- Flexible observation encoding with support for multiple camera views and separate encoders

#### Usage

##### Basic Initialization

```python
from internmanip.model.action_head.diffusion_action_head import DiffusionActionHead
from internmanip.configs.model.dp_cfg import DiffusionConfig

config = DiffusionConfig(
    action_dim=7,                           # Robot action dimension
    horizon=16,                             # Action sequence length
    n_obs_steps=2,                          # Number of observation steps
    n_action_steps=8,                       # Number of action steps to execute
    robot_state_dim=14,                     # Robot state dimension
    num_train_timesteps=100,                # Training denoising timesteps
    num_inference_steps=10,                 # Inference denoising steps
    use_language_conditioning=True,         # Enable language conditioning
    language_model_name="openai/clip-vit-base-patch32",
    noise_scheduler_type="DDPM"             # or "DDIM"
)

action_head = DiffusionActionHead(config)
```

##### Training

```python
batch = {
    "observation.state": robot_states,      # (B, n_obs_steps, state_dim)
    "observation.images": rgb_images,       # (B, n_obs_steps, num_cameras, C, H, W)
    "language": language_instructions,      # List[str] or str
    "action": target_actions,               # (B, horizon, action_dim)
    "action_is_pad": padding_mask           # (B, horizon)
}

loss_dict = action_head.compute_loss(batch)
loss = loss_dict["loss"]
loss.backward()
```

##### Inference

```python
with torch.no_grad():
    all_actions = action_head.generate_all_actions(batch)
    # Returns: (B, horizon, action_dim)
    executable_actions = action_head.generate_actions(batch)
    # Returns: (B, n_action_steps, action_dim)
```

### PI0FlowMatching

This model enables end-to-end multimodal action generation through diffusion-style flow matching, using PaliGemma for perception and Gemma Expert for reasoning.

#### Key Features

- Multi-Stage Embedding: Separately processes prefix (image & language) and suffix (robot state, actions, timestep).
- Diffusion-Based Control: Implements action prediction via learned denoising steps over time.
- Dual-Transformer Backbone: Combines PaliGemma with an Expert Gemma head for expert-guided decoding.
- Pretrained Vision-Language Model: Leverages the power of PaliGemma for grounded understanding.
- Fine-Grained Attention Control: Per-token control over attention and padding across all modalities.

#### Usage
##### Training Forward Pass
```python
loss = policy(
    images,        # List[Tensor]: shape [B, C, H, W]
    img_masks,     # List[Tensor]: shape [B] (bool)
    lang_tokens,   # Tensor: [B, T]
    lang_masks,    # Tensor: [B, T] (bool)
    state,         # Tensor: [B, T, state_dim]
    actions,       # Tensor: [B, T, action_dim]
)
# Output: [B, T, action_dim], per-step loss
```

##### Action Sampling
```python
actions = policy.sample_actions(
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state
)
# Output: [B, T, action_dim], denoised predicted actions
```
Internally runs an Euler integrator over diffusion steps to progressively refine the action trajectory.

### ACTActionHead

The `ACTActionHead` is designed specifically for the ACT framework, responsible for converting encoded features from the ACT backbone into executable robot actions through transformer-based decoding and optional VAE encoding.

It has several steps to process the input:

1. **Latent Encoding**: Optional VAE encoder for latent space modeling during training
2. **Transformer Decoding**: Uses a transformer decoder to generate action sequences
3. **Action Prediction**: Final linear projection to output robot actions
4. **Positional Embedding**: Learned positional embeddings for action sequence generation

#### Key Features

- **VAE Integration**: Optional variational autoencoder for latent space modeling
- **Transformer Decoder**: Multi-layer transformer decoder for action sequence generation
- **Action Chunking**: Generates action sequences of configurable length
- **Flexible Conditioning**: Supports various input modalities (state, image, language)
- **Temporal Ensembling**: Optional temporal ensemble for improved action consistency

#### Architecture Components

- **VAE Encoder**: BERT-style encoder for latent space modeling (optional)
- **Transformer Decoder**: Multi-layer decoder with cross-attention to encoder features
- **Action Head**: Linear projection layer for final action prediction
- **Positional Embeddings**: Learned embeddings for action sequence positions

#### Usage

##### Basic Initialization

```python
from internmanip.model.action_head.act_action_head import ACTActionHead
from internmanip.model.basemodel.act_detr.configuration_act import ACTConfig

config = ACTConfig()
config.dim_model = 256
config.latent_dim = 64
config.chunk_size = 16
config.n_decoder_layers = 4
config.n_heads = 8
config.dim_feedforward = 1024
config.dropout = 0.1
config.pre_norm = True
config.feedforward_activation = "relu"
config.use_vae = False

# Set up input/output features
from internmanip.model.types import PolicyFeature, FeatureType
input_features = {
    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    'observation.environment_state': PolicyFeature(type=FeatureType.ENV, shape=(10,))
}
output_features = {
    'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,))
}
config.set_input_output_features(input_features, output_features)

action_head = ACTActionHead(config)
```

##### Forward Pass

```python
# For training with VAE
batch = {
    "observation.state": torch.randn(batch_size, 8),
    "observation.environment_state": torch.randn(batch_size, 10),
    "action": torch.randn(batch_size, 16, 8),
    "action_is_pad": torch.zeros(batch_size, 16, dtype=torch.bool)
}

# Encode latent (if using VAE)
latent_sample, mu, log_sigma_x2 = action_head.encode_latent(batch)

# Decode actions
encoder_out = torch.randn(seq_len, batch_size, config.dim_model)
actions = action_head.decode_actions(encoder_out, batch_size, device)
# Returns: (batch_size, chunk_size, action_dim) tensor
```

##### Inference

```python
with torch.no_grad():
    # For inference without VAE
    batch = {
        "observation.state": torch.randn(batch_size, 8),
        "observation.environment_state": torch.randn(batch_size, 10)
    }

    latent_sample, _, _ = action_head.encode_latent(batch)
    # latent_sample will be zeros when not using VAE

    encoder_out = backbone.encode_features(batch, latent_sample)
    actions = action_head.decode_actions(encoder_out, batch_size, device)
    # Returns: (batch_size, chunk_size, action_dim) tensor
```

<!-- ## ACT Model Architecture Analysis

The ACT (Action Chunking Transformer) framework introduces a transformer-based approach to robotic manipulation, distinct from diffusion-based methods. Here's a detailed analysis of its architecture and principles:

### Architecture Overview

The ACT framework consists of three main components working together:

1. **ACTBackbone**: Processes multimodal inputs and generates contextual embeddings
2. **ACTActionHead**: Converts embeddings into action sequences through transformer decoding
3. **ACTPolicy**: Orchestrates the complete pipeline

### Core Principles

#### 1. Action Chunking
- **Concept**: Instead of predicting single actions, ACT generates action sequences (chunks) of configurable length
- **Benefits**:
  - Reduces inference frequency, improving real-time performance
  - Captures temporal dependencies in action sequences
  - Enables more sophisticated action planning

#### 2. Transformer-Based Processing
- **Encoder-Decoder Architecture**: Uses transformer encoder for feature fusion and decoder for action generation
- **Cross-Attention**: Decoder attends to encoder features, enabling rich context utilization
- **Self-Attention**: Captures dependencies within action sequences and input features

#### 3. Multimodal Fusion
- **Input Types**: Robot states, environment states, visual observations
- **Fusion Strategy**: Linear projections followed by transformer encoding
- **Positional Embeddings**: Different strategies for different input types (learned for states, sinusoidal for images)

#### 4. Optional VAE Integration
- **Purpose**: Provides latent space modeling for improved action generation
- **Training**: Uses variational autoencoder with KL divergence loss
- **Inference**: Can operate without VAE for simpler deployment

### Technical Details

#### ACTBackbone Processing Pipeline

```
Input Processing:
├── Image Features: CNN backbone → 2D positional embeddings → spatial averaging
├── State Features: Linear projection → learned positional embeddings
└── Latent Features: Linear projection → learned positional embeddings

Feature Fusion:
├── Token Stacking: Combine all features into sequence
├── Transformer Encoding: Multi-layer self-attention
└── Output: (seq_len, batch_size, dim_model)
```

#### ACTActionHead Processing Pipeline

```
Latent Encoding (Optional):
├── VAE Encoder: BERT-style encoder for action sequences
├── Latent Sampling: Reparameterization trick for training
└── Output: Latent representation

Action Decoding:
├── Transformer Decoder: Cross-attention to encoder features
├── Positional Embeddings: Learned embeddings for action positions
├── Action Head: Linear projection to action space
└── Output: (batch_size, chunk_size, action_dim)
```

### Key Innovations

1. **Temporal Ensembling**: Optional ensemble mechanism that weights actions based on temporal position
2. **Flexible Input Handling**: Supports various tensor shapes and dimensions automatically
3. **Modular Design**: Separates backbone and action head for easy customization
4. **Configurable Architecture**: Adjustable transformer layers, attention heads, and feature dimensions

### Comparison with Other Approaches

| Aspect | ACT | Diffusion Policy | Flow Matching |
|--------|-----|------------------|---------------|
| **Architecture** | Transformer | UNet | DiT |
| **Action Generation** | Direct prediction | Denoising diffusion | Flow matching |
| **Temporal Modeling** | Action chunking | Sequential denoising | Continuous flow |
| **Latent Space** | Optional VAE | Noise scheduling | Beta distribution |
| **Inference Speed** | Fast (direct) | Slower (iterative) | Medium (iterative) |

### Use Cases

- **Real-time Control**: ACT's direct prediction enables faster inference
- **Complex Manipulation**: Action chunking captures sophisticated action sequences
- **Multimodal Tasks**: Robust handling of vision, state, and language inputs
- **Research Applications**: Modular design facilitates experimentation -->

## Add a Custom Model

The Policy Model serves as the top-level orchestrator that integrates VLM backbone and action head components to create end-to-end robotic manipulation policies. It manages the complete pipeline from multimodal observations to executable robot actions.

You need to implement a policy model to drive the robot.

Create a new Python file in `internmanip/model/basemodel/` directory. Here is a template for a custom policy model:

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from internmanip.dataset.transform.base import BatchFeature
from internmanip.model.basemodel.base import BasePolicyModel


class CustomPolicyConfig(BaseModel):
    """Configuration for Custom Policy Model."""
    pass


@BasePolicyModel.register("custom_policy")
class CustomPolicy(BasePolicyModel):
    """Custom Policy Model implementation."""

    def __init__(self, config: CustomPolicyConfig):
        super().__init__()
        self.config = config
        pass

    def forward(self, batch: BatchFeature, train: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def inference(self, batch: BatchFeature, **kwargs) -> Dict[str, torch.Tensor]:
        """Inference-specific forward pass."""
        return self.forward(batch, train=False, **kwargs)

    def calc_loss(self, batch: BatchFeature, **kwargs) -> torch.Tensor:
        """Calculate training loss."""
        outputs = self.forward(batch, train=True, **kwargs)
        return outputs["loss"]
```
