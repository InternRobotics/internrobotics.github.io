# Model

This tutorial introduces the structure and implementation of both System 1 (navdp) and System 2 (rdp) policy models in the InterNav framework.

---

## System 1: Navdp

<!-- navdp content start -->

This tutorial introduces the structure and implementation of the navdp policy model in the InterNav framework, helping you understand and customize each module.

---

### Model Structure Overview

The navdp policy model in InterNav mainly consists of the following parts:

- **RGBD Encoder (NavDP_RGBD_Backbone)**: Extracts multi-frame RGB+Depth features.
- **Goal Point/Image Encoder**: Encodes goal point or goal image information.
- **Transformer Decoder**: Temporal modeling and action generation.
- **Action Head / Value Head**: Outputs action sequences and value estimation.
- **Diffusion Scheduler**: For action generation via diffusion process.

The model entry is `NavDPNet` (in `internnav/model/basemodel/navdp/navdp_policy.py`), which inherits from `transformers.PreTrainedModel` and supports HuggingFace-style loading and fine-tuning.

---

### Main Module Explanation

#### 1. RGBD Encoder

Located in `internnav/model/encoder/navdp_backbone.py`:

```python
class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self, image_size=224, embed_size=512, ...):
        ...
    def forward(self, images, depths):
        # Input: [B, T, H, W, 3], [B, T, H, W, 1]
        # Output: [B, memory_size*16, token_dim]
        ...
```

- Supports multi-frame historical image and depth input, outputs temporal features.
- Optional finetune.

#### 2. Goal Point/Image Encoder

- Goal point encoding: `nn.Linear(3, token_dim)`
- Goal image encoding: `NavDP_ImageGoal_Backbone` / `NavDP_PixelGoal_Backbone`

#### 3. Transformer Decoder

```python
self.decoder_layer = nn.TransformerDecoderLayer(d_model=token_dim, nhead=heads, ...)
self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=temporal_depth)
```

- Responsible for temporal modeling and conditional action generation.

#### 4. Action Head and Value Head

```python
self.action_head = nn.Linear(token_dim, 3)   # Output action
self.critic_head = nn.Linear(token_dim, 1)   # Output value
```

#### 5. Diffusion Scheduler

```python
self.noise_scheduler = DDPMScheduler(num_train_timesteps=10, ...)
```

- Used for diffusion-based action generation and denoising.

---

### Example Forward Process

The core forward process is as follows:

```python
def forward(self, goal_point, goal_image, input_images, input_depths, output_actions, augment_actions):
    # 1. Encode historical RGBD
    rgbd_embed = self.rgbd_encoder(input_images, input_depths)
    # 2. Encode goal point/image
    pointgoal_embed = self.point_encoder(goal_point).unsqueeze(1)
    # 3. Noise sampling and diffusion
    noise, time_embeds, noisy_action_embed = self.sample_noise(output_actions)
    # 4. Conditional decoding to generate actions
    cond_embedding = ...
    action_embeddings = ...
    output = self.decoder(tgt=action_embeddings, memory=cond_embedding, tgt_mask=self.tgt_mask)
    # 5. Output action and value
    action_pred = self.action_head(output)
    value_pred = self.critic_head(output.mean(dim=1))
    return action_pred, value_pred
```

---

### Key Code Snippets

#### Load Model

```python
from internnav.model.basemodel.navdp.navdp_policy import NavDPNet, NavDPModelConfig
model = NavDPNet(NavDPModelConfig(model_cfg=...))
```

#### Customization and Extension

To customize the backbone, decoder, or heads, refer to `navdp_policy.py` and `navdp_backbone.py`, implement your own modules, and replace them in the configuration.

---

### Reference
- [navdp_policy.py](../../internnav/model/basemodel/navdp/navdp_policy.py)
- [navdp_backbone.py](../../internnav/model/encoder/navdp_backbone.py)
- [navdp.py config](../../scripts/train/configs/navdp.py)

<!-- navdp content end -->

---

## System 2: InternVLA-N1-S2

*TODO
