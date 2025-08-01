# Model

This tutorial introduces the structure and implementation of both System 1 (NavDP) and whole-system (InternVLA-N1) policy models in the internNav framework.

---

## System 1: NavDP

<!-- NavDP content start -->

This tutorial introduces the structure and implementation of the NavDP policy model in the internNav framework, helping you understand and customize each module.

---

### Model Structure Overview

The NavDP policy model in internNav mainly consists of the following parts:

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
- [diffusion policy](https://github.com/real-stanford/diffusion_policy)

<!-- NavDP content end -->

---

## Dual System: InternVLA-N1
This tutorial provides a detailed guide for training the InternVLA-N1 policy model within the internNav framework.

1. Qwen2.5-VL Backbone
The system 2 model is built on Qwen2.5-VL, a state-of-the-art vision-language model:

```python
class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
```
Qwen2.5-VL supports multi-turn conversations, image understanding, and text generation. We finetune the qwenVL model on the self-collected navigation dataset.

2. Latent Queries
Our model learns a set of latent queries to query the latent vector of Qwen2.5-VL, which is used to model trajectory context.
```python
self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
```

3. NavDP Integration
Embeds the System 1 (NavDP) policy for low-level trajectory generation:

```python
def build_navdp(navdp_cfg):
    navdp = NavDP_Policy_DPT_CriticSum_DAT(navdp_pretrained=navdp_cfg.navdp_pretrained)
    navdp.load_model()
    return navdp
```
NavDP converts high-level waypoints from the language model to continuous action sequences.


### Reference
[Qwen2.5-VL Documentation](https://lmdeploy.readthedocs.io/en/latest/multi_modal/qwen2_5_vl.html)
