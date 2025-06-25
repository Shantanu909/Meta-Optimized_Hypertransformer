import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os


# -- Gated Residual Block --
class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable residual gate

    def forward(self, x, sublayer_out):
        return x + self.alpha * sublayer_out


# -- Custom Transformer Layer with per-head injection --
class CustomTransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.res1 = GatedResidual(model_dim)
        self.res2 = GatedResidual(model_dim)

    def forward(self, x, external_weights=None):
        if external_weights:
            self.inject_weights(external_weights)

        # Attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(self.res1(x, attn_out))

        # Feedforward + residual
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(self.res2(x, ff_out))
        return x

    def inject_weights(self, weights):
        with torch.no_grad():
            self.attn.in_proj_weight.copy_(weights['attn_proj_weight'])
            self.attn.in_proj_bias.copy_(weights['attn_proj_bias'])
            self.linear1.weight.copy_(weights['ff1_weight'])
            self.linear1.bias.copy_(weights['ff1_bias'])
            self.linear2.weight.copy_(weights['ff2_weight'])
            self.linear2.bias.copy_(weights['ff2_bias'])


# -- HyperNet for generating weights for one transformer layer --
class HyperNet(nn.Module):
    def __init__(self, context_dim, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        total_dim = (
            3 * self.model_dim * self.model_dim +      # QKV weights
            3 * self.model_dim +                       # QKV bias
            4 * self.model_dim * self.model_dim +      # FF1 weight
            4 * self.model_dim +                       # FF1 bias
            self.model_dim * 4 * self.model_dim +      # FF2 weight
            self.model_dim                             # FF2 bias
        )
        self.fc = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.ReLU(),
            nn.Linear(512, total_dim)
        )

    def forward(self, context_vector):
        batch_size = context_vector.shape[0]
        flat = self.fc(context_vector)
        return self.unflatten(flat, batch_size)

    def unflatten(self, flat, batch_size):
        idx = 0
        out = {}

        def take(name, shape):
            nonlocal idx
            numel = torch.prod(torch.tensor(shape)).item()
            out[name] = flat[:, idx:idx+numel].view(batch_size, *shape)
            idx += numel

        d = self.model_dim
        take('attn_proj_weight', [3 * d, d])
        take('attn_proj_bias', [3 * d])
        take('ff1_weight', [4 * d, d])
        take('ff1_bias', [4 * d])
        take('ff2_weight', [d, 4 * d])
        take('ff2_bias', [d])

        return out


# -- Learned Positional Encoding --
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, model_dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# -- Meta-Optimizer Stub --
class MetaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Future: take gradients/losses and update HyperNet parameters

    def forward(self, hypernets, loss):
        # No-op for now
        return


# -- Omniformer Full Model --
class Omniformer(nn.Module):
    def __init__(self, input_dim, context_dim, model_dim=128, num_layers=6, num_heads=4, seq_len=100):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = LearnedPositionalEncoding(seq_len, model_dim)

        self.layers = nn.ModuleList()
        self.hypernets = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(CustomTransformerLayer(model_dim, num_heads))
            self.hypernets.append(HyperNet(context_dim, model_dim, num_heads))

        self.meta_optimizer = MetaOptimizer()
        self.output_head = nn.Linear(model_dim, 1)  # For regression/logit

        # TensorBoard support
        logdir = "runs/Omniformer"
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        self.global_step = 0

    def forward(self, x, context_vector):
        x = self.input_proj(x)  # shape: [B, S, D]
        x = self.pos_enc(x)

        for i, (layer, hypernet) in enumerate(zip(self.layers, self.hypernets)):
            weights = hypernet(context_vector)
            batched_weights = {k: v[0] for k, v in weights.items()}  # Use first weight set for now
            x = layer(x, external_weights=batched_weights)

            # Optional layer-wise logging
            self.writer.add_scalar(f'layer_{i}/mean_activation', x.mean().item(), self.global_step)

        output = self.output_head(x[:, -1, :])
        return output

    def log_loss(self, loss):
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.global_step += 1

    def save_checkpoint(self, path="checkpoint.pt"):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path="checkpoint.pt"):
        self.load_state_dict(torch.load(path))