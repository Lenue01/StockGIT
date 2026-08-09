"""Transformer denoiser: reconstructs masked trailing timesteps of a window
conditioned on the unmasked prefix and the current mask ratio t."""

import math

import torch
import torch.nn as nn

import config
from windows import NUM_FEATURES, NUM_PRICE_VOLUME_FEATURES


class SinusoidalEmbedding(nn.Module):
    """Maps a scalar in [0, 1] to a d_model-dim sinusoidal embedding, the
    same trick DDPMs use for the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :] * 1000  # scale up since t in [0, 1]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MaskedWindowDenoiser(nn.Module):
    def __init__(self, window_size, d_model=config.D_MODEL, n_heads=config.N_HEADS,
                 n_layers=config.N_LAYERS, dim_feedforward=config.DIM_FEEDFORWARD,
                 dropout=config.DROPOUT):
        super().__init__()
        self.window_size = window_size

        # +1 channel for the mask flag, so the model knows what's real vs placeholder.
        self.input_proj = nn.Linear(NUM_FEATURES + 1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        self.t_embed = SinusoidalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Only predicts the price/volume channels -- the time-of-session
        # channels are always given as real input (even at "masked"
        # timesteps, since the future timestamp is known), never reconstructed.
        self.output_proj = nn.Linear(d_model, NUM_PRICE_VOLUME_FEATURES)

    def forward(self, x, mask, t):
        # x: (B, T, NUM_FEATURES) input (price/vol masked, time always real),
        # mask: (B, T) 1=known/0=masked, t: (B,)
        h = torch.cat([x, mask.unsqueeze(-1)], dim=-1)  # (B, T, NUM_FEATURES + 1)
        h = self.input_proj(h) + self.pos_embed

        t_emb = self.t_embed(t).unsqueeze(1)  # (B, 1, d_model), broadcast over sequence
        h = h + t_emb

        h = self.encoder(h)
        return self.output_proj(h)  # (B, T, NUM_PRICE_VOLUME_FEATURES) reconstruction


def masked_reconstruction_loss(pred, target, mask, feature_weights=None):
    # Loss only on masked (unknown) positions -- unmasked positions are
    # conditioning context, not prediction targets.
    #
    # feature_weights (5,) rescales each channel's squared error. Volume
    # (z-scored, variance ~1) swamps the price channels (percent-change,
    # variance ~0.00007) by orders of magnitude if left unweighted, so
    # gradients barely touch price -- the model just learns to predict
    # volume's mean and price predictions never improve. train.py weights
    # by 1/std and then zeroes volume's weight entirely, since even 1/std
    # still leaves volume dominating the total loss.
    if feature_weights is None:
        feature_weights = pred.new_ones(pred.shape[-1])

    unknown = (1 - mask).unsqueeze(-1)  # (B, T, 1)
    error = (pred - target) ** 2 * feature_weights * unknown
    return error.sum() / (unknown.sum().clamp_min(1) * feature_weights.numel())


if __name__ == '__main__':
    from dataset import MaskedWindowDataset
    from windows import build_dataset

    windows, _, _ = build_dataset()
    dataset = MaskedWindowDataset(windows)
    x, mask, t, target = (
        torch.stack([dataset[i]['input'] for i in range(4)]),
        torch.stack([dataset[i]['mask'] for i in range(4)]),
        torch.stack([dataset[i]['t'] for i in range(4)]),
        torch.stack([dataset[i]['target'] for i in range(4)]),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskedWindowDenoiser(window_size=config.WINDOW_SIZE).to(device)
    x, mask, t, target = x.to(device), mask.to(device), t.to(device), target.to(device)

    pred = model(x, mask, t)
    loss = masked_reconstruction_loss(pred, target, mask)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'device: {device}')
    print(f'params: {n_params:,}')
    print(f'pred shape: {tuple(pred.shape)}')
    print(f'loss: {loss.item():.4f}')
