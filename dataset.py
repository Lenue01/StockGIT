"""Random suffix-masking dataset for training the diffusion denoiser."""

import numpy as np
import torch
from torch.utils.data import Dataset

from windows import NUM_PRICE_VOLUME_FEATURES

MASK_VALUE = 0.0  # placeholder written into masked timesteps


def cosine_mask_ratio(t):
    # t in [0, 1] -> fraction of the window to mask. Weighted toward high
    # mask ratios (MaskGIT-style schedule), so training spends more time
    # on the "mostly unknown" regime that sampling starts from.
    return 1 - np.cos(t * np.pi / 2)


class MaskedWindowDataset(Dataset):
    """Wraps normalized (N, T, NUM_FEATURES) windows. Each draw samples a
    mask ratio from the schedule, masks that many trailing price/volume
    timesteps, and returns the masked input alongside the true target."""

    def __init__(self, windows, mask_ratio_fn=cosine_mask_ratio):
        self.windows = windows.astype(np.float32)
        self.mask_ratio_fn = mask_ratio_fn
        self.window_size = windows.shape[1]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        target = self.windows[idx]  # (T, NUM_FEATURES)

        t = np.random.uniform(0, 1)
        mask_len = int(round(self.mask_ratio_fn(t) * self.window_size))

        mask = np.ones(self.window_size, dtype=np.float32)  # 1 = known, 0 = masked
        if mask_len > 0:
            mask[self.window_size - mask_len:] = 0.0

        # Only price/volume get masked -- the time-of-session channels stay
        # real even at "masked" timesteps, since the future timestamp is
        # known; only the future price/volume is actually unknown.
        masked_input = target.copy()
        masked_input[mask == 0, :NUM_PRICE_VOLUME_FEATURES] = MASK_VALUE

        return {
            'input': torch.from_numpy(masked_input),  # (T, NUM_FEATURES) -- price/vol + real time
            'mask': torch.from_numpy(mask),            # (T,)
            't': torch.tensor(t, dtype=torch.float32),
            # Sliced to just the reconstructable channels so it lines up
            # directly with the model's output shape -- callers shouldn't
            # have to know or remember which channels are predictable.
            'target': torch.from_numpy(target[:, :NUM_PRICE_VOLUME_FEATURES]),  # (T, NUM_PRICE_VOLUME_FEATURES)
        }


if __name__ == '__main__':
    from windows import build_dataset

    windows, _, _ = build_dataset()
    dataset = MaskedWindowDataset(windows)
    sample = dataset[0]
    for key, value in sample.items():
        print(f'{key}: shape={tuple(value.shape)} dtype={value.dtype}')
    print(f'masked timesteps: {int((sample["mask"] == 0).sum())}/{dataset.window_size} (t={sample["t"]:.2f})')
