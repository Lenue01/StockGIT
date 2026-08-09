"""One-off diagnostic: trains the same model/data/split under different
optimizer settings to test whether the loss plateau we've been seeing is
caused by the LR schedule / weight decay, or by a ceiling in the data
itself. Everything is held identical (seed, model init, split, feature
weights) except lr and weight_decay -- if both arms plateau at roughly
the same val_loss, that's evidence against "it's an optimizer problem."
Run: `python experiment_lr.py`"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import MaskedWindowDataset
from model import MaskedWindowDenoiser
from train import run_epoch, set_seed
from windows import NUM_PRICE_VOLUME_FEATURES, build_dataset, chronological_split

EXPERIMENT_EPOCHS = 40
WARMUP_EPOCHS = 4


def run_arm(name, train_windows, val_windows, feature_weights, device, lr, weight_decay):
    set_seed(config.SEED)  # identical weight init across arms
    model = MaskedWindowDenoiser(window_size=config.WINDOW_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EXPERIMENT_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(MaskedWindowDataset(train_windows), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MaskedWindowDataset(val_windows), batch_size=config.BATCH_SIZE, shuffle=False)

    history = []
    for epoch in range(1, EXPERIMENT_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, feature_weights=feature_weights)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, feature_weights=feature_weights)
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        history.append((epoch, cur_lr, train_loss, val_loss))
        if epoch == 1 or epoch % 5 == 0:
            print(f'[{name}] epoch {epoch:3d}/{EXPERIMENT_EPOCHS}  lr={cur_lr:.2e}  '
                  f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    return history


def main():
    set_seed(config.SEED)
    windows, dates, tickers = build_dataset()
    print(f'tickers: {len(np.unique(tickers))}, total windows: {len(windows)}')

    train_windows, val_windows, _, _ = chronological_split(windows, dates, tickers)
    print(f'train windows: {len(train_windows)}, val windows: {len(val_windows)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channel_std = train_windows[..., :NUM_PRICE_VOLUME_FEATURES].reshape(-1, NUM_PRICE_VOLUME_FEATURES).std(axis=0)
    feature_weights = torch.tensor(1.0 / (channel_std + 1e-8), dtype=torch.float32, device=device)

    print('\n=== baseline: lr=3e-4, weight_decay=0.01 (current config.py) ===')
    baseline = run_arm('baseline', train_windows, val_windows, feature_weights, device,
                        lr=3e-4, weight_decay=0.01)

    print('\n=== test: lr=1.5e-3 (5x), weight_decay=0.0 ===')
    test = run_arm('test', train_windows, val_windows, feature_weights, device,
                    lr=1.5e-3, weight_decay=0.0)

    print('\n=== final val_loss comparison ===')
    print(f'baseline: {baseline[-1][3]:.4f}   test: {test[-1][3]:.4f}')
    best_baseline = min(h[3] for h in baseline)
    best_test = min(h[3] for h in test)
    print(f'best baseline val_loss: {best_baseline:.4f}   best test val_loss: {best_test:.4f}')


if __name__ == '__main__':
    main()
