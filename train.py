"""Trains the masked-window denoiser. Run directly: `python train.py`
All hyperparameters live in config.py."""

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import MaskedWindowDataset
from model import MaskedWindowDenoiser, masked_reconstruction_loss
from sample import evaluate, forecast, persistence_forecast
from windows import NUM_PRICE_VOLUME_FEATURES, build_dataset, chronological_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch):
    # Linear warmup for WARMUP_EPOCHS, then cosine decay to 0 by the final epoch.
    if epoch < config.WARMUP_EPOCHS:
        return (epoch + 1) / config.WARMUP_EPOCHS
    progress = (epoch - config.WARMUP_EPOCHS) / max(1, config.EPOCHS - config.WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_snapshot(model, viz_window, viz_ticker, epoch, train_loss, val_loss, device):
    # Cheap: one forward pass on a single fixed window. Only writes raw
    # arrays -- rendering them to an image is left to visualize_training.py,
    # which runs as its own process so the GPU loop never waits on matplotlib.
    was_training = model.training
    model.eval()
    pred = forecast(model, viz_window, device=device)
    model.train(was_training)

    snapshot_dir = Path(config.SNAPSHOT_DIR)
    snapshot_dir.mkdir(exist_ok=True)
    np.savez(
        snapshot_dir / f'epoch_{epoch:04d}.npz',
        target=viz_window, pred=pred, horizon=config.FORECAST_HORIZON,
        epoch=epoch, train_loss=train_loss, val_loss=val_loss, ticker=viz_ticker,
    )


def run_epoch(model, loader, optimizer, device, train, feature_weights, self_cond_prob=0.5):
    model.train(train)
    total_loss, total_batches = 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch['input'].to(device)
            mask = batch['mask'].to(device)
            t = batch['t'].to(device)
            target = batch['target'].to(device)

            # Self-conditioning: half the time, give the model a (no-grad)
            # draft of its own prediction as extra context, then supervise a
            # second pass that has access to it -- this is what teaches the
            # model to use/correct a prior guess at inference, instead of
            # only ever knowing how to fill a mask from real context once.
            # The other half keeps self_cond=None so it still works with no
            # prior guess (the first sampling step, or if self-conditioning
            # were disabled). Draft pass is always no-grad regardless of
            # `train` -- it's not something we backprop through.
            self_cond = None
            if np.random.rand() < self_cond_prob:
                with torch.no_grad():
                    self_cond = model(x, mask, t).detach()

            pred = model(x, mask, t, self_cond=self_cond)
            loss = masked_reconstruction_loss(pred, target, mask, feature_weights)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def main():
    set_seed(config.SEED)

    windows, dates, tickers = build_dataset()
    print(f'tickers: {list(np.unique(tickers))}, total windows: {len(windows)}')

    train_windows, val_windows, _, val_tickers = chronological_split(windows, dates, tickers)
    print(f'train windows: {len(train_windows)}, val windows: {len(val_windows)}')

    # One fixed window per ticker, rotated through across epochs -- each
    # individual window is stable enough to watch improve over several
    # consecutive epochs, but cycling tickers makes it obvious in the live
    # view that every ticker is actually part of training (a single
    # permanently-fixed window made that invisible: whichever ticker
    # happened to be first always got shown, no matter what else was added).
    viz_tickers = list(np.unique(val_tickers))
    viz_windows = {t: val_windows[np.where(val_tickers == t)[0][0]] for t in viz_tickers}

    # Fixed sample (separate RNG so it doesn't disturb the dataset's own
    # masking/shuffling randomness) for a per-epoch directional-accuracy /
    # MAE readout against the persistence baseline -- this is the number
    # that actually says whether it's learning something useful, not loss.
    dir_acc_idx = np.random.default_rng(config.SEED).choice(
        len(val_windows), size=min(config.DIR_ACC_SAMPLE_SIZE, len(val_windows)), replace=False)
    dir_acc_windows = val_windows[dir_acc_idx]
    baseline_mae, baseline_dir_acc = evaluate(persistence_forecast, dir_acc_windows)
    print(f'persistence baseline ({len(dir_acc_windows)} windows): MAE={baseline_mae:.4f}  dir_acc={baseline_dir_acc:.3f}')

    train_loader = DataLoader(MaskedWindowDataset(train_windows), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MaskedWindowDataset(val_windows), batch_size=config.BATCH_SIZE, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskedWindowDenoiser(window_size=config.WINDOW_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Price channels (percent-change, std ~0.008) and volume (z-scored,
    # std ~1) are wildly different scales. 1/std weighting was tried to
    # equalize them, but even that leaves volume contributing ~1.0 to the
    # loss vs ~0.007 per price channel (weight*variance = std) -- volume
    # still ate ~97% of the gradient budget and price never learned past
    # a persistence-level baseline. Since forecasting price is the actual
    # goal and volume is still given as conditioning input (just not a
    # prediction target), zero its loss weight out entirely rather than
    # trying to tune the disparity down further.
    channel_std = train_windows[..., :NUM_PRICE_VOLUME_FEATURES].reshape(-1, NUM_PRICE_VOLUME_FEATURES).std(axis=0)
    feature_weights = torch.tensor(1.0 / (channel_std + 1e-8), dtype=torch.float32, device=device)
    feature_weights[4] = 0.0  # volume: input/context only, not a reconstruction target
    print(f'feature weights (1/std, volume excluded): {feature_weights.tolist()}')

    best_val_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, feature_weights=feature_weights)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, feature_weights=feature_weights)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            marker = ' (saved)'

        if epoch % config.SNAPSHOT_EVERY == 0:
            viz_ticker = viz_tickers[(epoch - 1) % len(viz_tickers)]
            save_snapshot(model, viz_windows[viz_ticker], viz_ticker, epoch, train_loss, val_loss, device)

        was_training = model.training
        model.eval()
        mae, dir_acc = evaluate(lambda w: forecast(model, w, device=device), dir_acc_windows)
        model.train(was_training)

        print(f'epoch {epoch:3d}/{config.EPOCHS}  lr={lr:.2e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}  '
              f'MAE={mae:.4f}  dir_acc={dir_acc:.3f} (persistence={baseline_dir_acc:.3f})')


if __name__ == '__main__':
    main()
