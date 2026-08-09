"""Forecasts the trailing FORECAST_HORIZON minutes of a window via
progressive unmasking, and evaluates the trained checkpoint on held-out
validation windows. Run directly: `python sample.py`"""

import numpy as np
import torch

import config
from dataset import MASK_VALUE
from model import MaskedWindowDenoiser
from windows import NUM_PRICE_VOLUME_FEATURES, build_dataset, chronological_split

CLOSE_COL = 3  # index of Close in [Open, High, Low, Close, Volume]


def cosine_mask_ratio_inverse(ratio):
    # Inverse of cosine_mask_ratio: given a target mask fraction, find the t
    # that the schedule should start at, so it starts at exactly `horizon`.
    ratio = np.clip(ratio, 0.0, 1.0)
    return (2 / np.pi) * np.arccos(1 - ratio)


@torch.no_grad()
def forecast(model, context, horizon=config.FORECAST_HORIZON, steps=config.SAMPLING_STEPS, device='cpu'):
    """context: (WINDOW_SIZE, 5) array where the last `horizon` rows are
    unknown (any placeholder values -- they get overwritten). Returns a
    (WINDOW_SIZE, 5) array with the last `horizon` rows filled in.

    Unlike the old progressive-unmasking version, the horizon stays fully
    masked (mask=0, x=MASK_VALUE) for every step -- it's never permanently
    frozen a position at a time. Instead each step re-predicts the whole
    horizon from scratch and feeds that guess back in as self-conditioning
    for the next step, so later steps can revise earlier ones. This only
    works because training actually supervises the model to use self_cond
    (see run_epoch's self-conditioning pass) -- earlier we tried refining
    positions through the `mask=1`/x pathway instead and it blew up,
    because that pathway is never supervised by masked_reconstruction_loss
    (loss only applies where mask=0) so the model has no idea what to do
    with a value written there. self_cond is the pathway actually trained
    for this.

    t stays fixed at t_start across every step, rather than decaying to 0
    the way the old schedule did -- t is what tells the model how much of
    the window is masked, and here that never changes (always exactly
    `horizon` positions), so decaying t would tell the model "almost
    everything is known" while still showing it the same 10 unknown
    positions: a (t, mask) combination it never saw in training, since
    training always ties t to the actual mask size via mask_ratio_fn. Only
    self_cond is meant to signal "this is a later, more-informed guess."
    """
    window_size = context.shape[0]
    horizon_start = window_size - horizon
    t_start = cosine_mask_ratio_inverse(horizon / window_size)
    t = torch.full((1,), t_start, dtype=torch.float32, device=device)

    x = context.copy()
    mask = np.ones(window_size, dtype=np.float32)
    mask[horizon_start:] = 0.0
    # Only price/volume get blanked out -- the time-of-session channels stay
    # real even for "masked" positions, since the future timestamp is known.
    # This never changes during sampling -- the horizon stays "masked" the
    # whole time, exactly like every training example.
    x[horizon_start:, :NUM_PRICE_VOLUME_FEATURES] = MASK_VALUE

    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    self_cond = torch.zeros(1, window_size, NUM_PRICE_VOLUME_FEATURES, device=device)

    pred = None
    for i in range(steps):
        pred = model(x_t, mask_t, t, self_cond=self_cond)  # (1, T, NUM_PRICE_VOLUME_FEATURES)
        self_cond = pred  # next step gets to see and revise this step's guess

    result = x_t.clone()
    result[0, horizon_start:, :NUM_PRICE_VOLUME_FEATURES] = pred[0, horizon_start:]
    return result.squeeze(0).cpu().numpy()


def persistence_forecast(context, horizon=config.FORECAST_HORIZON):
    """Naive baseline: "nothing changes" -- repeats the last known row for
    the whole horizon. Any model worth using needs to beat this."""
    known_upto = context.shape[0] - horizon
    pred = context.copy()
    pred[known_upto:, :NUM_PRICE_VOLUME_FEATURES] = context[known_upto - 1, :NUM_PRICE_VOLUME_FEATURES]
    return pred


def directional_accuracy(pred_close, true_close, last_known_close):
    pred_dir = np.sign(pred_close - last_known_close)
    true_dir = np.sign(true_close - last_known_close)
    # A predicted-flat call (pred_dir == 0, as persistence always gives)
    # gets half credit rather than an automatic miss -- otherwise it's
    # structurally guaranteed to score ~0 regardless of actual merit,
    # since true_dir is essentially never exactly 0.
    hit = np.where(pred_dir == true_dir, 1.0, np.where(pred_dir == 0, 0.5, 0.0))
    return float(hit.mean())


def evaluate(forecast_fn, val_windows):
    last_known_idx = config.WINDOW_SIZE - config.FORECAST_HORIZON - 1

    errors, dir_hits = [], 0
    for w in val_windows:
        pred = forecast_fn(w)

        true_future = w[config.WINDOW_SIZE - config.FORECAST_HORIZON:, CLOSE_COL]
        pred_future = pred[config.WINDOW_SIZE - config.FORECAST_HORIZON:, CLOSE_COL]
        errors.append(np.abs(pred_future - true_future).mean())

        last_close = w[last_known_idx, CLOSE_COL]
        dir_hits += directional_accuracy(pred_future[-1], true_future[-1], last_close)

    return np.mean(errors), dir_hits / len(val_windows)


if __name__ == '__main__':
    windows, dates, tickers = build_dataset()
    _, val_windows, _, _ = chronological_split(windows, dates, tickers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskedWindowDenoiser(window_size=config.WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    model.eval()

    mae, dir_acc = evaluate(lambda w: forecast(model, w, device=device), val_windows)
    baseline_mae, baseline_dir_acc = evaluate(persistence_forecast, val_windows)

    print(f'val windows evaluated: {len(val_windows)}')
    print(f'{"":12s}{"MAE (close)":>14s}{"dir. accuracy":>16s}')
    print(f'{"model":12s}{mae:14.4f}{dir_acc:16.3f}')
    print(f'{"persistence":12s}{baseline_mae:14.4f}{baseline_dir_acc:16.3f}')
