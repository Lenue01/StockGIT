"""Forecasts the trailing FORECAST_HORIZON minutes of a window via
progressive unmasking, and evaluates the trained checkpoint on held-out
validation windows. Run directly: `python sample.py`"""

import numpy as np
import torch

import config
from dataset import MASK_VALUE, cosine_mask_ratio
from model import MaskedWindowDenoiser
from windows import NUM_PRICE_VOLUME_FEATURES, build_dataset, chronological_split

CLOSE_COL = 3  # index of Close in [Open, High, Low, Close, Volume]


def cosine_mask_ratio_inverse(ratio):
    # Inverse of cosine_mask_ratio: given a target mask fraction, find the t
    # that produces it, so the reveal schedule can start exactly at `horizon`.
    ratio = np.clip(ratio, 0.0, 1.0)
    return (2 / np.pi) * np.arccos(1 - ratio)


@torch.no_grad()
def forecast(model, context, horizon=config.FORECAST_HORIZON, steps=config.SAMPLING_STEPS, device='cpu'):
    """context: (WINDOW_SIZE, 5) array where the last `horizon` rows are
    unknown (any placeholder values -- they get overwritten). Returns a
    (WINDOW_SIZE, 5) array with the last `horizon` rows filled in by
    progressively revealing more of the schedule each step, using the
    model's own predictions as the newly "known" values."""
    window_size = context.shape[0]
    t_start = cosine_mask_ratio_inverse(horizon / window_size)
    schedule = np.linspace(t_start, 0.0, steps + 1)

    x = context.copy()
    mask = np.ones(window_size, dtype=np.float32)
    mask[window_size - horizon:] = 0.0
    # Only price/volume get blanked out -- the time-of-session channels stay
    # real even for "masked" positions, since the future timestamp is known.
    x[mask == 0, :NUM_PRICE_VOLUME_FEATURES] = MASK_VALUE

    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)

    for i in range(steps):
        t = torch.tensor([schedule[i]], dtype=torch.float32, device=device)
        pred = model(x_t, mask_t, t)  # (1, T, NUM_PRICE_VOLUME_FEATURES)

        next_mask_len = int(round(cosine_mask_ratio(schedule[i + 1]) * window_size))
        reveal_from = window_size - next_mask_len
        cur_masked_from = window_size - int(round(cosine_mask_ratio(schedule[i]) * window_size))

        # Positions that go from "masked" to "revealed" this step: accept
        # the model's prediction as the new known value going forward.
        if reveal_from > cur_masked_from:
            x_t[0, cur_masked_from:reveal_from, :NUM_PRICE_VOLUME_FEATURES] = pred[0, cur_masked_from:reveal_from]
            mask_t[0, cur_masked_from:reveal_from] = 1.0

    return x_t.squeeze(0).cpu().numpy()


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
