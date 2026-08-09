"""Live window showing three views of the tracked validation window as
train.py writes new epoch snapshots: what the model actually sees (the
masked/blanked tail), what it guesses for the hidden part, and the real
values.

Run in a separate terminal alongside `python train.py`. This is a fully
separate process doing CPU-only matplotlib rendering -- it never touches
the GPU or blocks the training loop.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import config
from dataset import MASK_VALUE

SNAPSHOT_DIR = Path(config.SNAPSHOT_DIR)
POLL_SECONDS = 0.5

UP_COLOR = '#0ca30c'
DOWN_COLOR = '#d03b3b'
MASK_COLOR = '#9aa0a6'
GUESS_COLOR = '#d97b0d'


def draw_candles(ax, ohlc, colors, title):
    ax.clear()
    for i, (o, h, l, c) in enumerate(ohlc):
        color = colors[i]
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-4), color=color))
    ax.set_xlim(-1, len(ohlc))
    ax.set_title(title, fontsize=10, loc='left')


def bar_colors(ohlc):
    return [UP_COLOR if c >= o else DOWN_COLOR for o, h, l, c in ohlc]


def latest_snapshot():
    snaps = sorted(SNAPSHOT_DIR.glob('epoch_*.npz'))
    return snaps[-1] if snaps else None


def update(fig, axes, snapshot_path):
    data = np.load(snapshot_path)
    target = data['target']  # (T, 5) ground truth, normalized
    pred = data['pred']       # (T, 5) model's reconstruction
    horizon = int(data['horizon'])
    epoch = int(data['epoch'])
    train_loss = float(data['train_loss'])
    val_loss = float(data['val_loss'])
    ticker = str(data['ticker']) if 'ticker' in data else '?'

    window_size = target.shape[0]
    known_upto = window_size - horizon
    ax_masked, ax_guess, ax_actual = axes

    # Panel 1: exactly what the model's input looked like -- known prefix
    # real, masked tail flattened to MASK_VALUE.
    masked_input = target.copy()
    masked_input[known_upto:] = MASK_VALUE
    colors = bar_colors(target[:known_upto, :4]) + [MASK_COLOR] * horizon
    draw_candles(ax_masked, masked_input[:, :4], colors, 'masked input (what the model sees)')

    # Panel 2: known prefix real (context), tail = model's guess.
    guess = target.copy()
    guess[known_upto:] = pred[known_upto:]
    colors = bar_colors(target[:known_upto, :4]) + [GUESS_COLOR] * horizon
    draw_candles(ax_guess, guess[:, :4], colors, "model's guess")

    # Panel 3: ground truth, nothing hidden.
    draw_candles(ax_actual, target[:, :4], bar_colors(target[:, :4]), 'actual')

    for ax in axes:
        ax.axvspan(known_upto - 0.5, window_size - 0.5, color=MASK_COLOR, alpha=0.08)

    fig.suptitle(f'epoch {epoch}   ticker={ticker}   train_loss={train_loss:.4f}   val_loss={val_loss:.4f}')
    fig.canvas.draw_idle()


def main():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, sharey=True)
    fig.canvas.manager.set_window_title('Training preview')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show(block=False)

    print(f'watching {SNAPSHOT_DIR}/ for new epoch snapshots (close the window or Ctrl+C to stop)...', flush=True)
    last_seen = None
    while plt.fignum_exists(fig.number):
        snapshot_path = latest_snapshot()
        if snapshot_path is not None and snapshot_path != last_seen:
            update(fig, axes, snapshot_path)
            last_seen = snapshot_path
        fig.canvas.flush_events()
        plt.pause(POLL_SECONDS)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
