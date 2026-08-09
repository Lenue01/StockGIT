"""Central knobs for the whole pipeline. Edit this file instead of hunting
through data/model/train/sample code for hardcoded values."""

# --- Data ---
TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "META", "GOOGL", "GOOG", "NFLX",
    "PLTR", "INTC", "BAC", "F", "PLUG", "SOFI", "NIO", "AAL", "MARA", "RIOT",
    "JPM", "BABA", "VALE", "RIG", "C", "PFE", "WFC", "XOM", "CVX", "LCID",
    "PYPL", "HOOD", "RIVN", "SMR", "DKNG", "COIN", "CLSK", "AUPH", "NOK", "SNAP",
    "UBER", "LYFT", "KVUE", "CSCO", "T", "VZ", "CMCSA", "DIS", "KO", "PEP",
    "WMT", "COST", "HD", "PG", "JNJ", "UNH", "LLY", "ABBV", "MRK", "BMY",
    "GILD", "AMGN", "BA", "GE", "CAT", "DE", "HON", "MMM", "LMT", "RTX",
    "DAL", "UAL", "LUV", "CCL", "RCL", "NCLH", "MGM", "WYNN", "PENN", "SCHW",
    "GS", "MS", "AXP", "V", "MA", "AFRM", "UPST", "SOUN", "BBAI",
    "AI", "RKLB", "ASTS", "HUT", "CIFR", "IREN", "BTBT", "MSTR", "MU"
]
 # add more tickers here to grow the dataset, e.g. ['OKLO', 'NVDA', 'SOFI']

PERIOD = '7d'        # yfinance's max lookback for 1-minute candles
INTERVAL = '1m'
WINDOW_SIZE = 60      # minutes per training window
VAL_DAYS = 1          # most recent trading day(s) per ticker held out for validation

# --- Model (Transformer denoiser) ---
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# --- Training ---
SEED = 0                # fixes weight init + masking/shuffling randomness so runs are comparable
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4
# Linear LR warmup before cosine decay kicks in. Scaled as ~4% of EPOCHS
# (the original tuned ratio was 20/500) rather than a fixed number --
# fixed at 20 this ate 40% of a 50-epoch test run, leaving barely any
# time for the cosine decay phase that actually converges the model,
# which is why dir_acc got worse after shortening EPOCHS for testing.
WARMUP_EPOCHS = max(1, EPOCHS // 25)
CHECKPOINT_PATH = 'denoiser.pt'

# --- Sampling / evaluation ---
FORECAST_HORIZON = 10  # trailing minutes to forecast at inference
# progressive-unmasking steps used to fill in the horizon. Was 50 -- but
# forecast() (sample.py) permanently freezes each revealed position, and
# the model is never trained to revise an already-"known" (mask=1)
# position (masked_reconstruction_loss in model.py only supervises
# mask=0 positions), so extra steps beyond ~FORECAST_HORIZON have no new
# information to add and just compound early guesses with no way to
# correct them. Swept 1/2/3/5/10/20/50 steps against the persistence
# baseline: MAE got monotonically worse past ~5 steps (0.0016 -> 0.0035
# at 50 steps, vs. persistence's 0.0014), while directional accuracy
# peaked around 5-10 steps (0.533-0.537 vs. persistence's 0.517). 5 is
# the best tradeoff -- near-best MAE and a real directional edge.
SAMPLING_STEPS = 10

# --- Training visualization ---
SNAPSHOT_DIR = 'snapshots'  # train.py drops one .npz here per epoch for visualize_training.py to render
SNAPSHOT_EVERY = 1           # epochs between snapshots

# Fixed random sample of val windows train.py forecasts every epoch to print
# directional accuracy / MAE against the persistence baseline -- loss alone
# doesn't tell you whether the model beats "nothing changes" (we've seen
# loss improve while forecast quality got worse). Kept well below the full
# ~32k val set since forecast() runs one window at a time (unbatched) --
# this adds SAMPLE_SIZE * SAMPLING_STEPS forward passes per epoch. Bump it
# up for a more trustworthy number once you're not watching it live.
DIR_ACC_SAMPLE_SIZE = 300
