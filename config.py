# config.py
# Central configuration for the chess piece object detection project.
# All paths, hyperparameters and constants live here.

import os


# ─── Run identifier ───────────────────────────────────────────────────────────
# Bump this manually before each new training run so output files
# (plots, weights) are never overwritten.

RUN_ID = 1

# ─── Base directories ─────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(*parts) -> str:
    """Build an absolute path relative to BASE_DIR."""
    return os.path.join(BASE_DIR, *parts)

# ─── Dataset paths ────────────────────────────────────────────────────────────
# Each dataset has a train / valid / test split with its own annotations CSV.
# Using os.path.join keeps this cross-platform (no hardcoded backslashes).

# Set 1
DS1_DIR         = _p("datasets", "set1")
DS1_TRAIN_DIR   = _p("datasets", "set1", "train")
DS1_VALID_DIR   = _p("datasets", "set1", "valid")
DS1_TEST_DIR    = _p("datasets", "set1", "test")
DS1_TRAIN_CSV   = os.path.join(DS1_TRAIN_DIR, "_annotations.csv")
DS1_VALID_CSV   = os.path.join(DS1_VALID_DIR, "_annotations.csv")
DS1_TEST_CSV    = os.path.join(DS1_TEST_DIR,  "_annotations.csv")

# Set 2
DS2_DIR         = _p("datasets", "set2")
DS2_TRAIN_DIR   = _p("datasets", "set2", "train")
DS2_VALID_DIR   = _p("datasets", "set2", "valid")
DS2_TEST_DIR    = _p("datasets", "set2", "test")
DS2_TRAIN_CSV   = os.path.join(DS2_TRAIN_DIR, "_annotations.csv")
DS2_VALID_CSV   = os.path.join(DS2_VALID_DIR, "_annotations.csv")
DS2_TEST_CSV    = os.path.join(DS2_TEST_DIR,  "_annotations.csv")

# Convenience: all splits as (csv_path, image_dir) pairs grouped by split
# data_loader can iterate over these instead of hardcoding each path
SPLITS = {
    "train": [
        (DS1_TRAIN_CSV, DS1_TRAIN_DIR),
        (DS2_TRAIN_CSV, DS2_TRAIN_DIR),
    ],
    "valid": [
        (DS1_VALID_CSV, DS1_VALID_DIR),
        (DS2_VALID_CSV, DS2_VALID_DIR),
    ],
    "test": [
        (DS1_TEST_CSV, DS1_TEST_DIR),
        (DS2_TEST_CSV, DS2_TEST_DIR),
    ],
}

# ─── Output directories ───────────────────────────────────────────────────────

OUTPUT_DIR  = _p("outputs")
PLOTS_DIR   = _p("outputs", "plots", f"run{RUN_ID}")
MODELS_DIR  = _p("outputs", "models")

# ─── Image settings ───────────────────────────────────────────────────────────

IMG_HEIGHT   = 64
IMG_WIDTH    = 64
IMG_CHANNELS = 3   # RGB

# ─── Class mapping ────────────────────────────────────────────────────────────
# Set2 uses short abbreviations; set1 uses full names.
# 'board' is intentionally absent — it is filtered out before this map is used.

CLASS_MAP = {
    # Short names (set2)
    'bb': 'black-bishop', 'bk': 'black-king',   'bn': 'black-knight',
    'bp': 'black-pawn',   'bq': 'black-queen',  'br': 'black-rook',
    'wb': 'white-bishop', 'wk': 'white-king',   'wn': 'white-knight',
    'wp': 'white-pawn',   'wq': 'white-queen',  'wr': 'white-rook',
    # Full names (set1)
    'black-bishop': 'black-bishop', 'black-king':   'black-king',
    'black-knight': 'black-knight', 'black-pawn':   'black-pawn',
    'black-queen':  'black-queen',  'black-rook':   'black-rook',
    'white-bishop': 'white-bishop', 'white-king':   'white-king',
    'white-knight': 'white-knight', 'white-pawn':   'white-pawn',
    'white-queen':  'white-queen',  'white-rook':   'white-rook',
}

CLASS_NAMES = sorted(set(CLASS_MAP.values()))
NUM_CLASSES  = len(CLASS_NAMES)

# ─── Training hyperparameters ─────────────────────────────────────────────────

BATCH_SIZE       = 32
EPOCHS           = 30
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.2
RANDOM_SEED      = 42

