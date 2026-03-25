# config.py
# Central configuration for the chess piece object detection project.

import os

# ─── Run identifier ───────────────────────────────────────────────────────────

RUN_ID = 12

# ─── Base directories ─────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(*parts) -> str:
    """Builds an absolute path relative to BASE_DIR."""
    return os.path.join(BASE_DIR, *parts)

# ─── Dataset paths ────────────────────────────────────────────────────────────

DS1_TRAIN_DIR   = _p("datasets", "set1", "train")
DS1_VALID_DIR   = _p("datasets", "set1", "valid")
DS1_TEST_DIR    = _p("datasets", "set1", "test")
DS1_TRAIN_CSV   = os.path.join(DS1_TRAIN_DIR, "_annotations.csv")
DS1_VALID_CSV   = os.path.join(DS1_VALID_DIR, "_annotations.csv")
DS1_TEST_CSV    = os.path.join(DS1_TEST_DIR,  "_annotations.csv")

DS2_TRAIN_DIR   = _p("datasets", "set2", "train")
DS2_VALID_DIR   = _p("datasets", "set2", "valid")
DS2_TEST_DIR    = _p("datasets", "set2", "test")
DS2_TRAIN_CSV   = os.path.join(DS2_TRAIN_DIR, "_annotations.csv")
DS2_VALID_CSV   = os.path.join(DS2_VALID_DIR, "_annotations.csv")
DS2_TEST_CSV    = os.path.join(DS2_TEST_DIR,  "_annotations.csv")

SPLITS = {
    "train": [(DS1_TRAIN_CSV, DS1_TRAIN_DIR), (DS2_TRAIN_CSV, DS2_TRAIN_DIR)],
    "valid": [(DS1_VALID_CSV, DS1_VALID_DIR), (DS2_VALID_CSV, DS2_VALID_DIR)],
    "test":  [(DS1_TEST_CSV,  DS1_TEST_DIR),  (DS2_TEST_CSV,  DS2_TEST_DIR)],
}

# ─── Output directories ───────────────────────────────────────────────────────

OUTPUT_DIR  = _p("outputs")
PLOTS_DIR   = _p("outputs", "plots", f"run{RUN_ID}")
MODELS_DIR  = _p("outputs", "models")

# ─── Image settings ───────────────────────────────────────────────────────────

IMG_HEIGHT   = 640
IMG_WIDTH    = 640
IMG_CHANNELS = 3

# ─── Detection grid ───────────────────────────────────────────────────────────

GRID_S = 20

# ─── Class mapping ────────────────────────────────────────────────────────────

CLASS_MAP = {
    'bb': 'black-bishop', 'bk': 'black-king',   'bn': 'black-knight',
    'bp': 'black-pawn',   'bq': 'black-queen',  'br': 'black-rook',
    'wb': 'white-bishop', 'wk': 'white-king',   'wn': 'white-knight',
    'wp': 'white-pawn',   'wq': 'white-queen',  'wr': 'white-rook',
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

BATCH_SIZE       = 8
EPOCHS           = 200
LEARNING_RATE    = 1e-4
RANDOM_SEED      = 42

# Extra multipliers on top of the auto-computed inverse-frequency weights.
# Use this to manually boost classes that are still being confused.
CLASS_WEIGHT_OVERRIDES = {
    'black-bishop': 3.0,
}