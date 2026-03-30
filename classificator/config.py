# config.py
# Centrale configuratie voor de chess piece classifier.

import os

# ─── Run identifier ───────────────────────────────────────────────────────────

RUN_ID = 18

# ─── Base directories ─────────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

def _r(*parts): return os.path.join(ROOT_DIR, *parts)
def _p(*parts): return os.path.join(BASE_DIR,  *parts)

# ─── Dataset paths (zelfde datasets als root project) ─────────────────────────

DS1_TRAIN_DIR = _r("datasets", "set1", "train")
DS1_VALID_DIR = _r("datasets", "set1", "valid")
DS1_TEST_DIR  = _r("datasets", "set1", "test")
DS1_TRAIN_CSV = os.path.join(DS1_TRAIN_DIR, "_annotations.csv")
DS1_VALID_CSV = os.path.join(DS1_VALID_DIR, "_annotations.csv")
DS1_TEST_CSV  = os.path.join(DS1_TEST_DIR,  "_annotations.csv")

DS2_TRAIN_DIR = _r("datasets", "set2", "train")
DS2_VALID_DIR = _r("datasets", "set2", "valid")
DS2_TEST_DIR  = _r("datasets", "set2", "test")
DS2_TRAIN_CSV = os.path.join(DS2_TRAIN_DIR, "_annotations.csv")
DS2_VALID_CSV = os.path.join(DS2_VALID_DIR, "_annotations.csv")
DS2_TEST_CSV  = os.path.join(DS2_TEST_DIR,  "_annotations.csv")

SPLITS = {
    "train": [(DS1_TRAIN_CSV, DS1_TRAIN_DIR), (DS2_TRAIN_CSV, DS2_TRAIN_DIR)],
    "valid": [(DS1_VALID_CSV, DS1_VALID_DIR), (DS2_VALID_CSV, DS2_VALID_DIR)],
    "test":  [(DS1_TEST_CSV,  DS1_TEST_DIR),  (DS2_TEST_CSV,  DS2_TEST_DIR)],
}

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

# ─── Classifier instellingen ──────────────────────────────────────────────────

PATCH_SIZE  = 80
BATCH_SIZE  = 16
EPOCHS      = 150

# ─── Output ───────────────────────────────────────────────────────────────────

PLOTS_DIR       = _p("outputs", "plots", f"run{RUN_ID}")
MODELS_DIR      = _p("outputs", "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f"classifier_run{RUN_ID}.h5")
