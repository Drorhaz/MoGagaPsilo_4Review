# Copyright (c) 2026 Dror Hazan. All Rights Reserved.
# Provided for academic peer review only. No reproduction, distribution,
# modification, or derivative works permitted without written consent.
# Contact: dror.hazan@mail.huji.ac.il

"""
src/utils.py — Demo configuration for publish4Jason review package.

Provides DEMO_CFG, a standalone dict that mirrors the structure
pipeline_config.CONFIG would produce, using the locked defaults
from config_v1.yaml. No YAML loading, no file I/O.
"""

from pathlib import Path


def get_demo_root() -> Path:
    """Locate the demo root by searching for data/ + src/ directories."""
    start = Path(__file__).resolve().parent.parent  # parent of src/
    for p in [start] + list(start.parents)[:3]:
        if (p / 'data').is_dir() and (p / 'src').is_dir():
            return p
    return start


DEMO_CFG = {
    # ------------------------------------------------------------------ #
    # Step 06 inherited parameters — locked defaults from config_v1.yaml  #
    # ------------------------------------------------------------------ #
    "fs_target": 120.0,
    "sg_window_sec": 0.175,
    "sg_polyorder": 3,
    "ref_search_sec": 8.0,
    "ref_window_sec": 1.0,
    "static_search_step_sec": 0.1,
    "reference_variance_threshold": 100.0,
    "min_run_seconds": 5.0,
    "max_gap_pos_sec": 1.0,

    # Uppercase aliases (pulsicity.py uses cfg.get(lower, cfg.get(upper, default)))
    "FS_TARGET": 120.0,
    "SG_WINDOW_SEC": 0.175,
    "SG_POLYORDER": 3,
    "REF_SEARCH_SEC": 8.0,
    "REF_WINDOW_SEC": 1.0,
    "STATIC_SEARCH_STEP_SEC": 0.1,
    "MAX_GAP_POS_SEC": 1.0,

    # Step 06 provenance — enforce_cleaning=False for subject 651 recordings
    "step_06": {"enforce_cleaning": False},
    "ENFORCE_CLEANING": False,
    "enforce_cleaning": False,
}
