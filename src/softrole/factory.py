"""Model construction from a saved or derived ``model_config`` dictionary."""
from __future__ import annotations

from softrole.baselines import BASELINE_CLASSES, BASELINE_MODES
from softrole.model import SoftRoleNet


def build_model(model_config):
    """Return the baseline actor for a baseline mode, otherwise ``SoftRoleNet``."""
    mode = model_config.get("mode")
    if mode in BASELINE_MODES:
        return BASELINE_CLASSES[mode](**model_config)
    return SoftRoleNet(**model_config)
