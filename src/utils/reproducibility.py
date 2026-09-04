"""
Reproducibility manager for setting random seeds across packages.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixes random seeds across Python built-ins, NumPy, PyTorch, and environment settings.

    Args:
        seed: Integer random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
