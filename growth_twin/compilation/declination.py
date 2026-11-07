import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_from_checkpoint(state: dict[str, Any]) -> int:
    seed = int(state.get("seed", 0))
    seed_everything(seed)
    return seed
