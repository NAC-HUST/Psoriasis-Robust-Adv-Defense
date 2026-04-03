from __future__ import annotations

import random

import numpy as np
import paddle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
