from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Solution:
    pixels: np.ndarray
    values: np.ndarray
    x: np.ndarray
    p_size: float

    fitnesses: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    is_adversarial: bool | None = None
    domination_count: int = 0
    dominated_solutions: list["Solution"] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0
    loss: float = 0.0

    def copy(self) -> "Solution":
        return deepcopy(self)

    @property
    def width(self) -> int:
        return int(self.x.shape[1])

    @property
    def delta(self) -> int:
        return int(len(self.pixels))

    def euc_distance(self, img: np.ndarray) -> float:
        return float(np.sum((img - self.x) ** 2))

    def generate_image(self) -> np.ndarray:
        x_adv = self.x.copy()
        width = self.width

        for i in range(self.delta):
            pixel_index = int(self.pixels[i])
            row = pixel_index // width
            col = pixel_index % width
            x_adv[row, col] += self.values[i] * self.p_size

        return np.clip(x_adv, 0.0, 1.0)

    def evaluate(self, loss_function: Callable[[np.ndarray], list[float] | tuple[float, ...]], include_dist: bool) -> None:
        img_adv = self.generate_image()
        result = loss_function(img_adv)

        self.is_adversarial = bool(result[0])
        objectives = [float(v) for v in result[1:]]
        objectives.append(self.euc_distance(img_adv) if include_dist else 0.0)

        self.fitnesses = np.array(objectives, dtype=np.float64)
        self.loss = float(objectives[0])

    def dominates(self, other: "Solution") -> bool:
        assert self.is_adversarial is not None and other.is_adversarial is not None

        if self.is_adversarial and not other.is_adversarial:
            return True
        if not self.is_adversarial and other.is_adversarial:
            return False

        if self.is_adversarial and other.is_adversarial:
            return bool(self.fitnesses[1] < other.fitnesses[1])

        return bool(self.fitnesses[0] < other.fitnesses[0])
