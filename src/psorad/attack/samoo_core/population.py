from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .solution import Solution


@dataclass
class Population:
    population: list[Solution]
    loss_function: Any
    include_dist: bool
    fronts: list[list[Solution]] | None = None

    def evaluate(self) -> None:
        for candidate in self.population:
            candidate.evaluate(self.loss_function, self.include_dist)

    def find_adv_solns(self, max_dist: float) -> list[Solution]:
        return [
            candidate
            for candidate in self.population
            if bool(candidate.is_adversarial) and float(candidate.fitnesses[1]) <= max_dist
        ]
