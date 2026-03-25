from __future__ import annotations

import numpy as np

from .solution import Solution


def mutation(soln: Solution, pm: float, all_pixels: np.ndarray, zero_prob: float) -> None:
    pixels = soln.pixels.copy()
    values = soln.values.copy()

    replace_count = max(int(len(soln.pixels) * pm), 1)
    eps = len(soln.pixels)

    keep_indices = np.random.choice(eps, size=(eps - replace_count,), replace=False)
    kept_pixels = pixels[keep_indices]
    kept_values = values[keep_indices]

    available_pixels = np.setdiff1d(all_pixels, pixels, assume_unique=False)
    new_pixels = np.random.choice(available_pixels, size=(replace_count,), replace=False)

    one_prob = (1.0 - zero_prob) / 2.0
    new_values = np.random.choice([-1, 1, 0], size=(replace_count, 3), p=(one_prob, one_prob, zero_prob))

    soln.pixels = np.concatenate([kept_pixels, new_pixels], axis=0)
    soln.values = np.concatenate([kept_values, new_values], axis=0)


def crossover(soln1: Solution, soln2: Solution, pc: float) -> tuple[Solution, Solution]:
    switch_count = max(int(len(soln1.pixels) * pc), 1)
    k = len(soln1.pixels)

    delta1 = np.asarray([idx for idx in range(k) if soln2.pixels[idx] not in soln1.pixels])
    child1 = soln1.copy()
    if len(delta1) > 0:
        count1 = min(switch_count, len(delta1))
        switched = np.random.choice(delta1, size=(count1,), replace=False)
        child1.pixels[switched] = soln2.pixels[switched].copy()
        child1.values[switched] = soln2.values[switched].copy()

    delta2 = np.asarray([idx for idx in range(k) if soln1.pixels[idx] not in soln2.pixels])
    child2 = soln2.copy()
    if len(delta2) > 0:
        count2 = min(switch_count, len(delta2))
        switched = np.random.choice(delta2, size=(count2,), replace=False)
        child2.pixels[switched] = soln1.pixels[switched].copy()
        child2.values[switched] = soln1.values[switched].copy()

    return child1, child2


def generate_offspring(
    parents: list[tuple[Solution, Solution]],
    pc: float,
    pm: float,
    all_pixels: np.ndarray,
    zero_prob: float,
) -> list[Solution]:
    children: list[Solution] = []

    for parent1, parent2 in parents:
        child1, child2 = crossover(parent1, parent2, pc)
        mutation(child1, pm, all_pixels, zero_prob)
        mutation(child2, pm, all_pixels, zero_prob)

        assert len(np.unique(child1.pixels)) == len(child1.pixels)
        assert len(np.unique(child2.pixels)) == len(child2.pixels)

        children.extend([child1, child2])

    return children
