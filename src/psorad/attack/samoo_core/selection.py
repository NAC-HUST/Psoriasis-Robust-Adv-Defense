from __future__ import annotations

import numpy as np

from .solution import Solution


def fast_nondominated_sort(population: list[Solution]) -> list[list[Solution]]:
    fronts: list[list[Solution]] = [[]]

    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []

        for other in population:
            if individual.dominates(other):
                individual.dominated_solutions.append(other)
            elif other.dominates(individual):
                individual.domination_count += 1

        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)

    i = 0
    while len(fronts[i]) > 0:
        next_front: list[Solution] = []
        for individual in fronts[i]:
            for dominated in individual.dominated_solutions:
                dominated.domination_count -= 1
                if dominated.domination_count == 0:
                    dominated.rank = i + 1
                    next_front.append(dominated)
        i += 1
        fronts.append(next_front)

    return fronts


def calculate_crowding_distance(front: list[Solution]) -> None:
    if not front:
        return

    n = len(front)
    for candidate in front:
        candidate.crowding_distance = 0.0

    objective_count = len(front[0].fitnesses)
    for objective_idx in range(objective_count):
        front.sort(key=lambda x: float(x.fitnesses[objective_idx]))
        front[0].crowding_distance = 1e9
        front[-1].crowding_distance = 1e9

        values = [float(x.fitnesses[objective_idx]) for x in front]
        scale = max(values) - min(values)
        if scale == 0.0:
            scale = 1.0

        for i in range(1, n - 1):
            numerator = float(front[i + 1].fitnesses[objective_idx]) - float(front[i - 1].fitnesses[objective_idx])
            front[i].crowding_distance += numerator / scale


def crowding_operator(a: Solution, b: Solution) -> int:
    if (a.rank < b.rank) or (a.rank == b.rank and a.crowding_distance > b.crowding_distance):
        return 1
    return -1


def tournament(population: list[Solution], tournament_size: int) -> Solution:
    participants = np.random.choice(population, size=(tournament_size,), replace=False)  # type: ignore[arg-type]
    best: Solution | None = None

    for participant in participants:
        if best is None or crowding_operator(participant, best) == 1:
            best = participant

    assert best is not None
    return best


def tournament_selection(population: list[Solution], tournament_size: int) -> list[tuple[Solution, Solution]]:
    parents: list[tuple[Solution, Solution]] = []
    while len(parents) < len(population) // 2:
        p1 = tournament(population, tournament_size)
        p2 = tournament(population, tournament_size)
        parents.append((p1, p2))
    return parents
