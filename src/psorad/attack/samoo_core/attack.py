from __future__ import annotations

from operator import attrgetter
from typing import Any

import numpy as np

from .population import Population
from .selection import calculate_crowding_distance, fast_nondominated_sort, tournament_selection
from .solution import Solution
from .variation import generate_offspring


class Attack:
    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.fitness_trace: list[np.ndarray] = []

    def _init_population(self, loss_function: Any) -> Population:
        image = self.params["x"]
        h, w, _ = image.shape
        all_pixels = np.arange(h * w)
        eps = int(self.params["eps"])

        zero_probability = float(self.params["zero_probability"])
        one_prob = (1.0 - zero_probability) / 2.0

        solutions = [
            Solution(
                pixels=np.random.choice(all_pixels, size=(eps,), replace=False),
                values=np.random.choice([-1, 1, 0], size=(eps, 3), p=(one_prob, one_prob, zero_probability)),
                x=image.copy(),
                p_size=float(self.params["p_size"]),
            )
            for _ in range(int(self.params["pop_size"]))
        ]

        population = Population(solutions, loss_function, bool(self.params["include_dist"]))
        population.evaluate()
        return population

    def _save_result(self, population: Population, loss_function: Any, query_count: int, success: bool) -> None:
        assert population.fronts is not None
        front0 = population.fronts[0]

        payload = {
            "front0_imgs": [soln.generate_image() for soln in front0],
            "queries": query_count,
            "true_label": loss_function.true,
            "adversarial_labels": [loss_function.get_label(soln.generate_image()) for soln in front0],
            "front0_fitness": [soln.fitnesses for soln in front0],
            "fitness_process": self.fitness_trace,
            "success": success,
        }
        np.save(self.params["save_directory"], payload, allow_pickle=True)

    def attack(self, loss_function: Any) -> None:
        image = self.params["x"]
        h, w, _ = image.shape
        all_pixels = np.arange(h * w)

        population = self._init_population(loss_function)
        query_count = len(population.population)

        for _ in range(1, int(self.params["iterations"])):
            population.fronts = fast_nondominated_sort(population.population)
            feasible = population.find_adv_solns(float(self.params["max_dist"]))

            best = min(population.population, key=attrgetter("loss")).fitnesses
            self.fitness_trace.append(best)

            if feasible:
                self._save_result(population, loss_function, query_count, success=True)
                return

            for front in population.fronts:
                calculate_crowding_distance(front)

            parents = tournament_selection(population.population, int(self.params["tournament_size"]))
            children = generate_offspring(
                parents=parents,
                pc=float(self.params["pc"]),
                pm=float(self.params["pm"]),
                all_pixels=all_pixels,
                zero_prob=float(self.params["zero_probability"]),
            )

            child_population = Population(children, loss_function, bool(self.params["include_dist"]))
            child_population.evaluate()
            query_count += len(child_population.population)

            merged = population.population + child_population.population
            fronts = fast_nondominated_sort(merged)

            next_generation: list[Solution] = []
            front_idx = 0
            pop_size = int(self.params["pop_size"])

            while front_idx < len(fronts) and len(next_generation) + len(fronts[front_idx]) <= pop_size:
                calculate_crowding_distance(fronts[front_idx])
                next_generation.extend(fronts[front_idx])
                front_idx += 1

            if front_idx < len(fronts) and len(next_generation) < pop_size:
                calculate_crowding_distance(fronts[front_idx])
                fronts[front_idx].sort(key=attrgetter("crowding_distance"), reverse=True)
                remain = pop_size - len(next_generation)
                next_generation.extend(fronts[front_idx][:remain])

            population = Population(next_generation, loss_function, bool(self.params["include_dist"]))

        population.fronts = fast_nondominated_sort(population.population)
        last_best = min(population.population, key=attrgetter("loss")).fitnesses
        self.fitness_trace.append(last_best)
        self._save_result(population, loss_function, query_count, success=False)
