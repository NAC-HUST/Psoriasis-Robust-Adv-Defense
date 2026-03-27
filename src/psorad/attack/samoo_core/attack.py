from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from typing import Any

import numpy as np

from .population import Population
from .selection import calculate_crowding_distance, fast_nondominated_sort, tournament_selection
from .solution import Solution
from .variation import generate_offspring


@dataclass
class AttackParams:
    x: np.ndarray
    eps: int
    iterations: int
    pc: float
    pm: float
    pop_size: int
    zero_probability: float
    include_dist: bool
    max_dist: float
    p_size: float
    tournament_size: int
    save_directory: str

    def __post_init__(self) -> None:
        if self.x.ndim != 3 or self.x.shape[2] != 3:
            raise ValueError("x 必须是 HWC 且通道数为 3 的图像。")
        if self.eps <= 0:
            raise ValueError("eps 必须大于 0。")
        if self.iterations <= 0:
            raise ValueError("iterations 必须大于 0。")
        if self.pop_size < 2:
            raise ValueError("pop_size 必须至少为 2。")
        if not (0.0 <= self.pc <= 1.0):
            raise ValueError("pc 必须在 [0, 1]。")
        if not (0.0 <= self.pm <= 1.0):
            raise ValueError("pm 必须在 [0, 1]。")
        if not (0.0 <= self.zero_probability <= 1.0):
            raise ValueError("zero_probability 必须在 [0, 1]。")
        if self.max_dist < 0.0:
            raise ValueError("max_dist 不能小于 0。")
        if self.p_size <= 0.0:
            raise ValueError("p_size 必须大于 0。")
        if self.tournament_size < 2:
            raise ValueError("tournament_size 必须至少为 2。")
        if self.tournament_size > self.pop_size:
            raise ValueError("tournament_size 不能大于 pop_size。")

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> AttackParams:
        return cls(
            x=np.asarray(params["x"], dtype=np.float32),
            eps=int(params["eps"]),
            iterations=int(params["iterations"]),
            pc=float(params["pc"]),
            pm=float(params["pm"]),
            pop_size=int(params["pop_size"]),
            zero_probability=float(params["zero_probability"]),
            include_dist=bool(params["include_dist"]),
            max_dist=float(params["max_dist"]),
            p_size=float(params["p_size"]),
            tournament_size=int(params["tournament_size"]),
            save_directory=str(params["save_directory"]),
        )


class Attack:
    def __init__(
        self,
        params: AttackParams | dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval: int | None = None,
    ):
        self.params = params if isinstance(params, AttackParams) else AttackParams.from_dict(params)
        self.fitness_trace: list[np.ndarray] = []
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval

    def _emit(self, event: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

    def _init_population(self, loss_function: Any) -> Population:
        image = self.params.x
        h, w, _ = image.shape
        all_pixels = np.arange(h * w)
        eps = int(self.params.eps)

        self._emit(
            {
                "phase": "init_population_start",
                "height": int(h),
                "width": int(w),
                "eps": eps,
                "pop_size": int(self.params.pop_size),
                "zero_probability": float(self.params.zero_probability),
                "p_size": float(self.params.p_size),
            }
        )

        if eps > len(all_pixels):
            raise ValueError(f"eps={eps} 超过图像像素数 {len(all_pixels)}。")

        zero_probability = float(self.params.zero_probability)
        one_prob = (1.0 - zero_probability) / 2.0

        solutions = [
            Solution(
                pixels=np.random.choice(all_pixels, size=(eps,), replace=False),
                values=np.random.choice([-1, 1, 0], size=(eps, 3), p=(one_prob, one_prob, zero_probability)),
                x=image.copy(),
                p_size=float(self.params.p_size),
            )
            for _ in range(int(self.params.pop_size))
        ]

        population = Population(solutions, loss_function, bool(self.params.include_dist))
        population.evaluate()
        self._emit(
            {
                "phase": "init_population_done",
                "population_size": len(population.population),
            }
        )
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
        np.save(self.params.save_directory, payload, allow_pickle=True)  # type: ignore[arg-type]

    def attack(self, loss_function: Any) -> None:
        image = self.params.x
        h, w, _ = image.shape
        all_pixels = np.arange(h * w)

        population = self._init_population(loss_function)
        query_count = len(population.population)
        total_iterations = int(self.params.iterations)
        interval = self.progress_interval if self.progress_interval is not None else max(1, total_iterations // 20)

        self._emit(
            {
                "phase": "attack_start",
                "iterations": total_iterations,
                "pc": float(self.params.pc),
                "pm": float(self.params.pm),
                "tournament_size": int(self.params.tournament_size),
                "include_dist": bool(self.params.include_dist),
                "max_dist": float(self.params.max_dist),
                "query_count": int(query_count),
            }
        )

        for iteration in range(1, total_iterations):
            population.fronts = fast_nondominated_sort(population.population)
            feasible = population.find_adv_solns(float(self.params.max_dist))

            best = min(population.population, key=attrgetter("loss")).fitnesses
            self.fitness_trace.append(best)

            should_emit = (iteration == 1) or (iteration % interval == 0) or (iteration == total_iterations - 1)
            if should_emit:
                self._emit(
                    {
                        "phase": "iteration",
                        "iteration": int(iteration),
                        "total_iterations": total_iterations,
                        "query_count": int(query_count),
                        "feasible_count": len(feasible),
                        "best_loss": float(best[0]) if len(best) > 0 else 0.0,
                    }
                )

            if feasible:
                self._emit(
                    {
                        "phase": "early_success",
                        "iteration": int(iteration),
                        "query_count": int(query_count),
                        "feasible_count": len(feasible),
                    }
                )
                self._save_result(population, loss_function, query_count, success=True)
                return

            for front in population.fronts:
                calculate_crowding_distance(front)

            parents = tournament_selection(population.population, int(self.params.tournament_size))
            children = generate_offspring(
                parents=parents,
                pc=float(self.params.pc),
                pm=float(self.params.pm),
                all_pixels=all_pixels,
                zero_prob=float(self.params.zero_probability),
            )

            child_population = Population(children, loss_function, bool(self.params.include_dist))
            child_population.evaluate()
            query_count += len(child_population.population)

            if should_emit:
                self._emit(
                    {
                        "phase": "generation_operators",
                        "iteration": int(iteration),
                        "parents_pairs": len(parents),
                        "children": len(children),
                        "post_query_count": int(query_count),
                    }
                )

            merged = population.population + child_population.population
            fronts = fast_nondominated_sort(merged)

            next_generation: list[Solution] = []
            front_idx = 0
            pop_size = int(self.params.pop_size)

            while front_idx < len(fronts) and len(next_generation) + len(fronts[front_idx]) <= pop_size:
                calculate_crowding_distance(fronts[front_idx])
                next_generation.extend(fronts[front_idx])
                front_idx += 1

            if front_idx < len(fronts) and len(next_generation) < pop_size:
                calculate_crowding_distance(fronts[front_idx])
                fronts[front_idx].sort(key=attrgetter("crowding_distance"), reverse=True)
                remain = pop_size - len(next_generation)
                next_generation.extend(fronts[front_idx][:remain])

            population = Population(next_generation, loss_function, bool(self.params.include_dist))

        population.fronts = fast_nondominated_sort(population.population)
        last_best = min(population.population, key=attrgetter("loss")).fitnesses
        self.fitness_trace.append(last_best)
        self._emit(
            {
                "phase": "attack_end",
                "success": False,
                "query_count": int(query_count),
                "best_loss": float(last_best[0]) if len(last_best) > 0 else 0.0,
            }
        )
        self._save_result(population, loss_function, query_count, success=False)
