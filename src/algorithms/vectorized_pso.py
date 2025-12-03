import numpy as np


class VectorizedPSO:
    def __init__(
        self,
        fitness_function: callable,
        search_space_bounds: tuple,
        seed: int,
        num_dimensions: int,
        num_particles: int,
        max_iterations: int,
        cognitive_coefficient: float,
        social_coefficient: float,
        inertia_weight: float,
    ) -> None:
        self.fitness_function = fitness_function
        self.search_space_bounds = search_space_bounds
        self.rng = np.random.default_rng(seed)

        self.num_dimensions = num_dimensions
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.inertia_weight = inertia_weight

        self.positions = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_fitness = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def _initialize_particles(self) -> None:
        lower, upper = self.search_space_bounds

        self.positions = self.rng.uniform(
            low=lower,
            high=upper,
            size=(self.num_particles, self.num_dimensions),
        )

        self.velocities = self.rng.uniform(
            low=-1,
            high=1,
            size=(self.num_particles, self.num_dimensions),
        )

        self.personal_best_positions = self.positions.copy()

        self.personal_best_fitness = np.array(
            [self.fitness_function(pos) for pos in self.positions],
        )

        best_index = np.argmin(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[best_index].copy()
        self.global_best_fitness = self.personal_best_fitness[best_index]

    def _update_velocities(self) -> None:
        r1 = self.rng.random(size=(self.num_particles, 1))
        r2 = self.rng.random(size=(self.num_particles, 1))

        inertia = self.inertia_weight * self.velocities

        cognitive = (
            self.cognitive_coefficient
            * r1
            * (self.personal_best_positions - self.positions)
        )

        social = (
            self.social_coefficient * r2 * (self.global_best_position - self.positions)
        )

        self.velocities = inertia + cognitive + social

    def _update_positions(self) -> None:
        lower, upper = self.search_space_bounds

        self.positions = self.positions + self.velocities
        self.positions = np.clip(self.positions, lower, upper)

    def _update_personal_global_best(self) -> None:
        current_fitness = np.array(
            [self.fitness_function(pos) for pos in self.positions],
        )

        improved = current_fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved].copy()
        self.personal_best_fitness[improved] = current_fitness[improved]

        best_index = np.argmin(current_fitness)
        if current_fitness[best_index] < self.global_best_fitness:
            self.global_best_position = self.positions[best_index].copy()
            self.global_best_fitness = current_fitness[best_index]

    def optimize(self) -> tuple:
        self._initialize_particles()
        for _ in range(self.max_iterations):
            self._update_velocities()
            self._update_positions()
            self._update_personal_global_best()

        return self.global_best_position, self.global_best_fitness
