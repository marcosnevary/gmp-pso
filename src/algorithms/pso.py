import random


class PSO:
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
        random.seed(seed)

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
        for _ in range(self.num_particles):
            position = [
                random.uniform(lower, upper) for _ in range(self.num_dimensions)
            ]
            velocity = [random.uniform(-1, 1) for _ in range(self.num_dimensions)]

            self.positions.append(position)
            self.velocities.append(velocity)

        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = [
            self.fitness_function(position) for position in self.positions
        ]

        best_index = self.personal_best_fitness.index(min(self.personal_best_fitness))
        self.global_best_position = self.personal_best_positions[best_index].copy()
        self.global_best_fitness = self.personal_best_fitness[best_index]

    def _update_velocities(self) -> None:
        for i in range(self.num_particles):
            new_velocity = []
            r1 = random.random()
            r2 = random.random()
            for d in range(self.num_dimensions):
                inertia = self.inertia_weight * self.velocities[i][d]
                cognitive = (
                    self.cognitive_coefficient
                    * r1
                    * (self.personal_best_positions[i][d] - self.positions[i][d])
                )
                social = (
                    self.social_coefficient
                    * r2
                    * (self.global_best_position[d] - self.positions[i][d])
                )
                new_velocity.append(inertia + cognitive + social)

            self.velocities[i] = new_velocity

    def _update_positions(self) -> None:
        lower, upper = self.search_space_bounds
        for i in range(self.num_particles):
            new_position = []
            for d in range(self.num_dimensions):
                position = self.positions[i][d] + self.velocities[i][d]
                position = max(lower, min(position, upper))
                new_position.append(position)

            self.positions[i] = new_position

    def _update_personal_global_best(self) -> None:
        for i in range(self.num_particles):
            current_fitness = self.fitness_function(self.positions[i])

            if current_fitness < self.personal_best_fitness[i]:
                self.personal_best_positions[i] = self.positions[i].copy()
                self.personal_best_fitness[i] = current_fitness

            if current_fitness < self.global_best_fitness:
                self.global_best_position = self.positions[i].copy()
                self.global_best_fitness = current_fitness

    def optimize(self) -> tuple:
        self._initialize_particles()
        for _ in range(self.max_iterations):
            self._update_velocities()
            self._update_positions()
            self._update_personal_global_best()

        return self.global_best_position, self.global_best_fitness
