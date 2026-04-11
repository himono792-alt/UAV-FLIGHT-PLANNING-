"""
Genetic Algorithm — Tìm đường bay tối ưu toàn cục bằng tiến hóa quần thể
Lý thuyết: Chương 4 (GA, population-based search, selection, crossover, mutation)
"""

import random
import time
from typing import List, Optional, Tuple

from src.optimizer.simulated_annealing import OptimizeResult


class GeneticAlgorithm:
    """
    Genetic Algorithm theo Ch4: Population-based search

    Pseudocode (Ch4):
        population ← init_population()
        for gen in generations:
            fitness_scores ← [fitness(p) for p in population]
            new_pop ← elitism(population, elite_size)
            while len(new_pop) < pop_size:
                p1 ← tournament_select(population, fitness_scores)
                p2 ← tournament_select(population, fitness_scores)
                child ← crossover(p1, p2)
                child ← mutate(child)
                new_pop.append(child)
            population ← new_pop
        return best individual

    Chromosome = list of (x,y,z) tuples = đường bay từ start đến goal

    Tham số:
        pop_size   : kích thước quần thể
        generations: số thế hệ tối đa
        mutation_rate  : xác suất đột biến 1 waypoint
        crossover_rate : xác suất áp dụng crossover (thay vì copy p1)
        elite_size : số cá thể tốt nhất giữ lại nguyên vẹn
    """

    def __init__(self, grid,
                 start: tuple, goal: tuple,
                 pop_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 seed: int = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evolve(self, seed_path: Optional[list] = None) -> OptimizeResult:
        """
        Chạy GA để tìm path tối ưu.

        Args:
            seed_path: path từ A* để seed quần thể ban đầu (tùy chọn)

        Returns:
            OptimizeResult với path tốt nhất tìm được
        """
        t0 = time.perf_counter()

        # Khởi tạo quần thể
        population = self._init_population(seed_path)

        if not population:
            # Không tạo được cá thể hợp lệ nào
            cost = float('inf')
            return OptimizeResult(
                path=[], cost=cost, initial_cost=cost,
                improvement=0.0, iterations=0,
                time_ms=(time.perf_counter() - t0) * 1000,
                algorithm="GA"
            )

        # Tính fitness quần thể ban đầu
        fitness_scores = [self._fitness(p) for p in population]
        best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
        best_path = list(population[best_idx])
        best_cost = self._path_cost(best_path)
        initial_cost = best_cost

        for gen in range(self.generations):
            fitness_scores = [self._fitness(p) for p in population]

            # Cập nhật best
            idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            c = self._path_cost(population[idx])
            if c < best_cost:
                best_cost = c
                best_path = list(population[idx])

            # Elitism: giữ top elite_size cá thể
            sorted_pop = sorted(
                zip(fitness_scores, population),
                key=lambda x: x[0], reverse=True
            )
            new_pop = [list(p) for _, p in sorted_pop[:self.elite_size]]

            # Điền đủ pop_size bằng crossover + mutation
            attempts = 0
            max_attempts = self.pop_size * 10
            while len(new_pop) < self.pop_size and attempts < max_attempts:
                attempts += 1
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)

                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = list(p1)

                child = self._mutate(child)

                if child and self._is_valid_path(child):
                    new_pop.append(child)

            # Nếu không điền đủ, lấy lại từ elites
            while len(new_pop) < self.pop_size:
                new_pop.append(list(random.choice(new_pop[:self.elite_size])))

            population = new_pop

        improvement = (initial_cost - best_cost) / initial_cost * 100 \
                      if initial_cost > 0 else 0.0

        return OptimizeResult(
            path=best_path,
            cost=best_cost,
            initial_cost=initial_cost,
            improvement=improvement,
            iterations=self.generations,
            time_ms=(time.perf_counter() - t0) * 1000,
            algorithm="GA"
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_population(self, seed_path: Optional[list] = None) -> list:
        """
        Tạo quần thể ban đầu:
        - Slot 0: seed_path (A* result) nếu có
        - Phần còn lại: random walk từ start đến goal
        """
        population = []

        # Seed từ A* nếu có
        if seed_path and len(seed_path) >= 2:
            population.append(list(seed_path))

        # Random walk để đa dạng quần thể
        attempts = 0
        max_attempts = self.pop_size * 20
        while len(population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            path = self._random_walk()
            if path:
                population.append(path)

        return population

    def _random_walk(self, max_steps: int = None) -> Optional[list]:
        """
        Tạo path ngẫu nhiên từ start đến goal bằng random walk.
        Dùng DFS có random shuffle neighbors + giới hạn bước.
        """
        if max_steps is None:
            w, h, d = self.grid.width, self.grid.height, self.grid.depth
            max_steps = (w + h + d) * 3

        stack = [(self.start, [self.start])]
        visited_global = set()

        while stack:
            current, path = stack.pop()

            if current == self.goal:
                return path

            if len(path) > max_steps:
                continue

            if current in visited_global:
                continue
            visited_global.add(current)

            neighbors = list(self.grid.get_neighbors(current))
            random.shuffle(neighbors)

            for nb in neighbors:
                if nb not in visited_global:
                    stack.append((nb, path + [nb]))

        return None

    def _fitness(self, path: list) -> float:
        """
        Fitness = 1 / (distance + collision_penalty)
        Fitness cao = path tốt (ngắn, không qua obstacle).
        """
        if not path or len(path) < 2:
            return 0.0

        distance = self._path_cost(path)
        collision_penalty = 0.0

        for pos in path:
            if not self.grid.is_free(pos):
                collision_penalty += 1000.0

        total = distance + collision_penalty
        if total <= 0:
            return 1e9
        return 1.0 / total

    def _tournament_select(self, population: list,
                           fitness_scores: list, k: int = 3) -> list:
        """Chọn k cá thể ngẫu nhiên, trả về cá thể có fitness cao nhất."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best = max(indices, key=lambda i: fitness_scores[i])
        return list(population[best])

    def _crossover(self, parent1: list, parent2: list) -> list:
        """
        Single-point crossover theo Ch4:
        - Tìm điểm chung (waypoint xuất hiện trong cả 2 path)
        - Ghép nửa đầu parent1 đến điểm chung + nửa sau parent2 từ điểm chung
        - Fallback: trả về bản copy của parent1
        """
        # Tìm tập điểm chung (bỏ start và goal)
        set1 = set(tuple(p) for p in parent1[1:-1])
        set2 = set(tuple(p) for p in parent2[1:-1])
        common = set1 & set2

        if not common:
            # Không có điểm chung → trả về copy parent1
            return list(parent1)

        # Chọn ngẫu nhiên 1 điểm chung
        crossover_point = random.choice(list(common))

        # Tìm index của crossover_point trong mỗi parent
        idx1 = next((i for i, p in enumerate(parent1) if tuple(p) == crossover_point), None)
        idx2 = next((i for i, p in enumerate(parent2) if tuple(p) == crossover_point), None)

        if idx1 is None or idx2 is None:
            return list(parent1)

        child = parent1[:idx1 + 1] + parent2[idx2 + 1:]
        return child

    def _mutate(self, path: list) -> list:
        """
        Mutation: với xác suất mutation_rate, chọn 1 waypoint giữa ngẫu nhiên
        và thay bằng ô kề hợp lệ.
        """
        if len(path) <= 2 or random.random() >= self.mutation_rate:
            return path

        path = list(path)
        idx = random.randint(1, len(path) - 2)
        neighbors = list(self.grid.get_neighbors(path[idx]))

        if not neighbors:
            return path

        random.shuffle(neighbors)
        for new_wp in neighbors:
            # Kiểm tra connectivity với waypoint trước và sau
            if self._is_connected(path[idx - 1], new_wp) and \
               self._is_connected(new_wp, path[idx + 1]):
                path[idx] = new_wp
                break

        return path

    def _path_cost(self, path: list) -> float:
        """Tổng step_cost trên toàn path"""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.grid.step_cost(path[i], path[i + 1])
        return total

    def _is_valid_path(self, path: list) -> bool:
        """
        Kiểm tra path hợp lệ:
        - Bắt đầu từ start, kết thúc tại goal
        - Mọi ô đều is_free
        - Các ô liên tiếp kề nhau
        """
        if not path or len(path) < 1:
            return False
        if tuple(path[0]) != tuple(self.start):
            return False
        if tuple(path[-1]) != tuple(self.goal):
            return False
        for pos in path:
            if not self.grid.is_free(pos):
                return False
        for i in range(len(path) - 1):
            if not self._is_connected(path[i], path[i + 1]):
                return False
        return True

    def _is_connected(self, a: tuple, b: tuple) -> bool:
        """Kiểm tra 2 ô kề nhau (6-direction, Manhattan distance = 1)"""
        return sum(abs(b[k] - a[k]) for k in range(3)) == 1
