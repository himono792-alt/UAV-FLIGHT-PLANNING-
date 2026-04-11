"""
Simulated Annealing — Tối ưu đường bay bằng Local Search
Lý thuyết: Chương 4 (Local Search, SA schedule, chấp nhận giải xấu)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimizeResult:
    """Kết quả trả về từ thuật toán tối ưu hóa"""
    path: list           # đường bay tối ưu [(x,y,z), ...]
    cost: float          # chi phí đường bay
    initial_cost: float  # chi phí ban đầu (trước khi tối ưu)
    improvement: float   # (initial_cost - cost) / initial_cost * 100 (%)
    iterations: int      # số vòng lặp đã chạy
    time_ms: float       # thời gian chạy (ms)
    algorithm: str

    def summary(self) -> str:
        return (
            f"[{self.algorithm}] "
            f"Cost: {self.initial_cost:.2f} → {self.cost:.2f} "
            f"({self.improvement:+.1f}%) | "
            f"Iters: {self.iterations} | "
            f"Time: {self.time_ms:.1f}ms"
        )


class SimulatedAnnealing:
    """
    Simulated Annealing theo Ch4: Local Search chấp nhận giải xấu hơn
    để thoát local minima.

    Pseudocode (Ch4):
        current ← initial_state
        T ← T0
        loop:
            next ← random_neighbor(current)
            ΔE ← VALUE(next) - VALUE(current)
            if ΔE > 0: current ← next          # tốt hơn → luôn chấp nhận
            else with prob e^(ΔE/T): current ← next  # xấu hơn → đôi khi chấp nhận
            T ← schedule(T)
            if T = 0: return current

    Tham số:
        T0    : nhiệt độ ban đầu cao → chấp nhận nhiều giải xấu
        alpha : hệ số giảm nhiệt ∈ (0,1) → T = T0 * alpha^t
        min_T : ngưỡng dừng
    """

    def __init__(self, grid,
                 T0: float = 100.0,
                 alpha: float = 0.995,
                 min_temp: float = 0.01,
                 max_iter: int = 5000,
                 seed: int = None):
        self.grid = grid
        self.T0 = T0
        self.alpha = alpha
        self.min_temp = min_temp
        self.max_iter = max_iter
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def optimize(self, path: list) -> OptimizeResult:
        """
        Tối ưu path bằng SA.

        Input:  path = [(x,y,z), ...] từ A* (hoặc bất kỳ)
        Output: OptimizeResult với path được cải thiện
        """
        t0 = time.perf_counter()

        if len(path) <= 2:
            # Start == Goal hoặc path chỉ có 2 điểm → không cần tối ưu
            cost = self._path_cost(path)
            return OptimizeResult(
                path=path, cost=cost, initial_cost=cost,
                improvement=0.0, iterations=0,
                time_ms=0.0, algorithm="SA"
            )

        initial_cost = self._path_cost(path)
        current = list(path)
        best = list(path)
        best_cost = initial_cost
        T = self.T0
        iters = 0

        while T > self.min_temp and iters < self.max_iter:
            # Tạo neighbor bằng cách perturb
            neighbor = self._perturb(current)

            if neighbor is not None:
                current_cost = self._path_cost(current)
                neighbor_cost = self._path_cost(neighbor)
                delta = neighbor_cost - current_cost  # delta < 0 = tốt hơn

                # Ch4: Chấp nhận neighbor
                if delta < 0:
                    current = neighbor           # tốt hơn → luôn chấp nhận
                elif T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:   # xấu hơn → chấp nhận với xác suất
                        current = neighbor

                # Cập nhật best
                current_cost_now = self._path_cost(current)
                if current_cost_now < best_cost:
                    best = list(current)
                    best_cost = current_cost_now

            # Schedule giảm nhiệt: T = T0 * alpha^t
            T *= self.alpha
            iters += 1

        improvement = (initial_cost - best_cost) / initial_cost * 100 if initial_cost > 0 else 0.0

        return OptimizeResult(
            path=best,
            cost=best_cost,
            initial_cost=initial_cost,
            improvement=improvement,
            iterations=iters,
            time_ms=(time.perf_counter() - t0) * 1000,
            algorithm="SA"
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _perturb(self, path: list) -> Optional[list]:
        """
        Tạo neighbor bằng cách dịch chuyển 1 waypoint ở giữa.
        Giữ nguyên path[0] (start) và path[-1] (goal).
        Trả về None nếu không tìm được neighbor hợp lệ.
        """
        if len(path) <= 2:
            return None

        # Chọn ngẫu nhiên 1 waypoint ở giữa (bỏ start và goal)
        idx = random.randint(1, len(path) - 2)
        current_wp = path[idx]

        # Lấy danh sách ô kề hợp lệ của waypoint đó
        neighbors = self.grid.get_neighbors(current_wp)
        if not neighbors:
            return None

        # Thử từng ô kề ngẫu nhiên
        random.shuffle(neighbors)
        for new_wp in neighbors:
            new_path = list(path)
            new_path[idx] = new_wp

            # Kiểm tra path vẫn liên tục sau khi dịch
            if self._is_connected(new_path[idx - 1], new_wp) and \
               self._is_connected(new_wp, new_path[idx + 1]):
                return new_path

        return None

    def _path_cost(self, path: list) -> float:
        """Tổng step_cost trên toàn path"""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.grid.step_cost(path[i], path[i + 1])
        return total

    def _is_valid_path(self, path: list) -> bool:
        """
        Kiểm tra path hợp lệ:
        - Mọi ô đều is_free
        - Các ô liên tiếp kề nhau (Manhattan distance = 1)
        """
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
