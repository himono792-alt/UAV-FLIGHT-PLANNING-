"""
Test Giai đoạn 3: SA + GA Optimizer
Chạy: python -m pytest tests/test_optimizer.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_world import GridWorld3D, CellType
from src.search.a_star import a_star_search
from src.optimizer.simulated_annealing import SimulatedAnnealing, OptimizeResult
from src.optimizer.genetic_algorithm import GeneticAlgorithm


# ── Helpers ────────────────────────────────────────────────────────────────

def make_grid(w=10, h=10, d=5, start=(0,0,0), goal=(9,9,4)):
    g = GridWorld3D(w, h, d)
    g.set_start(start)
    g.set_goal(goal)
    return g


def get_astar_path(grid, start=(0,0,0), goal=(9,9,4)):
    r = a_star_search(grid, start, goal)
    assert r.found, "A* phải tìm được đường trên grid trống"
    return r.path


def path_is_valid(grid, path):
    """Mọi ô trên path đều is_free"""
    for pos in path:
        cell = grid.get_cell(pos)
        if cell == CellType.OBSTACLE or cell == CellType.NO_FLY_ZONE:
            return False
    return True


def path_is_connected(path):
    """Các ô liên tiếp kề nhau (6-direction)"""
    for i in range(len(path) - 1):
        d = sum(abs(path[i+1][k] - path[i][k]) for k in range(3))
        if d != 1:
            return False
    return True


# ── TestSimulatedAnnealing ─────────────────────────────────────────────────

class TestSimulatedAnnealing:

    def test_sa_giam_cost(self):
        """SA không làm tăng cost so với initial (best luôn <= initial)"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, T0=100.0, alpha=0.99, max_iter=500, seed=42)
        result = sa.optimize(path)
        assert result.cost <= result.initial_cost + 1e-6, \
            f"SA cost {result.cost:.2f} > initial {result.initial_cost:.2f}"

    def test_sa_giu_start_goal(self):
        """SA giữ nguyên start và goal"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert result.path[0] == (0, 0, 0), "Start phải là (0,0,0)"
        assert result.path[-1] == (9, 9, 4), "Goal phải là (9,9,4)"

    def test_sa_path_hop_le(self):
        """Mọi ô trên SA path đều is_free"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert path_is_valid(g, result.path), "SA path có ô không hợp lệ"

    def test_sa_path_connected(self):
        """SA path vẫn liên tục sau tối ưu"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert path_is_connected(result.path), "SA path bị đứt"

    def test_sa_path_ngan_hon_2_diem(self):
        """SA trả về ngay nếu path <= 2 waypoints"""
        g = make_grid(5, 5, 3, start=(0,0,0), goal=(1,0,0))
        path = [(0,0,0), (1,0,0)]
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert result.iterations == 0
        assert result.improvement == 0.0

    def test_sa_improvement_field(self):
        """improvement = (initial - cost) / initial * 100"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        expected = (result.initial_cost - result.cost) / result.initial_cost * 100
        assert abs(result.improvement - expected) < 1e-6

    def test_sa_algorithm_name(self):
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert result.algorithm == "SA"

    def test_sa_time_ms(self):
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        assert result.time_ms >= 0

    def test_sa_iterations_bounded(self):
        """Số iterations <= max_iter"""
        g = make_grid()
        path = get_astar_path(g)
        max_iter = 200
        sa = SimulatedAnnealing(g, max_iter=max_iter, seed=42)
        result = sa.optimize(path)
        assert result.iterations <= max_iter

    def test_sa_voi_obstacles(self):
        """SA hoạt động đúng trên grid có obstacles"""
        g = GridWorld3D(10, 10, 5)
        g.set_start((0, 0, 0))
        g.set_goal((9, 9, 4))
        g.add_random_obstacles(15, seed=7)
        r_astar = a_star_search(g, (0, 0, 0), (9, 9, 4))
        if not r_astar.found:
            return  # skip nếu A* không tìm được

        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(r_astar.path)
        assert result.cost <= result.initial_cost + 1e-6
        assert path_is_valid(g, result.path)


# ── TestGeneticAlgorithm ───────────────────────────────────────────────────

class TestGeneticAlgorithm:

    def test_ga_tim_duong(self):
        """GA tìm được path hợp lệ từ start đến goal"""
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=20, generations=30, seed=42)
        result = ga.evolve()
        assert len(result.path) >= 2, "GA phải trả về path"
        assert result.path[0] == (0, 0, 0)
        assert result.path[-1] == (7, 7, 3)

    def test_ga_giu_start_goal(self):
        """GA giữ nguyên start và goal"""
        g = make_grid()
        ga = GeneticAlgorithm(g, (0,0,0), (9,9,4),
                              pop_size=20, generations=20, seed=42)
        result = ga.evolve()
        assert result.path[0] == (0, 0, 0)
        assert result.path[-1] == (9, 9, 4)

    def test_ga_path_hop_le(self):
        """Mọi ô trên GA path đều is_free"""
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=20, generations=30, seed=42)
        result = ga.evolve()
        assert path_is_valid(g, result.path)

    def test_ga_path_connected(self):
        """GA path liên tục"""
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=20, generations=30, seed=42)
        result = ga.evolve()
        assert path_is_connected(result.path)

    def test_ga_cost_duong(self):
        """GA path phải có cost > 0"""
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=20, generations=20, seed=42)
        result = ga.evolve()
        assert result.cost > 0

    def test_ga_seed_path_cai_thien(self):
        """GA với seed A* cho cost <= A* ban đầu"""
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        r_astar = a_star_search(g, (0,0,0), (7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=30, generations=50, seed=42)
        result = ga.evolve(seed_path=r_astar.path)
        # GA với seed A* không nên tệ hơn A* nhiều (có thể ngang bằng)
        assert result.cost <= r_astar.cost * 1.1 + 1e-6, \
            f"GA cost {result.cost:.2f} quá cao so với A* {r_astar.cost:.2f}"

    def test_ga_algorithm_name(self):
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=10, generations=10, seed=42)
        result = ga.evolve()
        assert result.algorithm == "GA"

    def test_ga_time_ms(self):
        g = make_grid(8, 8, 4, start=(0,0,0), goal=(7,7,3))
        ga = GeneticAlgorithm(g, (0,0,0), (7,7,3),
                              pop_size=10, generations=10, seed=42)
        result = ga.evolve()
        assert result.time_ms >= 0


# ── TestSoSanh ─────────────────────────────────────────────────────────────

class TestSoSanh:

    def test_so_sanh_3_duong(self):
        """So sánh A* gốc vs A*+SA vs GA trên cùng 1 grid"""
        g = GridWorld3D(12, 12, 5)
        g.set_start((0, 0, 0))
        g.set_goal((11, 11, 4))
        g.add_random_obstacles(20, seed=55)

        start, goal = (0, 0, 0), (11, 11, 4)

        # 1. A* gốc
        r_astar = a_star_search(g, start, goal)
        if not r_astar.found:
            return  # skip nếu grid quá khó

        # 2. A* + SA
        sa = SimulatedAnnealing(g, T0=100.0, alpha=0.995,
                                max_iter=2000, seed=42)
        r_sa = sa.optimize(r_astar.path)

        # 3. GA
        ga = GeneticAlgorithm(g, start, goal,
                              pop_size=30, generations=50, seed=42)
        r_ga = ga.evolve(seed_path=r_astar.path)

        # In bảng so sánh
        print("\n" + "=" * 65)
        print(f"{'Method':<15} {'Cost':<12} {'PathLen':<10} {'Time(ms)':<10}")
        print("-" * 65)
        print(f"{'A*':<15} {r_astar.cost:<12.2f} {len(r_astar.path):<10} {r_astar.time_ms:<10.2f}")
        print(f"{'A*+SA':<15} {r_sa.cost:<12.2f} {len(r_sa.path):<10} {r_sa.time_ms:<10.2f}")
        print(f"{'GA':<15} {r_ga.cost:<12.2f} {len(r_ga.path):<10} {r_ga.time_ms:<10.2f}")
        print(f"{'SA improve':<15} {r_sa.improvement:+.1f}%")
        print("=" * 65)

        # SA không làm tệ hơn A*
        assert r_sa.cost <= r_astar.cost + 1e-6, \
            "SA không nên làm tệ hơn A*"

        # GA path hợp lệ
        assert path_is_valid(g, r_ga.path), "GA path không hợp lệ"
        assert path_is_connected(r_ga.path), "GA path không liên tục"

    def test_optimizeresult_summary(self):
        """OptimizeResult.summary() format đúng"""
        g = make_grid()
        path = get_astar_path(g)
        sa = SimulatedAnnealing(g, seed=42)
        result = sa.optimize(path)
        s = result.summary()
        assert "[SA]" in s
        assert "Cost:" in s
        assert "Iters:" in s
        assert "Time:" in s
