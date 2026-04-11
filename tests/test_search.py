"""
Test Giai đoạn 2: A* Search + Heuristic
Chạy: python -m pytest tests/test_search.py -v
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_world import GridWorld3D, CellType
from src.search.graph_search import bfs, dfs, SearchResult
from src.search.heuristics import (
    euclidean_3d, manhattan_3d, wind_adjusted, energy_based, zero_heuristic
)
from src.search.a_star import a_star_search
from src.search.greedy_bfs import greedy_bfs


# ── Helpers ────────────────────────────────────────────────────────────────

def make_grid(w=10, h=10, d=5, start=(0,0,0), goal=(9,9,4)):
    g = GridWorld3D(w, h, d)
    g.set_start(start)
    g.set_goal(goal)
    return g

def make_blocked_grid():
    """Goal bị bao vây hoàn toàn bởi obstacles"""
    g = GridWorld3D(10, 10, 5)
    g.set_start((0, 0, 0))
    g.set_goal((9, 9, 4))
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                x, y, z = 9+dx, 9+dy, 4+dz
                if (x, y, z) != (9, 9, 4) and g.in_bounds((x, y, z)):
                    g.add_obstacle((x, y, z))
    return g

def path_is_valid(grid, path):
    """Kiểm tra mọi ô trên path đều is_free hoặc start/goal"""
    if len(path) < 1:
        return False
    for pos in path:
        cell = grid.get_cell(pos)
        if cell == CellType.OBSTACLE or cell == CellType.NO_FLY_ZONE:
            return False
    return True

def path_is_connected(path):
    """Kiểm tra các ô liên tiếp kề nhau (6-direction)"""
    for i in range(len(path) - 1):
        d = tuple(abs(path[i+1][k] - path[i][k]) for k in range(3))
        if sum(d) != 1:
            return False
    return True


# ── TestGraphSearch ─────────────────────────────────────────────────────────

class TestGraphSearch:

    def test_bfs_tim_duong_co_ban(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert r.found
        assert r.path[0] == (0,0,0)
        assert r.path[-1] == (9,9,4)

    def test_dfs_tim_duong_co_ban(self):
        g = make_grid()
        r = dfs(g, (0,0,0), (9,9,4))
        assert r.found
        assert r.path[0] == (0,0,0)
        assert r.path[-1] == (9,9,4)

    def test_bfs_khong_co_duong(self):
        g = make_blocked_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert not r.found
        assert r.path == []

    def test_bfs_path_hop_le(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert path_is_valid(g, r.path)

    def test_dfs_path_hop_le(self):
        g = make_grid()
        r = dfs(g, (0,0,0), (9,9,4))
        assert path_is_valid(g, r.path)

    def test_bfs_path_connected(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert path_is_connected(r.path)

    def test_start_la_goal(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (0,0,0))
        assert r.found
        assert r.path == [(0,0,0)]
        assert r.cost == 0.0
        assert r.nodes_expanded == 0

    def test_nodes_expanded_positive(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert r.nodes_expanded > 0

    def test_time_ms_positive(self):
        g = make_grid()
        r = bfs(g, (0,0,0), (9,9,4))
        assert r.time_ms >= 0


# ── TestHeuristics ─────────────────────────────────────────────────────────

class TestHeuristics:

    def test_euclidean_chinh_xac(self):
        h = euclidean_3d((0,0,0), (3,4,0))
        assert abs(h - 5.0) < 1e-9

    def test_euclidean_zero_khi_trung(self):
        assert euclidean_3d((5,5,2), (5,5,2)) == 0.0

    def test_manhattan_chinh_xac(self):
        h = manhattan_3d((0,0,0), (3,4,2))
        assert h == 9.0

    def test_manhattan_zero_khi_trung(self):
        assert manhattan_3d((3,3,3), (3,3,3)) == 0.0

    def test_euclidean_admissible(self):
        """h(n) ≤ chi phí thực → A* optimal"""
        g = make_grid(10, 10, 5, (0,0,0), (9,9,4))
        r = a_star_search(g, (0,0,0), (9,9,4), zero_heuristic)  # dijkstra = optimal
        r_euclid = a_star_search(g, (0,0,0), (9,9,4), euclidean_3d)
        # Cả 2 đều tìm được path có cùng optimal cost (admissible)
        assert r.found and r_euclid.found
        assert abs(r.cost - r_euclid.cost) < 1e-6

    def test_manhattan_admissible(self):
        g = make_grid(10, 10, 5, (0,0,0), (9,9,4))
        r_zero = a_star_search(g, (0,0,0), (9,9,4), zero_heuristic)
        r_manh = a_star_search(g, (0,0,0), (9,9,4), manhattan_3d)
        assert r_zero.found and r_manh.found
        assert abs(r_zero.cost - r_manh.cost) < 1e-6

    def test_energy_leo_cao_hon(self):
        """Heuristic energy phải ước lượng cao hơn khi goal ở trên"""
        h_flat  = energy_based((5,5,0), (5,5,0))
        h_climb = energy_based((5,5,0), (5,5,4))
        assert h_climb > h_flat

    def test_wind_adjusted_khong_am(self):
        """Heuristic không trả về giá trị âm"""
        h = wind_adjusted((0,0,0), (9,9,4), wind_vector=(10,0,0))
        assert h >= 0

    def test_zero_heuristic_luon_zero(self):
        for pos in [(0,0,0), (5,5,2), (9,9,4)]:
            assert zero_heuristic(pos, (9,9,4)) == 0.0


# ── TestAStar ──────────────────────────────────────────────────────────────

class TestAStar:

    def test_a_star_tim_duong(self):
        g = make_grid()
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert r.found
        assert r.path[0] == (0,0,0)
        assert r.path[-1] == (9,9,4)

    def test_a_star_optimal_hon_dfs(self):
        """A* cost ≤ DFS cost (A* optimal, DFS không)"""
        g = make_grid()
        r_astar = a_star_search(g, (0,0,0), (9,9,4))
        r_dfs   = dfs(g, (0,0,0), (9,9,4))
        assert r_astar.found and r_dfs.found
        assert r_astar.cost <= r_dfs.cost + 1e-6

    def test_a_star_voi_obstacles(self):
        """A* vẫn tìm được đường khi có obstacles"""
        g = GridWorld3D(10, 10, 5)
        g.set_start((0,0,0))
        g.set_goal((9,0,0))
        # Dựng tường dọc giữa
        for y in range(10):
            for z in range(5):
                g.add_obstacle((5, y, z))
        # Mở 1 lỗ ở z=4 để đi qua
        g.set_cell((5, 0, 4), CellType.FREE)
        r = a_star_search(g, (0,0,0), (9,0,0))
        assert r.found

    def test_a_star_goal_unreachable(self):
        g = make_blocked_grid()  # goal (9,9,4) bị bao vây hoàn toàn
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert not r.found, "Phải trả về not found khi goal bị khóa"

    def test_a_star_start_la_goal(self):
        g = make_grid()
        r = a_star_search(g, (3,3,2), (3,3,2))
        assert r.found
        assert len(r.path) == 1
        assert r.cost == 0.0

    def test_a_star_path_hop_le(self):
        g = make_grid()
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert path_is_valid(g, r.path)

    def test_a_star_path_connected(self):
        g = make_grid()
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert path_is_connected(r.path)

    def test_a_star_algorithm_name(self):
        g = make_grid()
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert r.algorithm == "A*"

    def test_a_star_cost_duong(self):
        g = make_grid()
        r = a_star_search(g, (0,0,0), (9,9,4))
        assert r.cost > 0


# ── TestGreedyBFS ──────────────────────────────────────────────────────────

class TestGreedyBFS:

    def test_greedy_tim_duong(self):
        g = make_grid()
        r = greedy_bfs(g, (0,0,0), (9,9,4))
        assert r.found

    def test_greedy_path_hop_le(self):
        g = make_grid()
        r = greedy_bfs(g, (0,0,0), (9,9,4))
        assert path_is_valid(g, r.path)

    def test_greedy_nhanh_hon_a_star(self):
        """Greedy expand ít node hơn A* (ưu tiên tốc độ)"""
        g = make_grid(20, 20, 10, (0,0,0), (19,19,9))
        r_astar  = a_star_search(g, (0,0,0), (19,19,9))
        r_greedy = greedy_bfs(g, (0,0,0), (19,19,9))
        # Greedy thường expand ít hơn (không phải luôn luôn nhưng với grid trống)
        assert r_greedy.nodes_expanded <= r_astar.nodes_expanded + 1  # slack

    def test_greedy_khong_optimal(self):
        """Greedy cost có thể >= A* cost (không đảm bảo optimal)"""
        # Với grid phức tạp, greedy có thể tệ hơn
        g = make_grid()
        r_astar  = a_star_search(g, (0,0,0), (9,9,4))
        r_greedy = greedy_bfs(g, (0,0,0), (9,9,4))
        # Greedy cost >= A* cost (A* optimal)
        assert r_greedy.cost >= r_astar.cost - 1e-6

    def test_greedy_algorithm_name(self):
        g = make_grid()
        r = greedy_bfs(g, (0,0,0), (9,9,4))
        assert r.algorithm == "Greedy BFS"


# ── TestSoSanh ─────────────────────────────────────────────────────────────

class TestSoSanh:

    def test_bang_so_sanh(self):
        """Chạy tất cả thuật toán, in bảng so sánh"""
        g = GridWorld3D(15, 15, 5)
        g.set_start((0, 0, 0))
        g.set_goal((14, 14, 4))
        g.add_random_obstacles(30, seed=99)

        algorithms = {
            "BFS":              lambda: bfs(g, (0,0,0), (14,14,4)),
            "DFS":              lambda: dfs(g, (0,0,0), (14,14,4)),
            "Greedy(Euclid)":  lambda: greedy_bfs(g, (0,0,0), (14,14,4), euclidean_3d),
            "A*(Euclid)":      lambda: a_star_search(g, (0,0,0), (14,14,4), euclidean_3d),
            "A*(Manhattan)":   lambda: a_star_search(g, (0,0,0), (14,14,4), manhattan_3d),
            "A*(Energy)":      lambda: a_star_search(g, (0,0,0), (14,14,4), energy_based),
            "A*(Zero/Dijkstr)":lambda: a_star_search(g, (0,0,0), (14,14,4), zero_heuristic),
        }

        print("\n" + "="*70)
        print(f"{'Algorithm':<20} {'Found':<6} {'PathLen':<10} {'Cost':<10} {'Nodes':<8} {'ms':<8}")
        print("-"*70)

        results = {}
        for name, fn in algorithms.items():
            r = fn()
            results[name] = r
            found = "Yes" if r.found else "No"
            plen  = str(len(r.path)) if r.found else "-"
            cost  = f"{r.cost:.2f}" if r.found else "-"
            print(f"{name:<20} {found:<6} {plen:<10} {cost:<10} {r.nodes_expanded:<8} {r.time_ms:<8.2f}")

        print("="*70)

        # Kiểm tra tính chất cốt lõi
        r_astar = results["A*(Euclid)"]
        r_zero  = results["A*(Zero/Dijkstr)"]

        if r_astar.found and r_zero.found:
            # A*(Euclid) optimal như Dijkstra
            assert abs(r_astar.cost - r_zero.cost) < 1e-6, \
                "A*(Euclid) phải cho cost bằng Dijkstra"
            # A* expand ít node hơn Dijkstra
            assert r_astar.nodes_expanded <= r_zero.nodes_expanded, \
                "A*(Euclid) phải expand ít node hơn Dijkstra"
