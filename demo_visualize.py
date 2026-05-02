"""
Demo trực quan UAV tìm đường — 3D Visualization

Chạy: python demo_visualize.py
       (mở cửa sổ matplotlib 3D — kéo chuột để xoay)

Demo 4 scenario:
  1. A* trong môi trường trống
  2. A* tránh nhiều obstacles
  3. A* qua hẻm hẹp (corridor)
  4. So sánh A* vs Greedy BFS

Mọi ảnh PNG được lưu vào folder outputs/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.grid_world import GridWorld3D
from src.environment.visualizer import UAVVisualizer
from src.search.a_star import a_star_search
from src.search.greedy_bfs import greedy_bfs
from src.search.heuristics import euclidean_3d


# Tạo folder outputs nếu chưa có
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_result(name: str, result):
    if result.found:
        print(f"  ✓ {name}")
        print(f"     Path length : {len(result.path)} bước")
        print(f"     Cost        : {result.cost:.2f}")
        print(f"     Nodes expanded: {result.nodes_expanded}")
        print(f"     Time        : {result.time_ms:.2f} ms")
    else:
        print(f"  ✗ {name}: KHÔNG TÌM THẤY ĐƯỜNG")


# ─────────────────────────────────────────────────────────────────────────
# Scenario 1: A* trong môi trường trống (bay thẳng đường chéo)
# ─────────────────────────────────────────────────────────────────────────

def scenario_1_simple():
    _print_header("Scenario 1: A* trong môi trường trống")

    grid = GridWorld3D(12, 12, 5, allow_diagonal=True)
    grid.set_start((0, 0, 0))
    grid.set_goal((11, 11, 4))

    result = a_star_search(grid, (0, 0, 0), (11, 11, 4),
                           heuristic_fn=euclidean_3d)
    _print_result("A* search", result)

    viz = UAVVisualizer(grid, title="Scenario 1: A* — Empty environment")
    save_path = os.path.join(OUTPUTS_DIR, "scenario_1_simple.png")
    viz.render(path=result.path, show=True, save_path=save_path)


# ─────────────────────────────────────────────────────────────────────────
# Scenario 2: A* tránh obstacle ngẫu nhiên
# ─────────────────────────────────────────────────────────────────────────

def scenario_2_obstacles():
    _print_header("Scenario 2: A* tránh 30 obstacles ngẫu nhiên")

    grid = GridWorld3D(12, 12, 5, allow_diagonal=True)
    grid.set_start((0, 0, 0))
    grid.set_goal((11, 11, 4))
    grid.add_random_obstacles(count=30, seed=42)

    print(f"  Obstacle density: {grid.obstacle_density() * 100:.1f}%")

    result = a_star_search(grid, (0, 0, 0), (11, 11, 4),
                           heuristic_fn=euclidean_3d)
    _print_result("A* search", result)

    viz = UAVVisualizer(grid, title="Scenario 2: A* — Avoid obstacles")
    save_path = os.path.join(OUTPUTS_DIR, "scenario_2_obstacles.png")
    viz.render(path=result.path, show=True, save_path=save_path)


# ─────────────────────────────────────────────────────────────────────────
# Scenario 3: A* qua hẻm hẹp (corridor)
# ─────────────────────────────────────────────────────────────────────────

def scenario_3_corridor():
    _print_header("Scenario 3: A* qua hẻm hẹp giữa 2 building")

    grid = GridWorld3D(15, 15, 6, allow_diagonal=True)
    grid.set_start((0, 7, 0))
    grid.set_goal((14, 7, 5))

    # 2 "tòa nhà" 2 bên, để 1 hẻm hẹp ở giữa
    grid.add_obstacle_box(5, 0, 0, 7, 5, 4)     # building bên trái
    grid.add_obstacle_box(5, 9, 0, 7, 14, 4)    # building bên phải
    # 1 obstacle giữa hẻm để buộc bay vòng/leo cao
    grid.add_obstacle_box(10, 6, 0, 11, 8, 2)

    result = a_star_search(grid, (0, 7, 0), (14, 7, 5),
                           heuristic_fn=euclidean_3d)
    _print_result("A* search", result)

    viz = UAVVisualizer(grid, title="Scenario 3: A* — Through corridor")
    save_path = os.path.join(OUTPUTS_DIR, "scenario_3_corridor.png")
    viz.render(path=result.path, show=True, save_path=save_path)


# ─────────────────────────────────────────────────────────────────────────
# Scenario 4: So sánh A* vs Greedy BFS
# ─────────────────────────────────────────────────────────────────────────

def scenario_4_compare():
    _print_header("Scenario 4: So sánh A* vs Greedy BFS")

    grid = GridWorld3D(12, 12, 5, allow_diagonal=True)
    grid.set_start((0, 0, 0))
    grid.set_goal((11, 11, 4))
    grid.add_random_obstacles(count=20, seed=7)

    res_a = a_star_search(grid, (0, 0, 0), (11, 11, 4),
                          heuristic_fn=euclidean_3d)
    res_g = greedy_bfs(grid, (0, 0, 0), (11, 11, 4),
                       heuristic_fn=euclidean_3d)

    _print_result("A* (optimal)", res_a)
    _print_result("Greedy BFS (fast but không optimal)", res_g)

    print(f"\n  → A* cost: {res_a.cost:.2f} | Greedy cost: {res_g.cost:.2f}")
    print(f"  → A* expanded {res_a.nodes_expanded} nodes vs Greedy {res_g.nodes_expanded}")
    if res_a.cost < res_g.cost:
        print(f"  → A* tốt hơn {(res_g.cost - res_a.cost) / res_a.cost * 100:.1f}%")

    # Vẽ A*
    viz = UAVVisualizer(grid, title="Scenario 4a: A* (optimal path)")
    viz.render(path=res_a.path, show=True,
               save_path=os.path.join(OUTPUTS_DIR, "scenario_4a_astar.png"))

    # Vẽ Greedy
    viz = UAVVisualizer(grid, title="Scenario 4b: Greedy BFS (greedy path)")
    viz.render(path=res_g.path, show=True,
               save_path=os.path.join(OUTPUTS_DIR, "scenario_4b_greedy.png"))


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("█" * 60)
    print("█  UAV FLIGHT PLANNING — VISUAL DEMO")
    print("█  Mở cửa sổ matplotlib 3D, kéo chuột để xoay")
    print("█  Đóng cửa sổ để chuyển sang scenario tiếp theo")
    print("█" * 60)

    scenario_1_simple()
    scenario_2_obstacles()
    scenario_3_corridor()
    scenario_4_compare()

    print()
    print("=" * 60)
    print(f"  ✅ Hoàn tất. Ảnh PNG đã lưu vào: {OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
