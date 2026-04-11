"""
UAV Flight Planning Agent — Entry Point
Demo 5 kịch bản áp dụng đầy đủ 9/9 chương TTNT (PTIT)

Chạy: python main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.environment.grid_world import GridWorld3D, CellType
from src.agent.uav_agent import UAVAgent


def _make_grid(w=12, h=12, d=5, start=(0, 0, 0), goal=(11, 11, 4)):
    g = GridWorld3D(w, h, d)
    g.set_start(start)
    g.set_goal(goal)
    return g


def _print_result(name, result):
    status = "✓" if result.success else "✗"
    print(f"\n  {status} {name}")
    print(f"     {result.summary()}")
    if result.events:
        for ev in result.events[:3]:
            print(f"     EVENT: {ev}")


# ── Kịch bản 1: Bay bình thường ────────────────────────────────────────── #

def demo_scenario_1():
    """Điều kiện lý tưởng — fly_direct, A* + Euclidean heuristic"""
    g = _make_grid()
    agent = UAVAgent(g, wind_speed=5.0, initial_battery=100.0, seed=1)
    result = agent.run_mission((0, 0, 0), (11, 11, 4), max_steps=300)
    _print_result("Scenario 1: Bay bình thường (gió 5 m/s, pin 100%)", result)
    return result


# ── Kịch bản 2: Nhiều chướng ngại vật ─────────────────────────────────── #

def demo_scenario_2():
    """Grid dày obstacles — A* tìm đường vòng, DT=AVOID → fly_safe"""
    g = _make_grid()
    g.add_random_obstacles(25, seed=42)
    agent = UAVAgent(g, wind_speed=8.0, initial_battery=100.0, seed=2)
    result = agent.run_mission((0, 0, 0), (11, 11, 4), max_steps=400)
    _print_result("Scenario 2: Nhiều chướng ngại vật (25 obstacles)", result)
    return result


# ── Kịch bản 3: Gió mạnh ──────────────────────────────────────────────── #

def demo_scenario_3():
    """Gió 22 m/s — BN Weather=Rainy, MEU→fly_safe, A*+SA optimize"""
    g = _make_grid()
    agent = UAVAgent(g, wind_speed=22.0, initial_battery=100.0, seed=3)
    result = agent.run_mission((0, 0, 0), (11, 11, 4), max_steps=300)
    _print_result("Scenario 3: Gió mạnh (22 m/s, BN+MEU quyết định)", result)
    return result


# ── Kịch bản 4: Pin yếu ───────────────────────────────────────────────── #

def demo_scenario_4():
    """Pin 18% — KB trigger RETURN_HOME, Greedy BFS về điểm xuất phát"""
    g = _make_grid()
    agent = UAVAgent(g, wind_speed=5.0, initial_battery=18.0, seed=4)
    result = agent.run_mission((0, 0, 0), (11, 11, 4), max_steps=100)
    _print_result("Scenario 4: Pin yếu (18%), KB→RETURN_HOME", result)
    return result


# ── Kịch bản 5: Dynamic re-planning ───────────────────────────────────── #

def demo_scenario_5():
    """
    Obstacle cột giữa đường (4-5, 4-5, all z) →
    LIDAR phát hiện → AVOID → A* tìm đường mới
    """
    g = _make_grid(10, 10, 4, start=(0, 0, 0), goal=(9, 9, 3))
    g.add_obstacle_box(4, 4, 0, 5, 5, 3)   # obstacle cột chặn giữa
    agent = UAVAgent(g, wind_speed=6.0, initial_battery=100.0, seed=5)
    result = agent.run_mission((0, 0, 0), (9, 9, 3), max_steps=400)
    _print_result("Scenario 5: Dynamic re-planning (obstacle giữa đường)", result)
    return result


# ── Main ───────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  UAV FLIGHT PLANNING AGENT — Demo")
    print("  Đồ án TTNT — PTIT")
    print("=" * 60)
    print("\nModules tích hợp:")
    print("  GĐ1: GridWorld3D + ObstacleManager      (Ch1, Ch2)")
    print("  GĐ2: A* Search + heuristics             (Ch3)")
    print("  GĐ3: SA + GA optimizer                  (Ch4)")
    print("  GĐ4: KB-Agent + Propositional + FOL     (Ch5, Ch6)")
    print("  GĐ5: Bayesian Network + MEU             (Ch7, Ch8)")
    print("  GĐ6: Decision Tree + Regression + QL    (Ch9)")
    print()

    results = []
    for fn in [demo_scenario_1, demo_scenario_2, demo_scenario_3,
               demo_scenario_4, demo_scenario_5]:
        results.append(fn())

    success = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"  Kết quả: {success}/5 scenarios thành công")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    main()
