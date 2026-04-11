"""
A* Search — Thuật toán tìm đường tối ưu cốt lõi
Lý thuyết: Chương 4 (A* Search, f(n) = g(n) + h(n))
"""

import heapq
import time
from typing import Callable, Optional

from src.search.graph_search import SearchResult, reconstruct_path, _compute_path_cost
from src.search.heuristics import euclidean_3d


def a_star_search(grid, start: tuple, goal: tuple,
                  heuristic_fn: Callable = euclidean_3d) -> SearchResult:
    """
    A* Search theo Ch4:

    f(n) = g(n) + h(n)
    - g(n): chi phí thực từ start đến n  (biết chính xác)
    - h(n): heuristic ước lượng từ n đến goal (phải admissible)

    Dùng heapq (min-heap) làm priority queue — luôn expand node có f nhỏ nhất.

    Tính chất (Ch4):
    - Complete: Có — nếu branching factor hữu hạn và chi phí > 0
    - Optimal:  Có — nếu h(n) admissible (h(n) ≤ chi phí thực)
    - Time:     O(b^d) worst case
    - Space:    O(b^d) — giữ tất cả node trong bộ nhớ (tốn hơn DFS)
    """
    t0 = time.perf_counter()

    if start == goal:
        return SearchResult(
            path=[start], cost=0.0,
            nodes_expanded=0, time_ms=0.0,
            algorithm="A*", found=True
        )

    # Priority queue: (f_score, counter, node)
    # counter là tie-breaker — heapq không so sánh được tuple nếu f bằng nhau
    counter = 0
    open_set = [(heuristic_fn(start, goal), counter, start)]

    # g_score[n] = chi phí thực từ start đến n
    g_score = {start: 0.0}

    # came_from[n] = node đến n
    came_from = {}

    # closed set: các node đã expand xong
    closed = set()

    nodes_expanded = 0

    while open_set:
        f, _, current = heapq.heappop(open_set)

        # Bỏ qua nếu đã expand (có thể bị push vào heap nhiều lần)
        if current in closed:
            continue

        # GOAL-TEST
        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = _compute_path_cost(grid, path)
            return SearchResult(
                path=path, cost=cost,
                nodes_expanded=nodes_expanded,
                time_ms=(time.perf_counter() - t0) * 1000,
                algorithm="A*", found=True
            )

        closed.add(current)
        nodes_expanded += 1

        for neighbor in grid.get_neighbors(current):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + grid.step_cost(current, neighbor)

            # Chỉ cập nhật nếu tìm được đường tốt hơn
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    # Goal không thể đến được
    return SearchResult(
        path=[], cost=float('inf'),
        nodes_expanded=nodes_expanded,
        time_ms=(time.perf_counter() - t0) * 1000,
        algorithm="A*", found=False
    )


def compare_algorithms(grid, start: tuple, goal: tuple) -> list:
    """
    Chạy tất cả thuật toán trên cùng 1 grid và in bảng so sánh.
    Trả về list SearchResult theo thứ tự: BFS, DFS, Greedy, A*(Euclid),
    A*(Manhattan), A*(Energy), A*(Zero/Dijkstra)
    """
    from src.search.graph_search import bfs, dfs
    from src.search.greedy_bfs import greedy_bfs
    from src.search.heuristics import (
        euclidean_3d, manhattan_3d, energy_based, zero_heuristic
    )

    results = [
        bfs(grid, start, goal),
        dfs(grid, start, goal),
        greedy_bfs(grid, start, goal, euclidean_3d),
        a_star_search(grid, start, goal, euclidean_3d),
        a_star_search(grid, start, goal, manhattan_3d),
        a_star_search(grid, start, goal, energy_based),
        a_star_search(grid, start, goal, zero_heuristic),
    ]

    # In bảng
    header = f"{'Algorithm':<20} {'Found':<6} {'Path len':<10} {'Cost':<10} {'Nodes':<8} {'Time(ms)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        name = r.algorithm
        found = "Yes" if r.found else "No"
        plen = str(len(r.path)) if r.found else "-"
        cost = f"{r.cost:.2f}" if r.found else "-"
        print(f"{name:<20} {found:<6} {plen:<10} {cost:<10} {r.nodes_expanded:<8} {r.time_ms:<10.2f}")

    return results
