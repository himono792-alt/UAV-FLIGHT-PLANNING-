"""
Greedy Best-First Search
Lý thuyết: Chương 4 — f(n) = h(n), nhanh nhưng KHÔNG optimal
"""

import heapq
import time
from typing import Callable

from src.search.graph_search import SearchResult, reconstruct_path, _compute_path_cost
from src.search.heuristics import euclidean_3d


def greedy_bfs(grid, start: tuple, goal: tuple,
               heuristic_fn: Callable = euclidean_3d) -> SearchResult:
    """
    Greedy Best-First Search theo Ch4:

    f(n) = h(n)  — chỉ dùng heuristic, BỎ QUA g(n)

    So sánh với A*:
    - Nhanh hơn: thường expand ít node hơn
    - KHÔNG optimal: có thể trả về đường dài hơn A*
    - Không complete trong không gian vô hạn (có thể bị kẹt vòng lặp)

    Use case UAV: pin yếu, cần tìm đường về gấp — ưu tiên tốc độ hơn tối ưu
    """
    t0 = time.perf_counter()

    if start == goal:
        return SearchResult(
            path=[start], cost=0.0,
            nodes_expanded=0, time_ms=0.0,
            algorithm="Greedy BFS", found=True
        )

    counter = 0
    # Chỉ dùng h(n) — không có g(n)
    open_set = [(heuristic_fn(start, goal), counter, start)]

    came_from = {}
    closed = set()
    in_open = {start}
    nodes_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        in_open.discard(current)

        if current in closed:
            continue

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = _compute_path_cost(grid, path)
            return SearchResult(
                path=path, cost=cost,
                nodes_expanded=nodes_expanded,
                time_ms=(time.perf_counter() - t0) * 1000,
                algorithm="Greedy BFS", found=True
            )

        closed.add(current)
        nodes_expanded += 1

        for neighbor in grid.get_neighbors(current):
            if neighbor in closed or neighbor in in_open:
                continue
            came_from[neighbor] = current
            h = heuristic_fn(neighbor, goal)
            counter += 1
            heapq.heappush(open_set, (h, counter, neighbor))
            in_open.add(neighbor)

    return SearchResult(
        path=[], cost=float('inf'),
        nodes_expanded=nodes_expanded,
        time_ms=(time.perf_counter() - t0) * 1000,
        algorithm="Greedy BFS", found=False
    )
