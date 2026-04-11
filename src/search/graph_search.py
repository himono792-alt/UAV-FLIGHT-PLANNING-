"""
Graph-Search tổng quát — BFS và DFS baseline
Lý thuyết: Chương 3 (Graph-Search, frontier + explored set)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """Kết quả trả về từ mọi thuật toán search"""
    path: list           # [(x,y,z), ...] từ start đến goal
    cost: float          # tổng chi phí đường đi
    nodes_expanded: int  # số node đã mở rộng (expand)
    time_ms: float       # thời gian chạy (milliseconds)
    algorithm: str       # tên thuật toán
    found: bool          # tìm được đường không

    def summary(self) -> str:
        if not self.found:
            return f"[{self.algorithm}] Không tìm được đường | {self.nodes_expanded} nodes | {self.time_ms:.2f}ms"
        return (
            f"[{self.algorithm}] "
            f"Path length={len(self.path)} | "
            f"Cost={self.cost:.2f} | "
            f"Nodes expanded={self.nodes_expanded} | "
            f"Time={self.time_ms:.2f}ms"
        )


def reconstruct_path(came_from: dict, current: tuple) -> list:
    """
    Truy ngược đường đi từ goal về start qua dict came_from.
    came_from[node] = parent của node đó
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _compute_path_cost(grid, path: list) -> float:
    """Tính tổng chi phí của 1 path"""
    total = 0.0
    for i in range(len(path) - 1):
        total += grid.step_cost(path[i], path[i + 1])
    return total


def graph_search(grid, start: tuple, goal: tuple,
                 strategy: str = "bfs") -> SearchResult:
    """
    Graph-Search tổng quát theo pseudocode Ch3:

    function GRAPH-SEARCH(problem) returns solution or failure
        frontier ← INSERT(MAKE-NODE(INITIAL-STATE), frontier)
        explored ← empty set
        loop do
            if EMPTY?(frontier) then return failure
            node ← REMOVE-FRONT(frontier)
            if GOAL-TEST(node.state) then return SOLUTION(node)
            add node.state to explored
            for each action in ACTIONS(node.state) do
                child ← CHILD-NODE(problem, node, action)
                if child.state not in explored and not in frontier then
                    frontier ← INSERT(child, frontier)

    strategy:
        "bfs" → frontier = deque (FIFO) — đầy đủ, tối ưu với step cost đều
        "dfs" → frontier = list/stack (LIFO) — KHÔNG tối ưu, có thể vô hạn
    """
    t0 = time.perf_counter()
    algo = strategy.upper()

    if start == goal:
        return SearchResult(
            path=[start], cost=0.0,
            nodes_expanded=0, time_ms=0.0,
            algorithm=algo, found=True
        )

    # Frontier: deque cho BFS, list (stack) cho DFS
    if strategy == "bfs":
        frontier = deque([start])
    else:
        frontier = [start]

    explored = set()             # tập nodes đã expand
    came_from = {}               # lưu parent để reconstruct path
    in_frontier = {start}        # tập kiểm tra nhanh O(1)
    nodes_expanded = 0

    while frontier:
        # REMOVE-FRONT
        if strategy == "bfs":
            current = frontier.popleft()
        else:
            current = frontier.pop()

        in_frontier.discard(current)

        # GOAL-TEST
        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = _compute_path_cost(grid, path)
            return SearchResult(
                path=path, cost=cost,
                nodes_expanded=nodes_expanded,
                time_ms=(time.perf_counter() - t0) * 1000,
                algorithm=algo, found=True
            )

        # Thêm vào explored
        explored.add(current)
        nodes_expanded += 1

        # Expand: duyệt các neighbors hợp lệ
        for neighbor in grid.get_neighbors(current):
            if neighbor not in explored and neighbor not in in_frontier:
                came_from[neighbor] = current
                if strategy == "bfs":
                    frontier.append(neighbor)
                else:
                    frontier.append(neighbor)
                in_frontier.add(neighbor)

    # Không tìm được đường
    return SearchResult(
        path=[], cost=float('inf'),
        nodes_expanded=nodes_expanded,
        time_ms=(time.perf_counter() - t0) * 1000,
        algorithm=algo, found=False
    )


def bfs(grid, start: tuple, goal: tuple) -> SearchResult:
    """
    Breadth-First Search — Ch3
    Tối ưu khi step cost đồng đều
    """
    return graph_search(grid, start, goal, strategy="bfs")


def dfs(grid, start: tuple, goal: tuple) -> SearchResult:
    """
    Depth-First Search — Ch3
    KHÔNG tối ưu, dùng làm baseline so sánh
    """
    return graph_search(grid, start, goal, strategy="dfs")
