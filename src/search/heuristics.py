"""
Heuristic functions cho A* Search
Lý thuyết: Chương 4 (Admissible heuristic, Informed Search)
"""

import math


def euclidean_3d(node: tuple, goal: tuple) -> float:
    """
    Khoảng cách Euclidean 3D — Ch4: Admissible heuristic ✓

    h(n) = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)

    Luôn admissible: đường chim bay là đường ngắn nhất có thể
    → h(n) ≤ chi phí thực → A* đảm bảo optimal
    """
    dx = goal[0] - node[0]
    dy = goal[1] - node[1]
    dz = goal[2] - node[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def manhattan_3d(node: tuple, goal: tuple) -> float:
    """
    Khoảng cách Manhattan 3D — Ch4: Admissible ✓ khi chỉ đi 6 hướng

    h(n) = |x2-x1| + |y2-y1| + |z2-z1|

    Admissible khi không cho di chuyển chéo (DIRECTIONS_6).
    Consistent (monotone): h(n) ≤ c(n,n') + h(n') → A* không cần re-expand
    """
    return (abs(goal[0] - node[0]) +
            abs(goal[1] - node[1]) +
            abs(goal[2] - node[2]))


def wind_adjusted(node: tuple, goal: tuple,
                  wind_vector: tuple = (0.0, 0.0, 0.0)) -> float:
    """
    Heuristic có tính gió — domain-specific cho UAV

    Bay xuôi gió: giảm chi phí ước lượng
    Bay ngược gió: tăng chi phí ước lượng

    Công thức:
        base = euclidean_3d(node, goal)
        direction = normalize(goal - node)
        wind_effect = dot(direction, wind_vector)   ∈ [-1, 1]
        h = base / (1 + max(wind_effect, 0))

    Lưu ý: chỉ giảm khi xuôi gió, không tăng quá để giữ admissibility
    """
    base = euclidean_3d(node, goal)
    if base < 1e-9:
        return 0.0

    # Vector hướng bay từ node đến goal (normalize)
    dx = (goal[0] - node[0]) / base
    dy = (goal[1] - node[1]) / base
    dz = (goal[2] - node[2]) / base

    # Dot product với wind_vector
    wind_effect = dx * wind_vector[0] + dy * wind_vector[1] + dz * wind_vector[2]

    # Chỉ giảm khi xuôi gió (wind_effect > 0), giữ admissibility
    divisor = 1.0 + max(wind_effect, 0.0) * 0.3
    return base / divisor


def energy_based(node: tuple, goal: tuple,
                 climb_penalty: float = 1.5) -> float:
    """
    Heuristic tính năng lượng — Leo cao tốn hơn bay ngang

    Công thức:
        horizontal = euclidean_2d(node, goal)
        vertical   = |z_goal - z_node| * climb_penalty
        h = horizontal + vertical  (khi goal cao hơn node)
        h = horizontal             (khi goal thấp hơn — lượn xuống dễ)

    Phản ánh thực tế: UAV tốn nhiều năng lượng khi tăng độ cao
    Admissibility: phụ thuộc climb_penalty — nên ≤ actual climb cost
    """
    dx = goal[0] - node[0]
    dy = goal[1] - node[1]
    dz = goal[2] - node[2]

    horizontal = math.sqrt(dx * dx + dy * dy)

    # Chỉ tính penalty khi leo cao (dz > 0)
    vertical = max(dz, 0) * (climb_penalty - 1.0)

    return horizontal + abs(dz) + vertical


def zero_heuristic(node: tuple, goal: tuple) -> float:
    """
    h(n) = 0 — A* thoái hóa thành Dijkstra

    Luôn admissible (h=0 ≤ mọi chi phí thực)
    Dùng để benchmark: optimal nhưng chậm nhất trong informed search
    """
    return 0.0


# Dict tiện dùng: tên → hàm
HEURISTICS = {
    "euclidean": euclidean_3d,
    "manhattan": manhattan_3d,
    "wind":      wind_adjusted,
    "energy":    energy_based,
    "zero":      zero_heuristic,
}
