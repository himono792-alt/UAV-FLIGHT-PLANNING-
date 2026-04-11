"""
Các hàm heuristic cho tìm kiếm đường đi UAV.
Ước lượng chi phí từ vị trí hiện tại đến đích.
Chương 3: Các hàm heuristic admissible cho tìm kiếm informed.
"""


def euclidean_3d(current, goal):
    """
    Heuristic khoảng cách Euclidean 3D.

    Ước lượng khoảng cách thẳng trong không gian 3D.

    Args:
        current: Tọa độ hiện tại (x, y, z)
        goal: Tọa độ đích (x, y, z)

    Returns:
        float: Khoảng cách Euclidean
    """
    # TODO: Triển khai ở Giai đoạn 2
    pass


def manhattan_3d(current, goal):
    """
    Heuristic khoảng cách Manhattan 3D.

    Tổng các khoảng cách tuyệt đối trên từng chiều.

    Args:
        current: Tọa độ hiện tại (x, y, z)
        goal: Tọa độ đích (x, y, z)

    Returns:
        float: Khoảng cách Manhattan
    """
    # TODO: Triển khai ở Giai đoạn 2
    pass


def wind_adjusted(current, goal, wind_vector):
    """
    Heuristic điều chỉnh theo gió.

    Tính toán chi phí di chuyển xét đến ảnh hưởng của gió.
    Gió cản sẽ làm tăng chi phí, gió thuận sẽ giảm.

    Args:
        current: Tọa độ hiện tại (x, y, z)
        goal: Tọa độ đích (x, y, z)
        wind_vector: Vector gió (vx, vy, vz)

    Returns:
        float: Chi phí di chuyển điều chỉnh theo gió
    """
    # TODO: Triển khai ở Giai đoạn 2
    pass


def energy_based(current, goal, altitude_cost=1.0):
    """
    Heuristic dựa trên năng lượng tiêu thụ.

    Tính toán chi phí năng lượng dự kiến với trọng số cho độ cao.
    Di chuyển lên cao hơn tiêu thụ nhiều năng lượng hơn.

    Args:
        current: Tọa độ hiện tại (x, y, z)
        goal: Tọa độ đích (x, y, z)
        altitude_cost: Hệ số chi phí cho độ cao (default: 1.0)

    Returns:
        float: Chi phí năng lượng dự kiến
    """
    # TODO: Triển khai ở Giai đoạn 2
    pass
