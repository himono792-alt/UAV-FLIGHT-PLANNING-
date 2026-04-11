"""
Thuật toán Greedy Best-First Search.
Mở rộng nút với heuristic tốt nhất, nhanh hơn A* nhưng có thể không tối ưu.
Chương 3: Giải quyết vấn đề bằng tìm kiếm được thông tin.
"""


class GreedyBFS:
    """
    Triển khai thuật toán Greedy Best-First Search.

    Mở rộng nút được đánh giá cao nhất theo heuristic h(n).
    Nhanh hơn A* nhưng không đảm bảo đường đi tối ưu.
    """

    def __init__(self, heuristic=None):
        """
        Khởi tạo Greedy BFS.

        Args:
            heuristic: Hàm heuristic cho bài toán
        """
        self.heuristic = heuristic

    # TODO: Triển khai ở Giai đoạn 2
    # - Implement greedy best-first search
    # - Use priority queue with heuristic values
    # - Support early termination when goal found
