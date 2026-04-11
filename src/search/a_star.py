"""
Thuật toán A* - Tìm kiếm sử dụng heuristic.
Kết hợp chi phí thực tế và chi phí dự đoán để tìm đường đi tối ưu.
Chương 3: Giải quyết vấn đề bằng tìm kiếm được thông tin.
"""


class AStarSearch:
    """
    Triển khai thuật toán A* tìm kiếm heuristic.

    A* kết hợp:
    - g(n): Chi phí từ đích bắt đầu đến nút n
    - h(n): Heuristic ước lượng chi phí từ n đến đích
    - f(n) = g(n) + h(n): Hàm đánh giá toàn bộ
    """

    def __init__(self, heuristic=None):
        """
        Khởi tạo A* Search.

        Args:
            heuristic: Hàm heuristic cho bài toán
        """
        self.heuristic = heuristic

    # TODO: Triển khai ở Giai đoạn 2
    # - Implement A* algorithm with priority queue
    # - Support custom heuristics
    # - Track path and cost
