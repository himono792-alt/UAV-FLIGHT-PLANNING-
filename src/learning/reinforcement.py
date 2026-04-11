"""
Học tăng cường (Reinforcement Learning).
Học chính sách tối ưu từ tương tác với môi trường.
Chương 9: Học máy.
"""


class QLearningAgent:
    """
    Tác nhân học tăng cường dùng Q-Learning.

    Q-Learning:
    - Học giá trị hành động (Q-values) mà không cần mô hình môi trường
    - Hành động: Fly, Turn, Descend, Land
    - Trạng thái: Vị trí, năng lượng, thời tiết
    - Phần thưởng: Hoàn thành nhiệm vụ, tiêu thụ năng lượng, an toàn

    Cập nhật Q-value:
    Q(s,a) <- Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Khởi tạo Q-Learning Agent.

        Args:
            learning_rate (α): Tốc độ học (0-1)
            discount_factor (γ): Hệ số chiết khấu cho phần thưởng tương lai
            epsilon: Tỷ lệ khám phá vs khai thác
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    # TODO: Triển khai ở Giai đoạn 6
    # - Initialize Q-table
    # - Epsilon-greedy action selection
    # - Q-value update
    # - Experience replay (optional)
    # - Policy evaluation and improvement
