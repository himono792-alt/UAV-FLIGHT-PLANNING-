"""
Suy luận trong mạng Bayes (Bayesian Inference).
Tính toán xác suất hậu nghiệm khi có bằng chứng mới.
Chương 7: Suy luận xác suất.
"""


class BayesianInference:
    """
    Công cụ suy luận trong mạng Bayes.

    Phương pháp:
    1. Elimination (loại bỏ biến)
    2. Prior sampling (lấy mẫu từ lựa chọn trước)
    3. Likelihood weighting (cân nặng độ tin cậy)
    4. Gibbs sampling (lấy mẫu Markov)
    """

    def __init__(self, bn=None):
        """
        Khởi tạo Bayesian Inference.

        Args:
            bn: Mạng Bayes
        """
        self.bn = bn

    # TODO: Triển khai ở Giai đoạn 5
    # - Variable elimination algorithm
    # - Prior sampling
    # - Likelihood weighting
    # - Query probability computation
    # - Handling evidence
