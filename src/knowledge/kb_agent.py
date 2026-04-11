"""
Tác nhân dựa trên cơ sở dữ liệu tri thức (Knowledge-Based Agent).
Lưu trữ sự kiện, quy tắc và suy luận dựa trên logic.
Chương 5: Tác nhân logic.
"""


class KnowledgeBase:
    """
    Cơ sở dữ liệu tri thức.

    Lưu trữ:
    - Sự kiện: Thông tin về trạng thái hiện tại
    - Quy tắc: Luật suy luận dạng IF-THEN
    """

    def __init__(self):
        """Khởi tạo Knowledge Base."""
        pass

    # TODO: Triển khai ở Giai đoạn 4
    # - Store facts
    # - Store rules
    # - Query facts
    # - Add/remove facts


class KBAgent:
    """
    Tác nhân dựa trên cơ sở dữ liệu tri thức.

    Chu trình:
    1. Cảm nhận trạng thái môi trường
    2. Cập nhật KB với quan sát mới
    3. Suy luận từ KB để quyết định hành động
    4. Thực hiện hành động
    """

    def __init__(self, kb=None):
        """
        Khởi tạo KB Agent.

        Args:
            kb: Cơ sở dữ liệu tri thức
        """
        self.kb = kb or KnowledgeBase()

    # TODO: Triển khai ở Giai đoạn 4
    # - Implement perception
    # - Implement reasoning loop
    # - Implement action selection based on KB
