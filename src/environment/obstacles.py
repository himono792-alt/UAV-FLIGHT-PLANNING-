"""
Obstacles - Quản lý chướng ngại vật và vùng cấm bay
=====================================================
Lý thuyết bám sát:
- Chương 5 (Wumpus World): Obstacle ↔ Pit, NoFlyZone ↔ Wumpus
- Chương 2 (Dynamic Environment): Chướng ngại vật có thể di chuyển

Giải thích cho sinh viên:
    Trong Wumpus World (Ch5), agent phải tránh Pit (hố) và Wumpus.
    Tương tự, UAV phải tránh chướng ngại vật (tòa nhà, cây) và
    vùng cấm bay (No-Fly Zone). Đây là ví dụ thực tế của bài toán
    Wumpus World trong không gian 3D.
"""

from typing import Tuple, List, Optional
from enum import Enum


class ObstacleType(Enum):
    """Phân loại chướng ngại vật."""
    STATIC = "static"       # Tĩnh: tòa nhà, núi, cây
    DYNAMIC = "dynamic"     # Động: chim, drone khác, máy bay


class Obstacle:
    """Chướng ngại vật trong môi trường.

    Tương tự PIT trong Wumpus World (Ch5):
    - Pit: agent rơi xuống → game over
    - Obstacle: UAV va chạm → hỏng/rơi

    Parameters
    ----------
    position : Tuple[int, int, int]
        Vị trí (x, y, z) góc dưới trái
    size : Tuple[int, int, int]
        Kích thước (sx, sy, sz) theo 3 trục
    obstacle_type : ObstacleType
        Loại: tĩnh hoặc động
    name : str, optional
        Tên mô tả (ví dụ: "Tòa nhà A", "Cây lớn")
    """

    def __init__(self, position: Tuple[int, int, int],
                 size: Tuple[int, int, int] = (1, 1, 1),
                 obstacle_type: ObstacleType = ObstacleType.STATIC,
                 name: str = ""):
        self.position = position
        self.size = size
        self.obstacle_type = obstacle_type
        self.name = name

        # Tính bounding box
        self.x_min, self.y_min, self.z_min = position
        self.x_max = self.x_min + size[0]
        self.y_max = self.y_min + size[1]
        self.z_max = self.z_min + size[2]

    def contains(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra điểm (x,y,z) có nằm trong chướng ngại vật không."""
        return (self.x_min <= x < self.x_max and
                self.y_min <= y < self.y_max and
                self.z_min <= z < self.z_max)

    def get_cells(self) -> List[Tuple[int, int, int]]:
        """Lấy tất cả ô bị chiếm bởi chướng ngại vật."""
        cells = []
        for x in range(self.x_min, self.x_max):
            for y in range(self.y_min, self.y_max):
                for z in range(self.z_min, self.z_max):
                    cells.append((x, y, z))
        return cells

    def move(self, dx: int, dy: int, dz: int):
        """Di chuyển chướng ngại vật (cho loại DYNAMIC).

        Thể hiện tính DYNAMIC của môi trường (Ch2):
        Chướng ngại vật di chuyển → UAV phải re-plan.
        """
        if self.obstacle_type == ObstacleType.DYNAMIC:
            self.position = (
                self.position[0] + dx,
                self.position[1] + dy,
                self.position[2] + dz,
            )
            self.x_min += dx
            self.y_min += dy
            self.z_min += dz
            self.x_max += dx
            self.y_max += dy
            self.z_max += dz

    def __repr__(self) -> str:
        return (f"Obstacle('{self.name}' at {self.position}, "
                f"size={self.size}, type={self.obstacle_type.value})")


class NoFlyZone:
    """Vùng cấm bay - tuyệt đối không được bay vào.

    Tương tự WUMPUS trong Wumpus World (Ch5):
    - Wumpus: agent chết → game over
    - NoFlyZone: UAV vi phạm pháp luật → bị phạt nặng

    Trong thực tế: sân bay, khu quân sự, khu dân cư đông đúc.

    Parameters
    ----------
    position : Tuple[int, int, int]
        Vị trí góc dưới trái
    size : Tuple[int, int, int]
        Kích thước vùng cấm
    name : str
        Tên vùng cấm (ví dụ: "Sân bay Nội Bài")
    penalty : float
        Mức phạt nếu vi phạm (dùng cho Utility function - Ch7)
    """

    def __init__(self, position: Tuple[int, int, int],
                 size: Tuple[int, int, int] = (1, 1, 1),
                 name: str = "", penalty: float = -1000.0):
        self.position = position
        self.size = size
        self.name = name
        self.penalty = penalty

        self.x_min, self.y_min, self.z_min = position
        self.x_max = self.x_min + size[0]
        self.y_max = self.y_min + size[1]
        self.z_max = self.z_min + size[2]

    def contains(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra điểm có nằm trong vùng cấm bay không."""
        return (self.x_min <= x < self.x_max and
                self.y_min <= y < self.y_max and
                self.z_min <= z < self.z_max)

    def get_cells(self) -> List[Tuple[int, int, int]]:
        """Lấy tất cả ô trong vùng cấm bay."""
        cells = []
        for x in range(self.x_min, self.x_max):
            for y in range(self.y_min, self.y_max):
                for z in range(self.z_min, self.z_max):
                    cells.append((x, y, z))
        return cells

    def __repr__(self) -> str:
        return f"NoFlyZone('{self.name}' at {self.position}, size={self.size})"


class ObstacleManager:
    """Quản lý tất cả chướng ngại vật và vùng cấm bay.

    Class này quản lý tập trung, giúp dễ dàng:
    - Thêm/xóa chướng ngại vật
    - Kiểm tra va chạm
    - Cập nhật vị trí chướng ngại vật động
    """

    def __init__(self):
        self.obstacles: List[Obstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []

    def add_obstacle(self, obstacle: Obstacle):
        """Thêm chướng ngại vật."""
        self.obstacles.append(obstacle)

    def add_no_fly_zone(self, zone: NoFlyZone):
        """Thêm vùng cấm bay."""
        self.no_fly_zones.append(zone)

    def is_collision(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra va chạm tại vị trí (x, y, z).

        Tương tự Percept trong Wumpus World (Ch5):
        Agent cảm nhận Breeze (gần Pit) hoặc Stench (gần Wumpus).
        Ở đây: kiểm tra trực tiếp có chướng ngại vật không.
        """
        for obs in self.obstacles:
            if obs.contains(x, y, z):
                return True
        return False

    def is_in_no_fly_zone(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra có nằm trong vùng cấm bay không."""
        for zone in self.no_fly_zones:
            if zone.contains(x, y, z):
                return True
        return False

    def is_safe(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra vị trí an toàn (không va chạm VÀ không cấm bay).

        Logic (Ch5): Safe(x,y,z) ⟺ ¬Obstacle(x,y,z) ∧ ¬NoFlyZone(x,y,z)
        """
        return not self.is_collision(x, y, z) and not self.is_in_no_fly_zone(x, y, z)

    def update_dynamic(self):
        """Cập nhật vị trí chướng ngại vật động.

        Thể hiện tính DYNAMIC của môi trường (Ch2):
        Chướng ngại vật di chuyển mỗi bước thời gian.
        """
        for obs in self.obstacles:
            if obs.obstacle_type == ObstacleType.DYNAMIC:
                # Di chuyển ngẫu nhiên (đơn giản hóa)
                import random
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
                obs.move(dx, dy, 0)

    def get_stats(self) -> dict:
        """Thống kê chướng ngại vật."""
        static = sum(1 for o in self.obstacles if o.obstacle_type == ObstacleType.STATIC)
        dynamic = sum(1 for o in self.obstacles if o.obstacle_type == ObstacleType.DYNAMIC)
        return {
            "tổng_chướng_ngại": len(self.obstacles),
            "tĩnh": static,
            "động": dynamic,
            "vùng_cấm_bay": len(self.no_fly_zones),
        }
