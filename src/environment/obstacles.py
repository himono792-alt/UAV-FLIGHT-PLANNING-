"""
Obstacles & NoFlyZone cho UAV Environment
Lý thuyết: Chương 2 (Environment types), Chương 3 (State constraints)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Obstacle:
    """
    Chướng ngại vật trong không gian 3D

    Static: tòa nhà, cây cối, địa hình
    Dynamic: máy bay khác, chim, drone khác
    """
    x: int
    y: int
    z: int
    size_x: int = 1
    size_y: int = 1
    size_z: int = 1
    is_dynamic: bool = False
    name: str = "obstacle"

    def get_cells(self) -> List[Tuple[int, int, int]]:
        """Trả về tất cả ô mà obstacle chiếm"""
        cells = []
        for dx in range(self.size_x):
            for dy in range(self.size_y):
                for dz in range(self.size_z):
                    cells.append((self.x + dx, self.y + dy, self.z + dz))
        return cells

    def is_collision(self, pos: Tuple[int, int, int]) -> bool:
        """Kiểm tra vị trí có va chạm với obstacle này không"""
        px, py, pz = pos
        return (self.x <= px < self.x + self.size_x and
                self.y <= py < self.y + self.size_y and
                self.z <= pz < self.z + self.size_z)

    def move(self, dx: int = 0, dy: int = 0, dz: int = 0):
        """Di chuyển obstacle (dynamic obstacles — Ch2: Dynamic environment)"""
        if self.is_dynamic:
            self.x += dx
            self.y += dy
            self.z += dz


@dataclass
class NoFlyZone:
    """
    Vùng cấm bay — No-Fly Zone

    Ví dụ: vùng sân bay, vùng quân sự, vùng dân cư
    UAV PHẢI tránh — luật cứng (Ch5: KB rules)
    """
    x1: int
    y1: int
    z1: int
    x2: int
    y2: int
    z2: int
    name: str = "no_fly_zone"
    reason: str = "restricted"

    def contains(self, pos: Tuple[int, int, int]) -> bool:
        """Kiểm tra vị trí có nằm trong vùng cấm không"""
        px, py, pz = pos
        return (self.x1 <= px <= self.x2 and
                self.y1 <= py <= self.y2 and
                self.z1 <= pz <= self.z2)

    def get_cells(self) -> List[Tuple[int, int, int]]:
        cells = []
        for x in range(self.x1, self.x2 + 1):
            for y in range(self.y1, self.y2 + 1):
                for z in range(self.z1, self.z2 + 1):
                    cells.append((x, y, z))
        return cells


class ObstacleManager:
    """
    Quản lý toàn bộ obstacles và no-fly zones trong môi trường
    Tích hợp với GridWorld3D
    """

    def __init__(self):
        self.obstacles: List[Obstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []

    # ------------------------------------------------------------------ #
    #  Thêm obstacles                                                      #
    # ------------------------------------------------------------------ #

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)

    def add_no_fly_zone(self, nfz: NoFlyZone):
        self.no_fly_zones.append(nfz)

    def add_building(self, x: int, y: int, height: int,
                     size_x: int = 1, size_y: int = 1, name: str = "building"):
        """Thêm toà nhà: từ z=0 lên đến height"""
        obs = Obstacle(x=x, y=y, z=0,
                       size_x=size_x, size_y=size_y, size_z=height,
                       name=name)
        self.obstacles.append(obs)
        return obs

    def generate_random_obstacles(self, count: int,
                                  width: int, height: int, depth: int,
                                  seed: int = None) -> List[Obstacle]:
        """Sinh chướng ngại vật ngẫu nhiên — phân bố đều không gian"""
        rng = np.random.default_rng(seed)
        new_obs = []
        for i in range(count):
            x = int(rng.integers(1, width - 2))
            y = int(rng.integers(1, height - 2))
            z = int(rng.integers(0, depth - 1))
            sx = int(rng.integers(1, 3))
            sy = int(rng.integers(1, 3))
            sz = int(rng.integers(1, max(2, depth // 3)))
            obs = Obstacle(x=x, y=y, z=z,
                           size_x=sx, size_y=sy, size_z=sz,
                           name=f"rand_obs_{i}")
            new_obs.append(obs)
            self.obstacles.append(obs)
        return new_obs

    # ------------------------------------------------------------------ #
    #  Kiểm tra va chạm                                                   #
    # ------------------------------------------------------------------ #

    def is_collision(self, pos: Tuple[int, int, int]) -> bool:
        """
        Kiểm tra va chạm tại vị trí pos với bất kỳ obstacle nào
        Dùng trong A* để loại bỏ state không an toàn (Ch3)
        """
        for obs in self.obstacles:
            if obs.is_collision(pos):
                return True
        return False

    def is_in_no_fly_zone(self, pos: Tuple[int, int, int]) -> bool:
        """Kiểm tra vị trí có trong vùng cấm bay không"""
        for nfz in self.no_fly_zones:
            if nfz.contains(pos):
                return True
        return False

    def is_safe(self, pos: Tuple[int, int, int]) -> bool:
        """An toàn = không va chạm VÀ không trong no-fly zone"""
        return not self.is_collision(pos) and not self.is_in_no_fly_zone(pos)

    # ------------------------------------------------------------------ #
    #  Áp dụng lên GridWorld3D                                            #
    # ------------------------------------------------------------------ #

    def apply_to_grid(self, grid):
        """
        Ghi tất cả obstacles và no-fly zones lên GridWorld3D
        Gọi sau khi thêm đủ obstacles
        """
        from src.environment.grid_world import CellType

        for obs in self.obstacles:
            for cell in obs.get_cells():
                if grid.in_bounds(cell):
                    grid.set_cell(cell, CellType.OBSTACLE)

        for nfz in self.no_fly_zones:
            for cell in nfz.get_cells():
                if grid.in_bounds(cell):
                    grid.set_cell(cell, CellType.NO_FLY_ZONE)

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def update_dynamic_obstacles(self, grid):
        """
        Cập nhật vị trí dynamic obstacles (gọi mỗi timestep)
        Phản ánh môi trường Dynamic (Ch2)
        """
        from src.environment.grid_world import CellType
        import random

        for obs in self.obstacles:
            if obs.is_dynamic:
                # Xóa vị trí cũ
                for cell in obs.get_cells():
                    if grid.in_bounds(cell):
                        grid.set_cell(cell, CellType.FREE)
                # Di chuyển ngẫu nhiên 1 bước
                dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
                obs.move(dx=dx, dy=dy)
                # Ghi vị trí mới
                for cell in obs.get_cells():
                    if grid.in_bounds(cell):
                        grid.set_cell(cell, CellType.OBSTACLE)

    def summary(self) -> str:
        total_cells = sum(
            obs.size_x * obs.size_y * obs.size_z for obs in self.obstacles
        )
        return (
            f"ObstacleManager:\n"
            f"  Obstacles: {len(self.obstacles)} ({total_cells} cells)\n"
            f"  No-Fly Zones: {len(self.no_fly_zones)}\n"
            f"  Dynamic: {sum(1 for o in self.obstacles if o.is_dynamic)}"
        )
