"""
GridWorld3D - Môi trường lưới 3D cho UAV
Lý thuyết: Chương 2 (PEAS, Environment), Chương 3 (State Space)
"""

import numpy as np
from enum import IntEnum


class CellType(IntEnum):
    FREE = 0
    OBSTACLE = 1
    NO_FLY_ZONE = 2
    START = 3
    GOAL = 4


# 6 hướng di chuyển (không chéo) — Ch3: Actions
DIRECTIONS_6 = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

# 26 hướng (gồm chéo) — Ch3: Actions mở rộng
DIRECTIONS_26 = [
    (dx, dy, dz)
    for dx in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dz in [-1, 0, 1]
    if not (dx == 0 and dy == 0 and dz == 0)
]


class GridWorld3D:
    """
    Môi trường lưới 3D — State Space cho UAV (Ch3)

    State Space: mỗi ô (x, y, z) là một state
    Actions: 6 hoặc 26 hướng di chuyển
    Transition Model: Result(state, action) = state mới nếu hợp lệ

    Phân loại môi trường (Ch2):
    - Partially observable: UAV chỉ thấy xung quanh
    - Stochastic: gió ảnh hưởng kết quả
    - Sequential: quyết định hiện tại ảnh hưởng tương lai
    - Dynamic: chướng ngại vật có thể di chuyển
    - Continuous: vị trí thực tế liên tục (lưới là rời rạc hóa)
    """

    def __init__(self, width: int = 20, height: int = 20, depth: int = 10,
                 allow_diagonal: bool = False):
        self.width = width      # trục x
        self.height = height    # trục y
        self.depth = depth      # trục z (độ cao)
        self.allow_diagonal = allow_diagonal
        self.directions = DIRECTIONS_26 if allow_diagonal else DIRECTIONS_6

        # Lưới trạng thái
        self.grid = np.zeros((width, height, depth), dtype=np.int8)

        self.start: tuple = None
        self.goal: tuple = None

    # ------------------------------------------------------------------ #
    #  Truy cập lưới                                                       #
    # ------------------------------------------------------------------ #

    def in_bounds(self, pos: tuple) -> bool:
        x, y, z = pos
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def is_free(self, pos: tuple) -> bool:
        """True nếu ô có thể bay qua (Ch3: passable state)"""
        if not self.in_bounds(pos):
            return False
        return self.grid[pos] not in (CellType.OBSTACLE, CellType.NO_FLY_ZONE)

    def get_cell(self, pos: tuple) -> CellType:
        if not self.in_bounds(pos):
            raise IndexError(f"Vị trí {pos} ngoài biên")
        return CellType(self.grid[pos])

    def set_cell(self, pos: tuple, cell_type: CellType):
        if not self.in_bounds(pos):
            raise IndexError(f"Vị trí {pos} ngoài biên")
        self.grid[pos] = cell_type

    # ------------------------------------------------------------------ #
    #  Khởi tạo                                                            #
    # ------------------------------------------------------------------ #

    def set_start(self, pos: tuple):
        self.start = pos
        self.set_cell(pos, CellType.START)

    def set_goal(self, pos: tuple):
        self.goal = pos
        self.set_cell(pos, CellType.GOAL)

    def add_obstacle(self, pos: tuple):
        self.set_cell(pos, CellType.OBSTACLE)

    def add_no_fly_zone(self, pos: tuple):
        self.set_cell(pos, CellType.NO_FLY_ZONE)

    def add_obstacle_box(self, x1, y1, z1, x2, y2, z2,
                         cell_type: CellType = CellType.OBSTACLE):
        """Thêm khối chướng ngại vật hình hộp"""
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    if self.in_bounds((x, y, z)):
                        self.set_cell((x, y, z), cell_type)

    def add_random_obstacles(self, count: int, seed: int = None):
        """Thêm chướng ngại vật ngẫu nhiên"""
        rng = np.random.default_rng(seed)
        added = 0
        attempts = 0
        while added < count and attempts < count * 10:
            pos = (
                int(rng.integers(0, self.width)),
                int(rng.integers(0, self.height)),
                int(rng.integers(0, self.depth)),
            )
            if self.grid[pos] == CellType.FREE:
                self.grid[pos] = CellType.OBSTACLE
                added += 1
            attempts += 1

    # ------------------------------------------------------------------ #
    #  State Space (Ch3)                                                   #
    # ------------------------------------------------------------------ #

    def get_neighbors(self, pos: tuple) -> list:
        """
        Expand node — trả về các state kế tiếp hợp lệ (Ch3)
        Result(state, action) = neighbor nếu không bị chặn
        """
        neighbors = []
        for d in self.directions:
            nxt = (pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])
            if self.is_free(nxt):
                neighbors.append(nxt)
        return neighbors

    def step_cost(self, from_pos: tuple, to_pos: tuple) -> float:
        """
        Chi phí di chuyển giữa 2 ô kề nhau (Ch3: Step cost)
        Di chéo tốn hơn di thẳng
        """
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        dz = abs(to_pos[2] - from_pos[2])
        # Leo cao tốn hơn
        vertical_penalty = 0.5 if dz > 0 else 0.0
        return (dx**2 + dy**2 + dz**2) ** 0.5 + vertical_penalty

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def get_all_free_cells(self) -> list:
        return [
            (x, y, z)
            for x in range(self.width)
            for y in range(self.height)
            for z in range(self.depth)
            if self.grid[x, y, z] == CellType.FREE
        ]

    def obstacle_density(self) -> float:
        total = self.width * self.height * self.depth
        blocked = np.sum(self.grid == CellType.OBSTACLE) + \
                  np.sum(self.grid == CellType.NO_FLY_ZONE)
        return blocked / total

    def summary(self) -> str:
        return (
            f"GridWorld3D ({self.width}x{self.height}x{self.depth})\n"
            f"  Start: {self.start}  |  Goal: {self.goal}\n"
            f"  Obstacles: {int(np.sum(self.grid == CellType.OBSTACLE))}\n"
            f"  No-Fly Zones: {int(np.sum(self.grid == CellType.NO_FLY_ZONE))}\n"
            f"  Obstacle density: {self.obstacle_density():.1%}"
        )

    def demo(self):
        """Demo nhanh — in slice z=0 ra terminal"""
        print(self.summary())
        print("\nSlice z=0 (O=obstacle, N=no-fly, S=start, G=goal, .=free):")
        symbols = {
            CellType.FREE: ".",
            CellType.OBSTACLE: "O",
            CellType.NO_FLY_ZONE: "N",
            CellType.START: "S",
            CellType.GOAL: "G",
        }
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                row += symbols[CellType(self.grid[x, y, 0])] + " "
            print(row)
