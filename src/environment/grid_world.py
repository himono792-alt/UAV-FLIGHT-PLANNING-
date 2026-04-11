"""
GridWorld3D - Không gian lưới 3D cho UAV
=========================================
Lý thuyết bám sát:
- Chương 3 (State Space): Mỗi ô lưới (x, y, z) là một STATE
- Chương 3 (Actions): 6 hướng cơ bản hoặc 26 hướng (gồm chéo)
- Chương 3 (Transition Model): Result(state, action) → state mới
- Chương 2 (Environment): Partially observable, stochastic,
  sequential, dynamic, continuous → rời rạc hóa thành grid

Giải thích cho sinh viên:
    State Space là tập hợp tất cả trạng thái có thể của bài toán.
    Ở đây, mỗi ô (x, y, z) trong lưới 3D là một trạng thái.
    UAV có thể ở bất kỳ ô nào → đó là state space.
"""

import numpy as np
import yaml
from enum import IntEnum
from typing import List, Tuple, Optional


class CellType(IntEnum):
    """Loại ô trong lưới 3D.

    Mỗi ô có một trạng thái xác định UAV có thể bay qua hay không.
    Tương tự Wumpus World (Ch5): mỗi ô có thể là trống, hố, hoặc có quái vật.
    """
    FREE = 0           # Ô trống - UAV bay được
    OBSTACLE = 1       # Chướng ngại vật - tòa nhà, cây, ...
    NO_FLY_ZONE = 2    # Vùng cấm bay - tương tự Wumpus trong Wumpus World
    START = 3          # Điểm xuất phát
    GOAL = 4           # Điểm đích - tương tự Gold trong Wumpus World


class GridWorld3D:
    """Môi trường lưới 3D cho UAV.

    Đây là STATE SPACE (Ch3) của bài toán tìm kiếm:
    - Mỗi ô (x, y, z) là một state
    - Tập hợp tất cả ô = state space
    - UAV di chuyển giữa các ô = transitions

    Phân loại môi trường (Ch2):
    - Partially Observable: UAV chỉ thấy trong phạm vi sensor_range
    - Stochastic: Gió có thể thay đổi ngẫu nhiên
    - Sequential: Quyết định hiện tại ảnh hưởng tương lai
    - Dynamic: Chướng ngại vật có thể di chuyển
    - Continuous → Rời rạc hóa: chia không gian liên tục thành lưới

    Parameters
    ----------
    width : int
        Chiều rộng (trục X)
    height : int
        Chiều cao (trục Y)
    depth : int
        Chiều sâu/độ cao (trục Z)
    """

    # 6 hướng di chuyển cơ bản: trái, phải, trước, sau, lên, xuống
    # Đây là ACTIONS trong formulation bài toán tìm kiếm (Ch3)
    ACTIONS_6 = [
        (1, 0, 0), (-1, 0, 0),   # phải, trái (trục X)
        (0, 1, 0), (0, -1, 0),   # trước, sau (trục Y)
        (0, 0, 1), (0, 0, -1),   # lên, xuống (trục Z)
    ]

    # 26 hướng di chuyển (gồm chéo) - thực tế hơn cho UAV
    ACTIONS_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                ACTIONS_26.append((dx, dy, dz))

    def __init__(self, width: int = 20, height: int = 20, depth: int = 10):
        self.width = width
        self.height = height
        self.depth = depth

        # Lưới 3D: mảng numpy, mặc định tất cả ô FREE
        self.grid = np.full((width, height, depth), CellType.FREE, dtype=np.int8)

        # Vị trí bắt đầu và đích
        self.start: Optional[Tuple[int, int, int]] = None
        self.goal: Optional[Tuple[int, int, int]] = None

        # Tham số gió (stochastic - Ch2)
        self.wind_vector = np.array([0.0, 0.0, 0.0])

    def set_start(self, x: int, y: int, z: int):
        """Đặt vị trí xuất phát (Initial State - Ch3)."""
        if self._is_valid_position(x, y, z):
            # Xóa start cũ nếu có
            if self.start:
                sx, sy, sz = self.start
                self.grid[sx, sy, sz] = CellType.FREE
            self.start = (x, y, z)
            self.grid[x, y, z] = CellType.START
        else:
            raise ValueError(f"Vị trí ({x},{y},{z}) nằm ngoài lưới!")

    def set_goal(self, x: int, y: int, z: int):
        """Đặt vị trí đích (Goal State - Ch3).

        Goal Test (Ch3): kiểm tra UAV đã đến đích chưa.
        Tương tự ô Gold trong Wumpus World (Ch5).
        """
        if self._is_valid_position(x, y, z):
            if self.goal:
                gx, gy, gz = self.goal
                self.grid[gx, gy, gz] = CellType.FREE
            self.goal = (x, y, z)
            self.grid[x, y, z] = CellType.GOAL
        else:
            raise ValueError(f"Vị trí ({x},{y},{z}) nằm ngoài lưới!")

    def add_obstacle(self, x: int, y: int, z: int):
        """Thêm chướng ngại vật tại vị trí (x, y, z).

        Tương tự Pit trong Wumpus World (Ch5):
        UAV không được bay qua ô có chướng ngại vật.
        """
        if self._is_valid_position(x, y, z):
            self.grid[x, y, z] = CellType.OBSTACLE

    def add_no_fly_zone(self, x: int, y: int, z: int):
        """Thêm vùng cấm bay.

        Tương tự Wumpus trong Wumpus World (Ch5):
        Vùng tuyệt đối không được bay vào.
        """
        if self._is_valid_position(x, y, z):
            self.grid[x, y, z] = CellType.NO_FLY_ZONE

    def add_obstacle_block(self, x_start: int, y_start: int, z_start: int,
                           x_size: int, y_size: int, z_size: int):
        """Thêm khối chướng ngại vật hình hộp (ví dụ: tòa nhà).

        Parameters
        ----------
        x_start, y_start, z_start : int
            Góc dưới trái của khối
        x_size, y_size, z_size : int
            Kích thước khối theo 3 trục
        """
        for x in range(x_start, min(x_start + x_size, self.width)):
            for y in range(y_start, min(y_start + y_size, self.height)):
                for z in range(z_start, min(z_start + z_size, self.depth)):
                    self.grid[x, y, z] = CellType.OBSTACLE

    def add_no_fly_zone_block(self, x_start: int, y_start: int, z_start: int,
                              x_size: int, y_size: int, z_size: int):
        """Thêm vùng cấm bay hình hộp."""
        for x in range(x_start, min(x_start + x_size, self.width)):
            for y in range(y_start, min(y_start + y_size, self.height)):
                for z in range(z_start, min(z_start + z_size, self.depth)):
                    self.grid[x, y, z] = CellType.NO_FLY_ZONE

    def is_free(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra ô có thể bay qua không.

        Đây là phần kiểm tra trước khi expand node trong
        Graph-Search (Ch3). Chỉ expand node FREE hoặc GOAL.
        """
        if not self._is_valid_position(x, y, z):
            return False
        cell = self.grid[x, y, z]
        return cell in (CellType.FREE, CellType.START, CellType.GOAL)

    def is_goal(self, x: int, y: int, z: int) -> bool:
        """Goal Test (Ch3): Kiểm tra đã đến đích chưa."""
        return (x, y, z) == self.goal

    def get_neighbors(self, x: int, y: int, z: int,
                      use_26_directions: bool = False) -> List[Tuple[int, int, int]]:
        """Lấy các ô lân cận có thể di chuyển đến.

        Đây là hàm EXPAND trong Graph-Search (Ch3):
        Từ node hiện tại, tìm tất cả node con hợp lệ
        (trong lưới VÀ không có chướng ngại vật).

        Parameters
        ----------
        x, y, z : int
            Vị trí hiện tại
        use_26_directions : bool
            True = 26 hướng (gồm chéo), False = 6 hướng cơ bản

        Returns
        -------
        List[Tuple[int, int, int]]
            Danh sách các vị trí lân cận hợp lệ
        """
        actions = self.ACTIONS_26 if use_26_directions else self.ACTIONS_6
        neighbors = []

        for dx, dy, dz in actions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.is_free(nx, ny, nz):
                neighbors.append((nx, ny, nz))

        return neighbors

    def get_move_cost(self, from_pos: Tuple[int, int, int],
                      to_pos: Tuple[int, int, int]) -> float:
        """Tính chi phí di chuyển giữa 2 ô liền kề.

        Đây là PATH COST trong formulation bài toán (Ch3):
        - Di chuyển thẳng: cost = 1.0
        - Di chuyển chéo 2D: cost = sqrt(2) ≈ 1.414
        - Di chuyển chéo 3D: cost = sqrt(3) ≈ 1.732
        - Leo cao (dz > 0): cost nhân thêm hệ số (tốn năng lượng hơn)

        Parameters
        ----------
        from_pos, to_pos : Tuple[int, int, int]
            Vị trí xuất phát và đích

        Returns
        -------
        float
            Chi phí di chuyển
        """
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        dz = abs(to_pos[2] - from_pos[2])

        # Khoảng cách Euclidean cơ bản
        base_cost = np.sqrt(dx**2 + dy**2 + dz**2)

        # Phạt khi leo cao (UAV tốn năng lượng hơn khi bay lên)
        altitude_penalty = 0.0
        if to_pos[2] > from_pos[2]:
            altitude_penalty = 0.5 * dz  # Tăng 50% khi bay lên

        return base_cost + altitude_penalty

    def set_wind(self, wx: float, wy: float, wz: float):
        """Đặt vector gió (mô phỏng tính stochastic - Ch2).

        Gió ảnh hưởng đến chi phí bay: bay ngược gió tốn hơn.
        Đây thể hiện tính STOCHASTIC của môi trường (Ch2).
        """
        self.wind_vector = np.array([wx, wy, wz])

    def generate_random_obstacles(self, num_obstacles: int = 10,
                                  max_size: int = 3, seed: int = None):
        """Sinh chướng ngại vật ngẫu nhiên.

        Parameters
        ----------
        num_obstacles : int
            Số khối chướng ngại vật
        max_size : int
            Kích thước tối đa mỗi khối
        seed : int, optional
            Random seed để tái tạo
        """
        if seed is not None:
            np.random.seed(seed)

        for _ in range(num_obstacles):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            z = np.random.randint(0, self.depth)
            sx = np.random.randint(1, max_size + 1)
            sy = np.random.randint(1, max_size + 1)
            sz = np.random.randint(1, max_size + 1)
            self.add_obstacle_block(x, y, z, sx, sy, sz)

        # Đảm bảo start và goal không bị chặn
        if self.start:
            sx, sy, sz = self.start
            self.grid[sx, sy, sz] = CellType.START
        if self.goal:
            gx, gy, gz = self.goal
            self.grid[gx, gy, gz] = CellType.GOAL

    def get_stats(self) -> dict:
        """Thống kê môi trường."""
        total = self.width * self.height * self.depth
        free = np.sum(self.grid == CellType.FREE)
        obstacles = np.sum(self.grid == CellType.OBSTACLE)
        no_fly = np.sum(self.grid == CellType.NO_FLY_ZONE)

        return {
            "kích_thước": f"{self.width}x{self.height}x{self.depth}",
            "tổng_ô": total,
            "ô_trống": int(free),
            "chướng_ngại_vật": int(obstacles),
            "vùng_cấm_bay": int(no_fly),
            "tỷ_lệ_chướng_ngại": f"{obstacles/total*100:.1f}%",
            "start": self.start,
            "goal": self.goal,
            "gió": self.wind_vector.tolist(),
        }

    def load_from_config(self, config_path: str):
        """Đọc cấu hình môi trường từ file YAML.

        Parameters
        ----------
        config_path : str
            Đường dẫn đến file YAML
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_config = config.get('environment', {})

        # Kích thước
        size = env_config.get('size', {})
        self.width = size.get('width', self.width)
        self.height = size.get('height', self.height)
        self.depth = size.get('depth', self.depth)
        self.grid = np.full((self.width, self.height, self.depth),
                            CellType.FREE, dtype=np.int8)

        # Start và Goal
        start = env_config.get('start', {})
        if start:
            self.set_start(start['x'], start['y'], start['z'])

        goal = env_config.get('goal', {})
        if goal:
            self.set_goal(goal['x'], goal['y'], goal['z'])

        # Chướng ngại vật
        for obs in env_config.get('obstacles', []):
            self.add_obstacle_block(
                obs['x'], obs['y'], obs['z'],
                obs.get('sx', 1), obs.get('sy', 1), obs.get('sz', 1)
            )

        # Vùng cấm bay
        for nfz in env_config.get('no_fly_zones', []):
            self.add_no_fly_zone_block(
                nfz['x'], nfz['y'], nfz['z'],
                nfz.get('sx', 1), nfz.get('sy', 1), nfz.get('sz', 1)
            )

        # Gió
        wind = env_config.get('wind', {})
        if wind:
            self.set_wind(wind.get('x', 0), wind.get('y', 0), wind.get('z', 0))

    def _is_valid_position(self, x: int, y: int, z: int) -> bool:
        """Kiểm tra vị trí có nằm trong lưới không."""
        return (0 <= x < self.width and
                0 <= y < self.height and
                0 <= z < self.depth)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"GridWorld3D({stats['kích_thước']}) - "
                f"{stats['chướng_ngại_vật']} obstacles, "
                f"{stats['vùng_cấm_bay']} no-fly zones")
