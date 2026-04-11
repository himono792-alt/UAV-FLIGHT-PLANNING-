"""
Visualizer - Hiển thị trực quan 3D
====================================
Hiển thị môi trường và đường bay bằng Matplotlib 3D.

Mục đích: Giúp DEMO trực quan khi bảo vệ đồ án.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional

from .grid_world import GridWorld3D, CellType


class Visualizer:
    """Hiển thị môi trường 3D và đường bay.

    Sử dụng Matplotlib 3D để vẽ:
    - Chướng ngại vật (đỏ)
    - Vùng cấm bay (cam)
    - Điểm Start (xanh lá)
    - Điểm Goal (xanh dương)
    - Đường bay (đường nét đứt)
    """

    # Bảng màu
    COLORS = {
        'obstacle': 'red',
        'no_fly_zone': 'orange',
        'start': 'lime',
        'goal': 'dodgerblue',
        'path_astar': 'blue',
        'path_sa': 'green',
        'path_ga': 'red',
        'free': 'lightgray',
    }

    def __init__(self, world: GridWorld3D):
        self.world = world

    def plot_environment(self, title: str = "Môi trường UAV 3D",
                         show_grid: bool = False,
                         paths: Optional[dict] = None,
                         save_path: Optional[str] = None):
        """Vẽ môi trường 3D.

        Parameters
        ----------
        title : str
            Tiêu đề biểu đồ
        show_grid : bool
            Hiển thị lưới nền
        paths : dict, optional
            Dictionary {tên_đường: [(x,y,z), ...]}
            Ví dụ: {"A*": path1, "A*+SA": path2, "GA": path3}
        save_path : str, optional
            Đường dẫn lưu hình ảnh
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Vẽ chướng ngại vật (đỏ)
        obs_x, obs_y, obs_z = [], [], []
        nfz_x, nfz_y, nfz_z = [], [], []

        for x in range(self.world.width):
            for y in range(self.world.height):
                for z in range(self.world.depth):
                    cell = self.world.grid[x, y, z]
                    if cell == CellType.OBSTACLE:
                        obs_x.append(x)
                        obs_y.append(y)
                        obs_z.append(z)
                    elif cell == CellType.NO_FLY_ZONE:
                        nfz_x.append(x)
                        nfz_y.append(y)
                        nfz_z.append(z)

        if obs_x:
            ax.scatter(obs_x, obs_y, obs_z, c=self.COLORS['obstacle'],
                       marker='s', s=50, alpha=0.6, label='Chướng ngại vật')

        if nfz_x:
            ax.scatter(nfz_x, nfz_y, nfz_z, c=self.COLORS['no_fly_zone'],
                       marker='s', s=50, alpha=0.4, label='Vùng cấm bay')

        # Vẽ Start và Goal
        if self.world.start:
            sx, sy, sz = self.world.start
            ax.scatter([sx], [sy], [sz], c=self.COLORS['start'],
                       marker='*', s=200, label=f'Start ({sx},{sy},{sz})')

        if self.world.goal:
            gx, gy, gz = self.world.goal
            ax.scatter([gx], [gy], [gz], c=self.COLORS['goal'],
                       marker='*', s=200, label=f'Goal ({gx},{gy},{gz})')

        # Vẽ đường bay (nếu có)
        if paths:
            path_colors = ['blue', 'green', 'red', 'purple', 'cyan']
            for i, (name, path) in enumerate(paths.items()):
                if path:
                    px = [p[0] for p in path]
                    py = [p[1] for p in path]
                    pz = [p[2] for p in path]
                    color = path_colors[i % len(path_colors)]
                    ax.plot(px, py, pz, color=color, linewidth=2,
                            linestyle='-', marker='.', markersize=3,
                            label=f'{name} ({len(path)} bước)')

        # Cài đặt trục
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Độ cao)')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)

        # Giới hạn trục
        ax.set_xlim(0, self.world.width)
        ax.set_ylim(0, self.world.height)
        ax.set_zlim(0, self.world.depth)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Đã lưu hình: {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_path_comparison(self, paths: dict,
                             title: str = "So sánh các thuật toán",
                             save_path: Optional[str] = None):
        """Vẽ so sánh nhiều đường bay trên cùng 1 bản đồ.

        Dùng để so sánh: A* vs A*+SA vs GA (Giai đoạn 3).

        Parameters
        ----------
        paths : dict
            {"A*": path1, "A*+SA": path2, "GA": path3}
        """
        self.plot_environment(title=title, paths=paths, save_path=save_path)

    def plot_2d_slice(self, z_level: int = 0,
                      path: Optional[List[Tuple[int, int, int]]] = None,
                      title: str = "Lát cắt 2D"):
        """Vẽ lát cắt 2D tại một độ cao z.

        Hữu ích khi bản đồ 3D quá phức tạp để nhìn.

        Parameters
        ----------
        z_level : int
            Độ cao cắt
        path : list, optional
            Đường bay để vẽ trên lát cắt
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Lấy lát cắt tại z_level
        slice_2d = self.world.grid[:, :, z_level].T  # Transpose để Y lên trên

        # Tạo colormap
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['white', 'red', 'orange', 'lime', 'dodgerblue'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(slice_2d, cmap=cmap, norm=norm, origin='lower')

        # Vẽ đường bay trên lát cắt này
        if path:
            path_on_slice = [(p[0], p[1]) for p in path if p[2] == z_level]
            if path_on_slice:
                px = [p[0] for p in path_on_slice]
                py = [p[1] for p in path_on_slice]
                ax.plot(px, py, 'b-o', linewidth=2, markersize=4, label='Đường bay')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{title} (z = {z_level})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
