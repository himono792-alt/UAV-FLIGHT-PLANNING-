"""
Visualizer - Hiển thị môi trường 3D cho UAV
Dùng Matplotlib 3D (không cần Pygame)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Màu sắc
COLOR_OBSTACLE  = (0.8, 0.2, 0.2, 0.6)   # đỏ
COLOR_NFZ       = (1.0, 0.5, 0.0, 0.4)   # cam
COLOR_START     = (0.2, 0.8, 0.2, 1.0)   # xanh lá
COLOR_GOAL      = (0.2, 0.4, 1.0, 1.0)   # xanh dương
COLOR_PATH      = (1.0, 0.8, 0.0, 0.9)   # vàng
COLOR_FREE      = (0.9, 0.9, 0.9, 0.05)  # trắng mờ


def _voxel_faces(x, y, z):
    """Tạo 6 mặt của 1 voxel đơn vị tại (x, y, z)"""
    return [
        [(x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z)],
        [(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)],
        [(x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1)],
        [(x,y+1,z),(x+1,y+1,z),(x+1,y+1,z+1),(x,y+1,z+1)],
        [(x,y,z),(x,y+1,z),(x,y+1,z+1),(x,y,z+1)],
        [(x+1,y,z),(x+1,y+1,z),(x+1,y+1,z+1),(x+1,y,z+1)],
    ]


class UAVVisualizer:
    """
    Vẽ môi trường 3D với Matplotlib

    Màu quy ước:
    - Đỏ:      Obstacle
    - Cam:     No-Fly Zone
    - Xanh lá: Điểm bắt đầu (Start)
    - Xanh dương: Điểm đích (Goal)
    - Vàng:    Đường bay (Path)
    """

    def __init__(self, grid, title: str = "UAV Environment 3D"):
        self.grid = grid
        self.title = title

    def _add_voxels(self, ax, cells, color):
        faces = []
        for (x, y, z) in cells:
            faces.extend(_voxel_faces(x, y, z))
        if faces:
            poly = Poly3DCollection(faces, alpha=color[3])
            poly.set_facecolor(color[:3])
            poly.set_edgecolor((0, 0, 0, 0.1))
            ax.add_collection3d(poly)

    def _setup_axes(self, ax):
        g = self.grid
        ax.set_xlim(0, g.width)
        ax.set_ylim(0, g.height)
        ax.set_zlim(0, g.depth)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z (Altitude)")
        ax.set_title(self.title)

    def render(self, path: list = None, show: bool = True, save_path: str = None):
        """
        Vẽ toàn bộ môi trường

        Args:
            path:      Danh sách tọa độ [(x,y,z), ...] — đường bay
            show:      True để hiện cửa sổ
            save_path: Đường dẫn lưu ảnh (vd: 'output.png')
        """
        from src.environment.grid_world import CellType

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)

        g = self.grid

        # Thu thập cells theo loại
        obstacles, nfz_cells, start_cells, goal_cells = [], [], [], []
        for x in range(g.width):
            for y in range(g.height):
                for z in range(g.depth):
                    ct = CellType(g.grid[x, y, z])
                    if ct == CellType.OBSTACLE:
                        obstacles.append((x, y, z))
                    elif ct == CellType.NO_FLY_ZONE:
                        nfz_cells.append((x, y, z))
                    elif ct == CellType.START:
                        start_cells.append((x, y, z))
                    elif ct == CellType.GOAL:
                        goal_cells.append((x, y, z))

        self._add_voxels(ax, obstacles,   COLOR_OBSTACLE)
        self._add_voxels(ax, nfz_cells,   COLOR_NFZ)
        self._add_voxels(ax, start_cells, COLOR_START)
        self._add_voxels(ax, goal_cells,  COLOR_GOAL)

        # Vẽ đường bay
        if path and len(path) > 1:
            xs = [p[0] + 0.5 for p in path]
            ys = [p[1] + 0.5 for p in path]
            zs = [p[2] + 0.5 for p in path]
            ax.plot(xs, ys, zs, color=COLOR_PATH[:3],
                    linewidth=2.5, marker='o', markersize=3, label="Path")
            # Đánh dấu start & goal
            ax.scatter(xs[0],  ys[0],  zs[0],  color='green', s=100, zorder=5)
            ax.scatter(xs[-1], ys[-1], zs[-1], color='blue',  s=100, zorder=5)

        # Legend thủ công
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor=COLOR_OBSTACLE[:3], label='Obstacle'),
            Patch(facecolor=COLOR_NFZ[:3],      label='No-Fly Zone'),
            Patch(facecolor=COLOR_START[:3],    label='Start'),
            Patch(facecolor=COLOR_GOAL[:3],     label='Goal'),
        ]
        if path:
            from matplotlib.lines import Line2D
            legend.append(Line2D([0],[0], color=COLOR_PATH[:3],
                                 linewidth=2, label='Path'))
        ax.legend(handles=legend, loc='upper left')

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Đã lưu ảnh: {save_path}")
        if show:
            plt.tight_layout()
            plt.show()
        plt.close(fig)

    def render_slice(self, z_level: int = 0, path: list = None,
                     show: bool = True, save_path: str = None):
        """
        Vẽ lát cắt 2D tại độ cao z_level (nhanh hơn render 3D)
        Hữu ích để debug khi không có màn hình 3D
        """
        from src.environment.grid_world import CellType

        g = self.grid
        fig, ax = plt.subplots(figsize=(10, 8))

        color_map = {
            CellType.FREE:        'white',
            CellType.OBSTACLE:    'red',
            CellType.NO_FLY_ZONE: 'orange',
            CellType.START:       'green',
            CellType.GOAL:        'blue',
        }

        for x in range(g.width):
            for y in range(g.height):
                ct = CellType(g.grid[x, y, z_level])
                rect = plt.Rectangle((x, y), 1, 1,
                                     facecolor=color_map[ct],
                                     edgecolor='lightgray', linewidth=0.3)
                ax.add_patch(rect)

        # Vẽ đường bay trên lát cắt
        if path:
            path_z = [(p[0]+0.5, p[1]+0.5) for p in path if p[2] == z_level]
            if len(path_z) > 1:
                xs, ys = zip(*path_z)
                ax.plot(xs, ys, 'y-o', linewidth=2, markersize=4, label='Path')

        ax.set_xlim(0, g.width)
        ax.set_ylim(0, g.height)
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{self.title} — Slice z={z_level}")

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        if show:
            plt.tight_layout()
            plt.show()
        plt.close(fig)
