"""
Test Giai đoạn 1: Môi trường mô phỏng 3D
Chạy: python -m pytest tests/test_environment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import pytest
except ImportError:
    import unittest as pytest  # fallback khi chưa cài pytest

from src.environment.grid_world import GridWorld3D, CellType
from src.environment.obstacles import Obstacle, NoFlyZone, ObstacleManager


# ======================================================================
#  GridWorld3D
# ======================================================================

class TestGridWorld3D:

    def test_khoi_tao(self):
        g = GridWorld3D(10, 10, 5)
        assert g.width == 10
        assert g.height == 10
        assert g.depth == 5
        # Tất cả ô ban đầu là FREE
        assert (g.grid == CellType.FREE).all()

    def test_in_bounds(self):
        g = GridWorld3D(10, 10, 5)
        assert g.in_bounds((0, 0, 0))
        assert g.in_bounds((9, 9, 4))
        assert not g.in_bounds((-1, 0, 0))
        assert not g.in_bounds((10, 0, 0))
        assert not g.in_bounds((0, 0, 5))

    def test_set_get_cell(self):
        g = GridWorld3D(10, 10, 5)
        g.set_cell((3, 3, 2), CellType.OBSTACLE)
        assert g.get_cell((3, 3, 2)) == CellType.OBSTACLE

    def test_start_goal(self):
        g = GridWorld3D(10, 10, 5)
        g.set_start((0, 0, 0))
        g.set_goal((9, 9, 4))
        assert g.start == (0, 0, 0)
        assert g.goal == (9, 9, 4)
        assert g.get_cell((0, 0, 0)) == CellType.START
        assert g.get_cell((9, 9, 4)) == CellType.GOAL

    def test_is_free(self):
        g = GridWorld3D(10, 10, 5)
        g.add_obstacle((5, 5, 2))
        assert g.is_free((4, 5, 2))
        assert not g.is_free((5, 5, 2))
        assert not g.is_free((-1, 0, 0))  # ngoài biên

    def test_add_obstacle_box(self):
        g = GridWorld3D(10, 10, 5)
        g.add_obstacle_box(2, 2, 0, 4, 4, 2)
        # Kiểm tra góc trong hộp
        assert g.get_cell((2, 2, 0)) == CellType.OBSTACLE
        assert g.get_cell((4, 4, 2)) == CellType.OBSTACLE
        # Kiểm tra ngoài hộp
        assert g.get_cell((5, 5, 3)) == CellType.FREE

    def test_get_neighbors_6_directions(self):
        g = GridWorld3D(10, 10, 5)
        neighbors = g.get_neighbors((5, 5, 2))
        # Ô giữa không có obstacle -> 6 hàng xóm
        assert len(neighbors) == 6
        assert (6, 5, 2) in neighbors
        assert (4, 5, 2) in neighbors
        assert (5, 6, 2) in neighbors
        assert (5, 4, 2) in neighbors
        assert (5, 5, 3) in neighbors
        assert (5, 5, 1) in neighbors

    def test_get_neighbors_blocked(self):
        g = GridWorld3D(10, 10, 5)
        g.add_obstacle((6, 5, 2))
        neighbors = g.get_neighbors((5, 5, 2))
        assert (6, 5, 2) not in neighbors

    def test_get_neighbors_corner(self):
        g = GridWorld3D(10, 10, 5)
        # Góc (0,0,0) chỉ có 3 hàng xóm hợp lệ
        neighbors = g.get_neighbors((0, 0, 0))
        assert len(neighbors) == 3

    def test_step_cost(self):
        g = GridWorld3D(10, 10, 5)
        # Đi thẳng ngang: cost = 1.0
        cost_h = g.step_cost((0, 0, 0), (1, 0, 0))
        assert abs(cost_h - 1.0) < 1e-6
        # Leo lên: cost > 1.0 (có vertical_penalty)
        cost_up = g.step_cost((0, 0, 0), (0, 0, 1))
        assert cost_up > 1.0

    def test_random_obstacles(self):
        g = GridWorld3D(20, 20, 10)
        g.add_random_obstacles(50, seed=42)
        blocked = (g.grid == CellType.OBSTACLE).sum()
        assert blocked <= 50  # có thể ít hơn nếu trùng

    def test_obstacle_density(self):
        g = GridWorld3D(10, 10, 5)
        g.add_obstacle_box(0, 0, 0, 4, 4, 4)
        density = g.obstacle_density()
        assert 0 < density < 1


# ======================================================================
#  Obstacles
# ======================================================================

class TestObstacle:

    def test_is_collision(self):
        obs = Obstacle(x=5, y=5, z=2, size_x=2, size_y=2, size_z=2)
        assert obs.is_collision((5, 5, 2))
        assert obs.is_collision((6, 6, 3))
        assert not obs.is_collision((7, 5, 2))
        assert not obs.is_collision((5, 5, 4))

    def test_get_cells(self):
        obs = Obstacle(x=0, y=0, z=0, size_x=2, size_y=2, size_z=2)
        cells = obs.get_cells()
        assert len(cells) == 8
        assert (0, 0, 0) in cells
        assert (1, 1, 1) in cells

    def test_dynamic_move(self):
        obs = Obstacle(x=5, y=5, z=0, is_dynamic=True)
        obs.move(dx=1, dy=0)
        assert obs.x == 6
        assert obs.y == 5

    def test_static_no_move(self):
        obs = Obstacle(x=5, y=5, z=0, is_dynamic=False)
        obs.move(dx=1)
        assert obs.x == 5  # static không di chuyển


class TestNoFlyZone:

    def test_contains(self):
        nfz = NoFlyZone(x1=3, y1=3, z1=0, x2=7, y2=7, z2=4)
        assert nfz.contains((5, 5, 2))
        assert nfz.contains((3, 3, 0))
        assert not nfz.contains((2, 5, 2))
        assert not nfz.contains((5, 5, 5))


# ======================================================================
#  ObstacleManager
# ======================================================================

class TestObstacleManager:

    def test_apply_to_grid(self):
        g = GridWorld3D(10, 10, 5)
        mgr = ObstacleManager()
        mgr.add_obstacle(Obstacle(x=3, y=3, z=0))
        mgr.add_no_fly_zone(NoFlyZone(x1=6, y1=6, z1=0, x2=8, y2=8, z2=2))
        mgr.apply_to_grid(g)
        assert g.get_cell((3, 3, 0)) == CellType.OBSTACLE
        assert g.get_cell((7, 7, 1)) == CellType.NO_FLY_ZONE
        assert g.get_cell((0, 0, 0)) == CellType.FREE

    def test_is_collision(self):
        mgr = ObstacleManager()
        mgr.add_obstacle(Obstacle(x=5, y=5, z=2, size_x=2, size_y=2, size_z=2))
        assert mgr.is_collision((5, 5, 2))
        assert not mgr.is_collision((3, 3, 0))

    def test_is_safe(self):
        mgr = ObstacleManager()
        mgr.add_obstacle(Obstacle(x=5, y=5, z=0))
        mgr.add_no_fly_zone(NoFlyZone(x1=8, y1=8, z1=0, x2=9, y2=9, z2=4))
        assert mgr.is_safe((0, 0, 0))
        assert not mgr.is_safe((5, 5, 0))   # obstacle
        assert not mgr.is_safe((8, 8, 2))   # no-fly zone

    def test_add_building(self):
        mgr = ObstacleManager()
        mgr.add_building(x=4, y=4, height=5, size_x=2, size_y=2)
        assert mgr.is_collision((4, 4, 0))
        assert mgr.is_collision((5, 5, 4))
        assert not mgr.is_collision((4, 4, 5))  # trên đỉnh nhà
