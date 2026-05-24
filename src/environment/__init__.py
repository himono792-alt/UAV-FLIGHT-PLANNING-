"""
Module Environment - Môi trường mô phỏng 3D cho UAV
=====================================================
Lý thuyết: Chương 2 (PEAS, Environment types), Chương 3 (State Space)

Bao gồm:
- GridWorld3D: Không gian lưới 3D (State Space - Ch3)
- Obstacles: Chướng ngại vật và vùng cấm bay
- Visualizer: Hiển thị trực quan 3D
"""

from .grid_world import GridWorld3D
from .city_grid import CityGrid3D
from .obstacles import Obstacle, NoFlyZone, ObstacleManager
