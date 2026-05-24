"""Sparse 3D city grid built from projected building footprints."""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Optional

from src.environment.grid_world import CellType, DIRECTIONS_6, DIRECTIONS_26


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _cell_intersects_polygon(
    gx: int,
    gy: int,
    cell_size: float,
    polygon: list[tuple[float, float]],
) -> bool:
    min_x = gx * cell_size
    min_y = gy * cell_size
    max_x = min_x + cell_size
    max_y = min_y + cell_size

    samples = [
        ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ]
    if any(point_in_polygon(x, y, polygon) for x, y in samples):
        return True

    return any(min_x <= px <= max_x and min_y <= py <= max_y for px, py in polygon)


class CityGrid3D:
    """Sparse grid compatible with the existing search algorithms.

    Coordinates are grid cells `(x, y, z)`. Metadata keeps the real-world cell
    dimensions so costs and exports can remain in metres.
    """

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        cell_xy_m: float = 25.0,
        cell_z_m: float = 10.0,
        allow_diagonal: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        self.cell_xy_m = float(cell_xy_m)
        self.cell_z_m = float(cell_z_m)
        self.allow_diagonal = allow_diagonal
        self.directions = DIRECTIONS_26 if allow_diagonal else DIRECTIONS_6

        self.obstacles: set[tuple[int, int, int]] = set()
        self.no_fly_zones: set[tuple[int, int, int]] = set()
        self.start: Optional[tuple[int, int, int]] = None
        self.goal: Optional[tuple[int, int, int]] = None
        self.building_count = 0

    def in_bounds(self, pos: tuple[int, int, int]) -> bool:
        x, y, z = pos
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def is_free(self, pos: tuple[int, int, int]) -> bool:
        if not self.in_bounds(pos):
            return False
        return pos not in self.obstacles and pos not in self.no_fly_zones

    def get_cell(self, pos: tuple[int, int, int]) -> CellType:
        if not self.in_bounds(pos):
            raise IndexError(f"Position {pos} out of bounds")
        if pos == self.start:
            return CellType.START
        if pos == self.goal:
            return CellType.GOAL
        if pos in self.no_fly_zones:
            return CellType.NO_FLY_ZONE
        if pos in self.obstacles:
            return CellType.OBSTACLE
        return CellType.FREE

    def set_cell(self, pos: tuple[int, int, int], cell_type: CellType):
        if not self.in_bounds(pos):
            raise IndexError(f"Position {pos} out of bounds")
        self.obstacles.discard(pos)
        self.no_fly_zones.discard(pos)
        if cell_type == CellType.OBSTACLE:
            self.obstacles.add(pos)
        elif cell_type == CellType.NO_FLY_ZONE:
            self.no_fly_zones.add(pos)

    def set_start(self, pos: tuple[int, int, int]):
        if not self.is_free(pos):
            raise ValueError(f"Start position {pos} is blocked")
        self.start = pos

    def set_goal(self, pos: tuple[int, int, int]):
        if not self.is_free(pos):
            raise ValueError(f"Goal position {pos} is blocked")
        self.goal = pos

    def world_to_grid(self, x_m: float, y_m: float, z_m: float) -> tuple[int, int, int]:
        return (
            max(0, min(self.width - 1, int(x_m // self.cell_xy_m))),
            max(0, min(self.height - 1, int(y_m // self.cell_xy_m))),
            max(0, min(self.depth - 1, int(z_m // self.cell_z_m))),
        )

    def grid_to_world(self, pos: tuple[int, int, int]) -> tuple[float, float, float]:
        x, y, z = pos
        return (
            (x + 0.5) * self.cell_xy_m,
            (y + 0.5) * self.cell_xy_m,
            (z + 0.5) * self.cell_z_m,
        )

    def get_neighbors(self, pos: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        neighbors = []
        for dx, dy, dz in self.directions:
            nxt = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if self.is_free(nxt):
                neighbors.append(nxt)
        return neighbors

    def step_cost(self, from_pos: tuple[int, int, int], to_pos: tuple[int, int, int]) -> float:
        dx = (to_pos[0] - from_pos[0]) * self.cell_xy_m
        dy = (to_pos[1] - from_pos[1]) * self.cell_xy_m
        dz = (to_pos[2] - from_pos[2]) * self.cell_z_m
        climb_penalty = max(0.0, dz) * 0.15
        return math.sqrt(dx * dx + dy * dy + dz * dz) + climb_penalty

    def euclidean_cost(self, node: tuple[int, int, int], goal: tuple[int, int, int]) -> float:
        dx = (goal[0] - node[0]) * self.cell_xy_m
        dy = (goal[1] - node[1]) * self.cell_xy_m
        dz = (goal[2] - node[2]) * self.cell_z_m
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def add_building_polygon(
        self,
        polygon_m: list[tuple[float, float]],
        height_m: float,
        clearance_m: float = 0.0,
    ) -> int:
        """Voxelize one building footprint into sparse blocked cells."""

        if len(polygon_m) < 4:
            return 0

        min_x = min(p[0] for p in polygon_m)
        max_x = max(p[0] for p in polygon_m)
        min_y = min(p[1] for p in polygon_m)
        max_y = max(p[1] for p in polygon_m)

        gx0 = max(0, int(math.floor(min_x / self.cell_xy_m)))
        gx1 = min(self.width - 1, int(math.floor(max_x / self.cell_xy_m)))
        gy0 = max(0, int(math.floor(min_y / self.cell_xy_m)))
        gy1 = min(self.height - 1, int(math.floor(max_y / self.cell_xy_m)))
        z_top = min(self.depth, max(1, int(math.ceil((height_m + clearance_m) / self.cell_z_m))))

        added = 0
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                if not _cell_intersects_polygon(gx, gy, self.cell_xy_m, polygon_m):
                    continue
                for gz in range(z_top):
                    cell = (gx, gy, gz)
                    if cell not in self.obstacles:
                        added += 1
                    self.obstacles.add(cell)

        if added:
            self.building_count += 1
        return added

    def add_buildings(self, buildings: Iterable, clearance_m: float = 0.0) -> int:
        added = 0
        for building in buildings:
            added += self.add_building_polygon(
                building.polygon_m,
                float(building.height_m),
                clearance_m=clearance_m,
            )
        return added

    def add_no_fly_box(self, x1, y1, z1, x2, y2, z2):
        for x in range(max(0, x1), min(self.width - 1, x2) + 1):
            for y in range(max(0, y1), min(self.height - 1, y2) + 1):
                for z in range(max(0, z1), min(self.depth - 1, z2) + 1):
                    self.no_fly_zones.add((x, y, z))

    def nearest_free(
        self,
        pos: tuple[int, int, int],
        max_radius: int = 20,
    ) -> Optional[tuple[int, int, int]]:
        """Return the nearest free cell using BFS in grid space."""

        if self.is_free(pos):
            return pos

        seen = {pos}
        queue = deque([(pos, 0)])
        while queue:
            current, dist = queue.popleft()
            if dist >= max_radius:
                continue
            for dx, dy, dz in DIRECTIONS_6:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nxt in seen or not self.in_bounds(nxt):
                    continue
                if self.is_free(nxt):
                    return nxt
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def obstacle_density(self) -> float:
        total = self.width * self.height * self.depth
        return (len(self.obstacles) + len(self.no_fly_zones)) / total if total else 0.0

    def summary(self) -> str:
        return (
            f"CityGrid3D ({self.width}x{self.height}x{self.depth}) "
            f"cell={self.cell_xy_m:.1f}m/{self.cell_z_m:.1f}m | "
            f"buildings={self.building_count} | "
            f"blocked={len(self.obstacles)} | density={self.obstacle_density():.1%}"
        )
