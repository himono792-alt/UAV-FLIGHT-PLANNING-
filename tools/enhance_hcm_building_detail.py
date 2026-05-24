"""Enhance the baked HCM demo with denser building detail from OSM caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.city_grid import CityGrid3D
from src.geodata.osm_loader import (
    EARTH_METRES_PER_DEGREE_LAT,
    bbox_size_m,
    make_bbox_projector,
    parse_osm_buildings,
)
from src.search.a_star import a_star_search


DEFAULT_BBOX = (10.7600, 106.6900, 10.8585, 106.7995)
DEFAULT_START_LONLAT = (106.7856729, 10.8477107)
DEFAULT_GOAL_LONLAT = (106.7009, 10.7769)


def _round_point(point: tuple[float, float]) -> list[float]:
    return [round(point[0], 2), round(point[1], 2)]


def _building_area(building) -> float:
    x0, y0, x1, y1 = building.area_bbox_m
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _detail_score(building) -> float:
    # Prioritize landmarks, high-rises, and large footprints for true polygons.
    height_factor = max(1.0, min(float(building.height_m), 140.0) / 12.0)
    named_bonus = 2.0 if building.tags.get("name") else 1.0
    return _building_area(building) * height_factor * named_bonus


def _building_to_scene(building) -> dict:
    bbox = building.area_bbox_m
    return {
        "id": building.osm_id,
        "height": round(float(building.height_m), 1),
        "height_source": building.height_source,
        "name": building.tags.get("name", ""),
        "bbox": [round(v, 2) for v in bbox],
        "polygon": [_round_point(p) for p in building.polygon_m],
    }


def _building_to_dense(building) -> dict:
    bbox = building.area_bbox_m
    return {
        "id": building.osm_id,
        "height": round(float(building.height_m), 1),
        "bbox": [round(v, 2) for v in bbox],
    }


def _inverse_projector(bbox):
    south, west, north, _east = bbox
    center_lat = (south + north) / 2.0
    metres_per_degree_lon = EARTH_METRES_PER_DEGREE_LAT * math.cos(math.radians(center_lat))

    def inverse(x_m: float, y_m: float) -> tuple[float, float]:
        lon = west + x_m / metres_per_degree_lon
        lat = south + y_m / EARTH_METRES_PER_DEGREE_LAT
        return lon, lat

    return inverse


def load_unique_buildings(cache_dir: Path, bbox) -> list:
    buildings = {}
    for path in sorted(cache_dir.glob("q1_thuduc_buildings_r*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for building in parse_osm_buildings(raw, bbox):
            buildings.setdefault(building.osm_id, building)
    return list(buildings.values())


def compute_grid_and_path(buildings: list, bbox, cell_xy_m: float, cell_z_m: float, max_altitude_m: float, clearance_m: float):
    width_m, height_m = bbox_size_m(bbox)
    grid = CityGrid3D(
        math.ceil(width_m / cell_xy_m),
        math.ceil(height_m / cell_xy_m),
        math.ceil(max_altitude_m / cell_z_m),
        cell_xy_m=cell_xy_m,
        cell_z_m=cell_z_m,
        allow_diagonal=True,
    )
    grid.add_buildings(buildings, clearance_m=clearance_m)

    project = make_bbox_projector(bbox)
    start_x, start_y = project(*DEFAULT_START_LONLAT)
    goal_x, goal_y = project(*DEFAULT_GOAL_LONLAT)
    cruise_altitude_m = 80.0
    start = grid.nearest_free(grid.world_to_grid(start_x, start_y, cruise_altitude_m), max_radius=30)
    goal = grid.nearest_free(grid.world_to_grid(goal_x, goal_y, cruise_altitude_m), max_radius=30)
    if start is None or goal is None:
        raise RuntimeError(f"Could not place mission endpoints: start={start}, goal={goal}")

    result = a_star_search(grid, start, goal, grid.euclidean_cost)
    if not result.found:
        raise RuntimeError("A* could not find a path on the dense building grid")

    return grid, result


def extract_scene_data(html: str) -> tuple[dict, int, int]:
    prefix = "const sceneData = "
    try:
        start = html.index(prefix) + len(prefix)
        end = html.index(";\n", start)
    except ValueError:
        raise RuntimeError("Could not find sceneData in HTML")
    return json.loads(html[start:end]), start, end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", default="outputs/hcm_3d_demo.html")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--detailed-limit", type=int, default=22_000)
    args = parser.parse_args()

    html_path = Path(args.html)
    cache_dir = Path(args.cache_dir)
    html = html_path.read_text(encoding="utf-8")
    scene, scene_start, scene_end = extract_scene_data(html)

    bbox = tuple(scene.get("bbox") or DEFAULT_BBOX)
    cell_xy_m = float(scene.get("cell_xy_m", 40.0))
    cell_z_m = float(scene.get("cell_z_m", 10.0))
    max_altitude_m = float(scene.get("stats", {}).get("planned_altitude_m", 80.0))
    max_altitude_m = max(max_altitude_m, float(scene.get("grid_depth", 26) - 1) * cell_z_m + cell_z_m)
    clearance_m = float(scene.get("stats", {}).get("safety_clearance_m", 20.0))

    buildings = load_unique_buildings(cache_dir, bbox)
    buildings.sort(key=_detail_score, reverse=True)

    detailed_limit = max(0, min(args.detailed_limit, len(buildings)))
    detailed = buildings[:detailed_limit]
    dense = buildings[detailed_limit:]

    grid, result = compute_grid_and_path(buildings, bbox, cell_xy_m, cell_z_m, max_altitude_m, clearance_m)
    inverse = _inverse_projector(bbox)

    path_m = [[round(v, 2) for v in grid.grid_to_world(p)] for p in result.path]
    path_lonlat = [
        [round(lon, 7), round(lat, 7), round(z, 2)]
        for x, y, z in path_m
        for lon, lat in [inverse(x, y)]
    ]

    scene["buildings"] = [_building_to_scene(b) for b in detailed]
    scene["dense_buildings"] = [_building_to_dense(b) for b in dense]
    scene["blocked_cells_grid"] = [list(cell) for cell in sorted(grid.obstacles)]
    scene["grid_width"] = grid.width
    scene["grid_height"] = grid.height
    scene["grid_depth"] = grid.depth
    scene["start_grid"] = list(result.path[0])
    scene["goal_grid"] = list(result.path[-1])
    scene["start_m"] = path_m[0]
    scene["goal_m"] = path_m[-1]
    scene["path_grid"] = [list(p) for p in result.path]
    scene["path_m"] = path_m
    scene["path_lonlat"] = path_lonlat

    stats = scene.setdefault("stats", {})
    stats["buildings_loaded"] = len(buildings)
    stats["buildings_detailed"] = len(detailed)
    stats["buildings_dense"] = len(dense)
    stats["blocked_cells"] = len(grid.obstacles)
    stats["obstacle_density"] = round(grid.obstacle_density(), 4)
    stats["path_found"] = result.found
    stats["path_points"] = len(result.path)
    stats["path_cost_m"] = round(result.cost, 2)
    stats["nodes_expanded"] = result.nodes_expanded
    stats["search_time_ms"] = round(result.time_ms, 2)
    stats["data_source"] = "OpenStreetMap/Overpass tiled dense"
    stats["max_building_height_m"] = round(max(float(b.height_m) for b in buildings), 1)

    new_scene = json.dumps(scene, ensure_ascii=False, separators=(",", ":"))
    html_path.write_text(html[:scene_start] + new_scene + html[scene_end:], encoding="utf-8")

    print(
        f"Enhanced {html_path}: {len(detailed)} detailed buildings, "
        f"{len(dense)} dense instances, {len(grid.obstacles)} blocked cells, "
        f"path {len(result.path)} points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
