import math

from src.environment.city_grid import CityGrid3D, point_in_polygon
from src.geodata.osm_loader import (
    infer_building_height,
    parse_length_m,
    parse_osm_map_features,
    parse_osm_buildings,
)
from src.search.a_star import a_star_search


def test_parse_length_m_common_osm_values():
    assert parse_length_m("12 m") == 12
    assert parse_length_m("18.5") == 18.5
    assert parse_length_m("10,5 m") == 10.5
    assert math.isclose(parse_length_m("30 ft"), 9.144)
    assert parse_length_m("unknown") is None


def test_infer_building_height_priority_order():
    assert infer_building_height({"height": "42 m", "building:levels": "5"}) == (42.0, "height")
    assert infer_building_height({"building:levels": "4"}) == (12.8, "building:levels")
    assert infer_building_height({"building": "office"}) == (24.0, "default:office")


def test_parse_osm_buildings_projects_closed_way():
    raw = {
        "elements": [
            {
                "type": "way",
                "id": 123,
                "tags": {"building": "yes", "building:levels": "3"},
                "geometry": [
                    {"lat": 10.0, "lon": 106.0},
                    {"lat": 10.0, "lon": 106.0001},
                    {"lat": 10.0001, "lon": 106.0001},
                    {"lat": 10.0001, "lon": 106.0},
                    {"lat": 10.0, "lon": 106.0},
                ],
            }
        ]
    }
    buildings = parse_osm_buildings(raw, (10.0, 106.0, 10.001, 106.001))
    assert len(buildings) == 1
    assert buildings[0].osm_id == 123
    assert math.isclose(buildings[0].height_m, 9.6)
    assert buildings[0].polygon_m[0] == (0.0, 0.0)


def test_city_grid_voxelizes_building_height_and_clearance():
    grid = CityGrid3D(6, 6, 5, cell_xy_m=20, cell_z_m=10, allow_diagonal=False)
    polygon = [(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)]

    added = grid.add_building_polygon(polygon, height_m=15, clearance_m=5)

    assert added > 0
    assert point_in_polygon(30, 30, polygon)
    assert (1, 1, 0) in grid.obstacles
    assert (1, 1, 1) in grid.obstacles
    assert (1, 1, 2) not in grid.obstacles


def test_a_star_can_use_city_grid_interface():
    grid = CityGrid3D(8, 5, 3, cell_xy_m=10, cell_z_m=10, allow_diagonal=False)
    wall = [(30, 0), (40, 0), (40, 50), (30, 50), (30, 0)]
    grid.add_building_polygon(wall, height_m=10)

    start = (0, 2, 0)
    goal = (7, 2, 0)
    result = a_star_search(grid, start, goal, grid.euclidean_cost)

    assert result.found
    assert result.path[0] == start
    assert result.path[-1] == goal
    assert max(p[2] for p in result.path) > 0


def test_parse_osm_map_features_classifies_roads_and_water():
    raw = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "primary"},
                "geometry": [
                    {"lat": 10.0, "lon": 106.0},
                    {"lat": 10.0001, "lon": 106.0001},
                ],
            },
            {
                "type": "way",
                "id": 2,
                "tags": {"natural": "water"},
                "geometry": [
                    {"lat": 10.0, "lon": 106.0},
                    {"lat": 10.0, "lon": 106.0001},
                    {"lat": 10.0001, "lon": 106.0001},
                    {"lat": 10.0, "lon": 106.0},
                ],
            },
        ]
    }
    features = parse_osm_map_features(raw, (10.0, 106.0, 10.001, 106.001))

    assert [f.kind for f in features] == ["road:primary", "water"]
    assert features[1].closed


def test_parse_osm_map_features_classifies_green_and_campus():
    raw = {
        "elements": [
            {
                "type": "way",
                "id": 3,
                "tags": {"leisure": "park", "name": "Park"},
                "geometry": [
                    {"lat": 10.0, "lon": 106.0},
                    {"lat": 10.0, "lon": 106.0001},
                    {"lat": 10.0001, "lon": 106.0001},
                    {"lat": 10.0, "lon": 106.0},
                ],
            },
            {
                "type": "way",
                "id": 4,
                "tags": {"amenity": "university", "name": "Campus"},
                "geometry": [
                    {"lat": 10.0, "lon": 106.0},
                    {"lat": 10.0001, "lon": 106.0001},
                ],
            },
        ]
    }
    features = parse_osm_map_features(raw, (10.0, 106.0, 10.001, 106.001))

    assert [f.kind for f in features] == ["green", "campus"]
