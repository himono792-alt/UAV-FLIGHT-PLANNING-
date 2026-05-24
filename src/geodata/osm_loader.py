"""Small OpenStreetMap loader for city-scale UAV planning demos.

The loader intentionally uses only Python's standard library so the existing
project can run without adding GIS-heavy dependencies. It fetches building ways
from Overpass, projects lon/lat to local metres, and infers LoD1 heights from
OSM tags.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


# OSM/Overpass bbox order: south, west, north, east.
BBox = tuple[float, float, float, float]

EARTH_METRES_PER_DEGREE_LAT = 111_320.0
DEFAULT_FLOOR_HEIGHT_M = 3.2


@dataclass(frozen=True)
class BuildingFootprint:
    """A projected building footprint with an estimated height."""

    osm_id: int
    polygon_m: list[tuple[float, float]]
    polygon_lonlat: list[tuple[float, float]]
    height_m: float
    height_source: str
    tags: dict

    @property
    def area_bbox_m(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon_m]
        ys = [p[1] for p in self.polygon_m]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class MapFeature:
    """A projected OSM feature used only for visual map context."""

    osm_id: int
    kind: str
    geometry_m: list[tuple[float, float]]
    closed: bool
    tags: dict


def parse_length_m(value) -> Optional[float]:
    """Parse common OSM length strings into metres."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    text = str(value).strip().lower().replace(",", ".")
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    metres = float(match.group(0))
    if metres <= 0:
        return None

    # Feet are uncommon in Vietnam, but this keeps the parser safe for reused
    # data in other locations.
    if "ft" in text or "feet" in text:
        metres *= 0.3048
    return metres


def _parse_levels(value) -> Optional[float]:
    levels = parse_length_m(value)
    if levels is None or levels <= 0:
        return None
    return levels


def infer_building_height(
    tags: dict,
    default_floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> tuple[float, str]:
    """Infer building height from OSM tags."""

    for key in ("height", "building:height", "est_height"):
        height = parse_length_m(tags.get(key))
        if height is not None:
            return height, key

    for key in ("building:levels", "levels", "building:level"):
        levels = _parse_levels(tags.get(key))
        if levels is not None:
            return levels * default_floor_height_m, key

    building_type = str(tags.get("building", "yes")).lower()
    defaults = {
        "apartments": 18.0,
        "commercial": 15.0,
        "construction": 10.0,
        "detached": 7.0,
        "garage": 4.0,
        "hospital": 18.0,
        "hotel": 24.0,
        "house": 7.0,
        "industrial": 10.0,
        "office": 24.0,
        "public": 12.0,
        "residential": 10.0,
        "retail": 8.0,
        "school": 10.0,
        "service": 5.0,
        "warehouse": 9.0,
        "yes": 12.0,
    }
    return defaults.get(building_type, 12.0), "default:" + building_type


def make_bbox_projector(bbox: BBox) -> Callable[[float, float], tuple[float, float]]:
    """Create a local equirectangular projector for a small bbox."""

    south, west, north, _east = bbox
    center_lat = (south + north) / 2.0
    metres_per_degree_lon = EARTH_METRES_PER_DEGREE_LAT * math.cos(math.radians(center_lat))

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - west) * metres_per_degree_lon
        y = (lat - south) * EARTH_METRES_PER_DEGREE_LAT
        return x, y

    return project


def bbox_size_m(bbox: BBox) -> tuple[float, float]:
    """Return approximate width/height of a bbox in metres."""

    south, west, north, east = bbox
    project = make_bbox_projector(bbox)
    x0, y0 = project(west, south)
    x1, y1 = project(east, north)
    return abs(x1 - x0), abs(y1 - y0)


def _overpass_query(bbox: BBox) -> str:
    south, west, north, east = bbox
    return f"""[out:json][timeout:180];
way["building"]({south},{west},{north},{east});
out tags geom;"""


def _map_features_query(bbox: BBox) -> str:
    south, west, north, east = bbox
    return f"""[out:json][timeout:180];
(
  way["highway"]({south},{west},{north},{east});
  way["waterway"]({south},{west},{north},{east});
  way["natural"="water"]({south},{west},{north},{east});
  way["water"]({south},{west},{north},{east});
  way["landuse"="reservoir"]({south},{west},{north},{east});
  way["leisure"~"^(park|garden|recreation_ground)$"]({south},{west},{north},{east});
  way["landuse"~"^(grass|forest|meadow|recreation_ground|cemetery)$"]({south},{west},{north},{east});
  way["natural"~"^(wood|grassland)$"]({south},{west},{north},{east});
  way["amenity"~"^(university|college|school)$"]({south},{west},{north},{east});
);
out tags geom;"""


def _fetch_overpass_json(query: str, endpoints: Iterable[str]) -> dict:
    encoded = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_error = None

    for endpoint in endpoints:
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "UAV-Flight-Planning-HCM-Demo/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network fallback
            last_error = exc

    raise RuntimeError(f"Overpass request failed: {last_error}")


def _valid_closed_polygon(points: list[tuple[float, float]]) -> bool:
    return len(points) >= 4 and points[0] == points[-1]


def parse_osm_buildings(raw: dict, bbox: BBox, max_buildings: Optional[int] = None) -> list[BuildingFootprint]:
    """Parse Overpass JSON into projected building footprints."""

    project = make_bbox_projector(bbox)
    buildings: list[BuildingFootprint] = []

    for element in raw.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        lonlat = [(float(p["lon"]), float(p["lat"])) for p in geometry if "lon" in p and "lat" in p]
        if not _valid_closed_polygon(lonlat):
            continue

        polygon_m = [project(lon, lat) for lon, lat in lonlat]
        height_m, height_source = infer_building_height(tags)
        buildings.append(
            BuildingFootprint(
                osm_id=int(element["id"]),
                polygon_m=polygon_m,
                polygon_lonlat=lonlat,
                height_m=height_m,
                height_source=height_source,
                tags=tags,
            )
        )

        if max_buildings is not None and len(buildings) >= max_buildings:
            break

    return buildings


def parse_osm_map_features(raw: dict, bbox: BBox, max_features: Optional[int] = None) -> list[MapFeature]:
    """Parse visual road/water OSM ways into projected map features."""

    project = make_bbox_projector(bbox)
    features: list[MapFeature] = []

    for element in raw.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags") or {}
        geometry = element.get("geometry") or []
        points = [
            project(float(p["lon"]), float(p["lat"]))
            for p in geometry
            if "lon" in p and "lat" in p
        ]
        if len(points) < 2:
            continue

        if "highway" in tags:
            kind = "road:" + str(tags.get("highway", "road"))
        elif tags.get("natural") == "water" or "water" in tags or tags.get("landuse") == "reservoir":
            kind = "water"
        elif "waterway" in tags:
            kind = "waterway:" + str(tags.get("waterway", "waterway"))
        elif tags.get("amenity") in {"university", "college", "school"}:
            kind = "campus"
        elif (
            tags.get("leisure") in {"park", "garden", "recreation_ground"}
            or tags.get("landuse") in {"grass", "forest", "meadow", "recreation_ground", "cemetery"}
            or tags.get("natural") in {"wood", "grassland"}
        ):
            kind = "green"
        else:
            continue

        closed = len(points) >= 4 and points[0] == points[-1]
        features.append(
            MapFeature(
                osm_id=int(element["id"]),
                kind=kind,
                geometry_m=points,
                closed=closed,
                tags=tags,
            )
        )

        if max_features is not None and len(features) >= max_features:
            break

    return features


def load_osm_buildings(
    bbox: BBox,
    cache_path: Optional[str | Path] = None,
    use_cache: bool = True,
    max_buildings: Optional[int] = None,
    endpoints: Optional[list[str]] = None,
    projection_bbox: Optional[BBox] = None,
) -> list[BuildingFootprint]:
    """Load building footprints from cache or Overpass."""

    raw = None
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and use_cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))

    if raw is None:
        query = _overpass_query(bbox)
        raw = _fetch_overpass_json(
            query,
            endpoints
            or [
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass-api.de/api/interpreter",
            ],
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw), encoding="utf-8")

    return parse_osm_buildings(raw, projection_bbox or bbox, max_buildings=max_buildings)


def load_osm_map_features(
    bbox: BBox,
    cache_path: Optional[str | Path] = None,
    use_cache: bool = True,
    max_features: Optional[int] = None,
    endpoints: Optional[list[str]] = None,
    projection_bbox: Optional[BBox] = None,
) -> list[MapFeature]:
    """Load roads and water features for map visualization."""

    raw = None
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and use_cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))

    if raw is None:
        query = _map_features_query(bbox)
        raw = _fetch_overpass_json(
            query,
            endpoints
            or [
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass-api.de/api/interpreter",
            ],
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw), encoding="utf-8")

    return parse_osm_map_features(raw, projection_bbox or bbox, max_features=max_features)
