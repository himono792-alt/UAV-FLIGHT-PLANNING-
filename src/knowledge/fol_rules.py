"""
First-Order Logic (FOL) Knowledge Base cho UAV domain — Ch6

Predicates:
    Obstacle(x,y,z)       — có obstacle tại (x,y,z)
    NoFlyZone(x,y,z)      — vùng cấm bay tại (x,y,z)
    Safe(x,y,z)           — ô an toàn bay qua
    At(uav, x,y,z)        — UAV đang ở vị trí (x,y,z)
    Battery(uav, level)   — mức pin của UAV
    CanFly(uav, x,y,z)    — UAV có thể bay đến (x,y,z)
    Charger(x,y,z)        — trạm sạc tại (x,y,z)

FOL Rules:
    R1: ∀x,y,z: Obstacle(x,y,z) → ¬Safe(x,y,z)
    R2: ∀x,y,z: NoFlyZone(x,y,z) → ¬Safe(x,y,z)
    R3: ∀x,y,z: ¬Obstacle(x,y,z) ∧ ¬NoFlyZone(x,y,z) → Safe(x,y,z)
    R4: ∀u,x,y,z: At(u,x,y,z) ∧ Safe(x,y,z) → CanFly(u,x,y,z)
    R5: ∀u: Battery(u) < threshold → ∃s: Charger(s) ∧ FlyTo(u,s)
"""

from typing import Optional, List, Tuple
from src.environment.grid_world import GridWorld3D, CellType


class FOLKnowledgeBase:
    """
    FOL Knowledge Base cho UAV domain.
    Biểu diễn predicates dưới dạng dict[name → set[tuple]].

    Cách dùng:
        fol_kb = FOLKnowledgeBase(grid)
        fol_kb.build_from_grid()          # tự động populate từ GridWorld3D
        safe = fol_kb.query_safe(3, 4, 2)
        neighbors = fol_kb.get_safe_neighbors((3, 4, 2))
        charger = fol_kb.find_nearest_charger((0, 0, 0))
    """

    def __init__(self, grid: GridWorld3D):
        self.grid = grid
        self.predicates: dict = {
            "Obstacle":   set(),   # {(x,y,z), ...}
            "NoFlyZone":  set(),
            "Safe":       set(),
            "Charger":    set(),
            "At":         {},      # {uav_id: (x,y,z)}
            "Battery":    {},      # {uav_id: float}
        }

    # ------------------------------------------------------------------ #
    #  Build predicates từ GridWorld3D                                    #
    # ------------------------------------------------------------------ #

    def build_from_grid(self):
        """
        Áp dụng R1, R2, R3 để tính Safe:
            R1: Obstacle → ¬Safe
            R2: NoFlyZone → ¬Safe
            R3: ¬Obstacle ∧ ¬NoFlyZone → Safe
        """
        self.predicates["Obstacle"].clear()
        self.predicates["NoFlyZone"].clear()
        self.predicates["Safe"].clear()

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                for z in range(self.grid.depth):
                    pos = (x, y, z)
                    cell = self.grid.get_cell(pos)

                    if cell == CellType.OBSTACLE:
                        self.predicates["Obstacle"].add(pos)
                    elif cell == CellType.NO_FLY_ZONE:
                        self.predicates["NoFlyZone"].add(pos)
                    else:
                        # R3: nếu không phải obstacle/NFZ → Safe
                        self.predicates["Safe"].add(pos)

    # ------------------------------------------------------------------ #
    #  TELL predicates thủ công                                          #
    # ------------------------------------------------------------------ #

    def add_predicate(self, name: str, *args):
        """Thêm 1 predicate instance vào KB"""
        if name in ("At", "Battery"):
            # Dạng {uav_id: value}
            if len(args) == 2:
                self.predicates[name][args[0]] = args[1]
        elif name in self.predicates:
            if isinstance(self.predicates[name], set):
                pos = tuple(args)
                self.predicates[name].add(pos)
                # Cập nhật Safe theo rules
                if name in ("Obstacle", "NoFlyZone"):
                    self.predicates["Safe"].discard(pos)
        else:
            self.predicates[name] = {tuple(args)}

    def remove_predicate(self, name: str, *args):
        """RETRACT predicate instance"""
        if name in self.predicates and isinstance(self.predicates[name], set):
            pos = tuple(args)
            self.predicates[name].discard(pos)
            # Nếu không còn obstacle/NFZ → có thể Safe lại
            if name in ("Obstacle", "NoFlyZone"):
                if pos not in self.predicates["Obstacle"] and \
                   pos not in self.predicates["NoFlyZone"]:
                    self.predicates["Safe"].add(pos)

    def add_charger(self, x: int, y: int, z: int):
        """Đăng ký trạm sạc tại (x,y,z)"""
        self.predicates["Charger"].add((x, y, z))

    # ------------------------------------------------------------------ #
    #  Query predicates (ASK)                                            #
    # ------------------------------------------------------------------ #

    def query_safe(self, x: int, y: int, z: int) -> bool:
        """
        ASK Safe(x,y,z) — ô này có an toàn không?
        R1+R2: Obstacle hoặc NoFlyZone → False
        R3: ngược lại → True
        """
        pos = (x, y, z)
        return pos in self.predicates["Safe"]

    def query_can_fly(self, uav_pos: tuple, target_pos: tuple) -> bool:
        """
        R4: CanFly(u, x,y,z) ← At(u, x,y,z) ∧ Safe(x,y,z)
        Simplified: target phải Safe và kề với uav_pos
        """
        tx, ty, tz = target_pos
        if not self.query_safe(tx, ty, tz):
            return False
        # Kề nhau (6-direction)
        dist = sum(abs(target_pos[k] - uav_pos[k]) for k in range(3))
        return dist == 1

    def get_safe_neighbors(self, pos: tuple) -> List[tuple]:
        """
        Dùng thay thế grid.get_neighbors() trong A* —
        chỉ trả về neighbors đã được KB xác nhận Safe.
        """
        neighbors = []
        for nb in self.grid.get_neighbors(pos):
            if self.query_safe(*nb):
                neighbors.append(nb)
        return neighbors

    def find_nearest_charger(self, pos: tuple) -> Optional[tuple]:
        """
        R5 support: tìm trạm sạc gần nhất theo Manhattan distance.
        Returns None nếu không có charger nào được đăng ký.
        """
        chargers = self.predicates.get("Charger", set())
        if not chargers:
            return None
        return min(chargers,
                   key=lambda c: sum(abs(c[k] - pos[k]) for k in range(3)))

    def query_obstacle(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) in self.predicates["Obstacle"]

    def query_no_fly_zone(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) in self.predicates["NoFlyZone"]

    def get_all_safe(self) -> set:
        """Trả về tập tất cả ô Safe"""
        return set(self.predicates["Safe"])

    # ------------------------------------------------------------------ #
    #  Summary                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        obs = len(self.predicates["Obstacle"])
        nfz = len(self.predicates["NoFlyZone"])
        safe = len(self.predicates["Safe"])
        chargers = len(self.predicates.get("Charger", set()))
        total = self.grid.width * self.grid.height * self.grid.depth
        return (
            f"FOL KB | Grid {self.grid.width}×{self.grid.height}×{self.grid.depth} "
            f"({total} cells)\n"
            f"  Obstacle : {obs}\n"
            f"  NoFlyZone: {nfz}\n"
            f"  Safe     : {safe}\n"
            f"  Charger  : {chargers}"
        )
