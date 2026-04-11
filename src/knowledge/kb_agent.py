"""
Knowledge-Based Agent — Framework TELL/ASK theo Ch5
Áp dụng cho UAV: ra quyết định bay an toàn dựa trên tri thức + luật logic
"""

import time
from typing import Any, Optional
from src.knowledge.propositional import evaluate_rules, SAFETY_RULES


class KnowledgeBase:
    """
    Knowledge Base theo Ch5: TELL/ASK/RETRACT

    Lưu tri thức dưới dạng:
    - facts : dict[str, Any]  — trạng thái quan sát hiện tại
    - rules : list[dict]      — từ SAFETY_RULES (propositional.py)

    TELL(key, value)    → cập nhật fact
    ASK(query)          → tra cứu fact / áp dụng rules
    RETRACT(key)        → xóa fact
    """

    def __init__(self):
        self.facts: dict = {}
        self.rules: list = list(SAFETY_RULES)
        self.history: list = []   # log (fact, timestep)

    # ------------------------------------------------------------------ #
    #  TELL / ASK / RETRACT                                               #
    # ------------------------------------------------------------------ #

    def tell(self, key: str, value: Any, timestep: int = 0):
        """TELL(KB, sentence) — cập nhật tri thức mới"""
        self.facts[key] = value
        self.history.append((timestep, key, value))

    def ask(self, query: str) -> Any:
        """
        ASK(KB, query):
        - Nếu query là key của fact → trả về value
        - query == "action" → forward chaining qua rules
        - Không tìm thấy → None
        """
        if query == "action":
            return evaluate_rules(self.facts)
        return self.facts.get(query, None)

    def retract(self, key: str):
        """RETRACT — rút lui tri thức (khi thông tin thay đổi)"""
        self.facts.pop(key, None)

    def tell_percept(self, percept: dict, timestep: int = 0):
        """
        Batch TELL toàn bộ percept dict.
        Tự động derive: battery_low, wind_high, wind_dangerous
        """
        for k, v in percept.items():
            self.tell(k, v, timestep)

        # Derive higher-level facts từ raw values
        if "battery" in percept:
            bat = percept["battery"]
            self.tell("battery_low",      bat < 20, timestep)
            self.tell("battery_critical", bat < 10, timestep)

        if "wind_speed" in percept:
            ws = percept["wind_speed"]
            self.tell("wind_high",      ws > 15, timestep)
            self.tell("wind_dangerous", ws > 25, timestep)

    def summary(self) -> str:
        lines = [f"KB ({len(self.facts)} facts, {len(self.rules)} rules):"]
        for k, v in sorted(self.facts.items()):
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────── #

class KBAgent:
    """
    KB-Agent theo pseudocode Ch5:

        function KB-AGENT(percept) returns action
            TELL(KB, MAKE-PERCEPT-SENTENCE(percept, t))
            action ← ASK(KB, MAKE-ACTION-QUERY(t))
            TELL(KB, MAKE-ACTION-SENTENCE(action, t))
            t ← t + 1
            return action

    Mapping sang UAV domain (tương tự Wumpus World):
        Wumpus       ↔  Dynamic Obstacle
        Pit          ↔  No-Fly Zone
        Gold         ↔  Goal
        Stench       ↔  LIDAR phát hiện obstacle gần
        Breeze       ↔  Wind Sensor
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb if kb is not None else KnowledgeBase()
        self.t = 0          # timestep
        self.action_log: list = []

    def agent_function(self, percept: dict) -> str:
        """
        percept = {
            "position"          : (x, y, z),
            "battery"           : float  0-100,
            "wind_speed"        : float  m/s,
            "obstacle_detected" : bool,
            "in_no_fly_zone"    : bool,
            "on_path"           : bool,
            "near_charger"      : bool,
        }

        returns: "CONTINUE" | "AVOID" | "REDUCE_SPEED" |
                 "RETURN_HOME" | "LAND" | "EXIT_NFZ"
        """
        # Step 1: TELL KB percept + derived facts
        self.kb.tell_percept(percept, self.t)

        # Step 2: ASK KB → action via forward chaining
        action = self.kb.ask("action")

        # Step 3: TELL action vào KB (để log + future inference)
        self.kb.tell("last_action", action, self.t)

        # Step 4: log & advance timestep
        self.action_log.append((self.t, percept.get("position"), action))
        self.t += 1

        return action

    def reset(self):
        """Khởi tạo lại agent (giữ rules, xóa facts)"""
        self.kb.facts.clear()
        self.kb.history.clear()
        self.action_log.clear()
        self.t = 0

    def get_action_history(self) -> list:
        """Trả về log (timestep, position, action)"""
        return list(self.action_log)
