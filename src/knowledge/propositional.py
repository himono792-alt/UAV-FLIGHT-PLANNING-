"""
Propositional Logic — 6 luật an toàn bay theo Ch5
Forward chaining: duyệt rules theo priority, trả về action đầu tiên match.

Thresholds:
  battery_low      : battery < 20%
  battery_critical : battery < 10%
  wind_high        : wind_speed > 15 m/s
  wind_dangerous   : wind_speed > 25 m/s
"""

from typing import Optional

# ── Battery & Wind thresholds ──────────────────────────────────────────── #
BATTERY_LOW_THRESHOLD        = 20.0   # %
BATTERY_CRITICAL_THRESHOLD   = 10.0   # %
WIND_HIGH_THRESHOLD          = 15.0   # m/s
WIND_DANGEROUS_THRESHOLD     = 25.0   # m/s

# ── Action constants ───────────────────────────────────────────────────── #
ACTION_AVOID         = "AVOID"
ACTION_EXIT_NFZ      = "EXIT_NFZ"
ACTION_LAND          = "LAND"
ACTION_RETURN_HOME   = "RETURN_HOME"
ACTION_REDUCE_SPEED  = "REDUCE_SPEED"
ACTION_CONTINUE      = "CONTINUE"


# ── 6 Safety Rules — Propositional Logic ───────────────────────────────── #
# Sắp xếp theo priority tăng dần (1 = cao nhất)

SAFETY_RULES = [
    {
        "id": "R1",
        "name": "Avoid Obstacle",
        "description": "ObstacleDetected → Avoid",
        "condition": lambda f: f.get("obstacle_detected") is True,
        "action": ACTION_AVOID,
        "priority": 1,
    },
    {
        "id": "R5",
        "name": "Exit No-Fly Zone",
        "description": "InNoFlyZone → ExitImmediately",
        "condition": lambda f: f.get("in_no_fly_zone") is True,
        "action": ACTION_EXIT_NFZ,
        "priority": 2,
    },
    {
        "id": "R4",
        "name": "Land — Wind Dangerous",
        "description": "WindDangerous → Land",
        "condition": lambda f: f.get("wind_dangerous") is True,
        "action": ACTION_LAND,
        "priority": 3,
    },
    {
        "id": "R2",
        "name": "Return Home — Battery Low",
        "description": "BatteryLow ∧ ¬NearCharger → ReturnHome",
        "condition": lambda f: (
            f.get("battery_low") is True and
            f.get("near_charger") is not True
        ),
        "action": ACTION_RETURN_HOME,
        "priority": 4,
    },
    {
        "id": "R3",
        "name": "Reduce Speed — Wind High",
        "description": "WindHigh ∧ ¬WindDangerous → ReduceSpeed",
        "condition": lambda f: (
            f.get("wind_high") is True and
            f.get("wind_dangerous") is not True
        ),
        "action": ACTION_REDUCE_SPEED,
        "priority": 5,
    },
    {
        "id": "R6",
        "name": "Continue — All Clear",
        "description": "¬ObstacleDetected ∧ ¬NoFlyZone ∧ OnPath → Continue",
        "condition": lambda f: (
            f.get("obstacle_detected") is not True and
            f.get("in_no_fly_zone") is not True and
            f.get("on_path") is True
        ),
        "action": ACTION_CONTINUE,
        "priority": 6,
    },
]

# Sắp xếp sẵn theo priority (chỉ sort 1 lần lúc import)
SAFETY_RULES = sorted(SAFETY_RULES, key=lambda r: r["priority"])


# ── Forward Chaining ───────────────────────────────────────────────────── #

def evaluate_rules(facts: dict) -> str:
    """
    Forward chaining theo Ch5:
    Duyệt rules theo priority, trả về action của rule đầu tiên thỏa.
    Default → "CONTINUE" nếu không rule nào match.
    """
    for rule in SAFETY_RULES:
        try:
            if rule["condition"](facts):
                return rule["action"]
        except Exception:
            continue
    return ACTION_CONTINUE


def explain_action(facts: dict) -> dict:
    """
    Trả về dict giải thích tại sao action được chọn.
    Dùng để debug / báo cáo.
    """
    for rule in SAFETY_RULES:
        try:
            if rule["condition"](facts):
                return {
                    "action": rule["action"],
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "description": rule["description"],
                }
        except Exception:
            continue
    return {
        "action": ACTION_CONTINUE,
        "rule_id": "DEFAULT",
        "rule_name": "Default",
        "description": "Không rule nào match → Continue",
    }


def get_rule_summary() -> str:
    """In bảng tóm tắt tất cả rules"""
    lines = ["=" * 60,
             f"{'ID':<5} {'Priority':<10} {'Name':<25} {'Action':<15}",
             "-" * 60]
    for r in SAFETY_RULES:
        lines.append(f"{r['id']:<5} {r['priority']:<10} {r['name']:<25} {r['action']:<15}")
    lines.append("=" * 60)
    return "\n".join(lines)
