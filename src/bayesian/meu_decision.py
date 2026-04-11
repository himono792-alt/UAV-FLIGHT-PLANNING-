"""
MEU (Maximum Expected Utility) Decision Maker cho UAV — Ch7
Chọn action tối ưu dưới uncertainty:

    action* = argmax_a Σ P(result | action, evidence) * U(result)

Dùng BayesianNetwork để tính P(FlightSafety | evidence),
sau đó tính EU cho mỗi action và chọn action có EU cao nhất.
"""

from typing import Dict, Tuple, List
from src.bayesian.bayesian_network import BayesianNetwork, SAFETY_VALUES


# ── Utility Table ──────────────────────────────────────────────────────── #
# Phản ánh ưu tiên: an toàn > hiệu quả > tốc độ

UTILITY_TABLE: Dict[str, int] = {
    "reach_goal_safe":      +100,   # đến đích an toàn
    "reach_goal_slow":      +50,    # đến đích chậm (qua safe path)
    "return_home_safe":     +10,    # quay về an toàn
    "emergency_land":       -20,    # hạ cánh khẩn cấp
    "collision":            -1000,  # va chạm (tệ nhất)
    "battery_dead_midair":  -500,   # hết pin giữa chừng
}

# Các action mà UAV có thể chọn
ACTIONS = ["fly_direct", "fly_safe", "return_home", "land"]


# ── Outcome probabilities P(outcome | action, safety_level) ────────────── #
# Mỗi action × safety level → xác suất mỗi outcome

ACTION_OUTCOME_PROBS: Dict[str, Dict[str, Dict[str, float]]] = {
    "fly_direct": {
        "Safe":      {"reach_goal_safe": 0.90, "reach_goal_slow": 0.05,
                      "emergency_land":  0.03, "collision":       0.02,
                      "return_home_safe": 0.00, "battery_dead_midair": 0.00},
        "Risky":     {"reach_goal_safe": 0.45, "reach_goal_slow": 0.15,
                      "emergency_land":  0.15, "collision":       0.20,
                      "return_home_safe": 0.05, "battery_dead_midair": 0.00},
        "Dangerous": {"reach_goal_safe": 0.05, "reach_goal_slow": 0.05,
                      "emergency_land":  0.20, "collision":       0.60,
                      "return_home_safe": 0.05, "battery_dead_midair": 0.05},
    },
    "fly_safe": {
        "Safe":      {"reach_goal_safe": 0.70, "reach_goal_slow": 0.25,
                      "emergency_land":  0.02, "collision":       0.01,
                      "return_home_safe": 0.02, "battery_dead_midair": 0.00},
        "Risky":     {"reach_goal_safe": 0.50, "reach_goal_slow": 0.35,
                      "emergency_land":  0.08, "collision":       0.05,
                      "return_home_safe": 0.02, "battery_dead_midair": 0.00},
        "Dangerous": {"reach_goal_safe": 0.20, "reach_goal_slow": 0.30,
                      "emergency_land":  0.25, "collision":       0.15,
                      "return_home_safe": 0.05, "battery_dead_midair": 0.05},
    },
    "return_home": {
        "Safe":      {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.02, "collision":       0.00,
                      "return_home_safe": 0.95, "battery_dead_midair": 0.03},
        "Risky":     {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.05, "collision":       0.05,
                      "return_home_safe": 0.85, "battery_dead_midair": 0.05},
        "Dangerous": {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.15, "collision":       0.10,
                      "return_home_safe": 0.60, "battery_dead_midair": 0.15},
    },
    "land": {
        "Safe":      {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.90, "collision":       0.00,
                      "return_home_safe": 0.00, "battery_dead_midair": 0.10},
        "Risky":     {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.90, "collision":       0.02,
                      "return_home_safe": 0.00, "battery_dead_midair": 0.08},
        "Dangerous": {"reach_goal_safe": 0.00, "reach_goal_slow": 0.00,
                      "emergency_land":  0.85, "collision":       0.05,
                      "return_home_safe": 0.00, "battery_dead_midair": 0.10},
    },
}


# ── MEUDecisionMaker ───────────────────────────────────────────────────── #

class MEUDecisionMaker:
    """
    MEU theo Ch7:
        action* = argmax_a Σ P(result | action, evidence) * U(result)

    Workflow:
        1. evidence → BN.query("FlightSafety") → P(Safe/Risky/Dangerous)
        2. Với mỗi action: EU = Σ_safety Σ_outcome P(safety|evidence) * P(outcome|action,safety) * U(outcome)
        3. Chọn action có EU max
    """

    def __init__(self, bayesian_network: BayesianNetwork = None):
        self.bn = bayesian_network if bayesian_network is not None \
                  else BayesianNetwork()

    # ------------------------------------------------------------------ #
    #  Expected Utility                                                   #
    # ------------------------------------------------------------------ #

    def expected_utility(self, action: str,
                         safety_dist: Dict[str, float]) -> float:
        """
        EU(action | safety_dist) = Σ_{safety} P(safety) *
                                   Σ_{outcome} P(outcome|action,safety) * U(outcome)

        Args:
            action       : "fly_direct" | "fly_safe" | "return_home" | "land"
            safety_dist  : {"Safe": p1, "Risky": p2, "Dangerous": p3}
                           — output từ BN.query("FlightSafety", evidence)

        Returns: expected utility float
        """
        eu = 0.0
        outcome_table = ACTION_OUTCOME_PROBS.get(action, {})

        for safety_val in SAFETY_VALUES:
            p_safety = safety_dist.get(safety_val, 0.0)
            outcomes = outcome_table.get(safety_val, {})
            for outcome, p_out in outcomes.items():
                u = UTILITY_TABLE.get(outcome, 0)
                eu += p_safety * p_out * u

        return round(eu, 4)

    def decide(self, evidence: Dict[str, str]) -> Tuple[str, float]:
        """
        Chọn action tối ưu:
            action* = argmax_a EU(a | evidence)

        Args:
            evidence: {"Weather": "Rainy", "BatteryLevel": "Low", ...}

        Returns: (best_action, max_expected_utility)
        """
        safety_dist = self.bn.query("FlightSafety", evidence)

        best_action = None
        best_eu = float("-inf")

        for action in ACTIONS:
            eu = self.expected_utility(action, safety_dist)
            if eu > best_eu:
                best_eu = eu
                best_action = action

        return best_action, best_eu

    # ------------------------------------------------------------------ #
    #  Explain                                                           #
    # ------------------------------------------------------------------ #

    def explain_decision(self, evidence: Dict[str, str]) -> str:
        """
        Giải thích tại sao chọn action:
        - P(FlightSafety | evidence)
        - EU của từng action
        - Highlight action được chọn
        """
        safety_dist = self.bn.query("FlightSafety", evidence)
        best_action, best_eu = self.decide(evidence)

        lines = [
            "=" * 60,
            "MEU Decision Explanation",
            f"Evidence: {evidence}",
            "-" * 60,
            "P(FlightSafety | evidence):",
        ]
        for s, p in safety_dist.items():
            lines.append(f"  {s:<12} = {p:.4f}")

        lines += ["-" * 60, "Expected Utility per action:"]
        for action in ACTIONS:
            eu = self.expected_utility(action, safety_dist)
            marker = " ← CHOSEN" if action == best_action else ""
            lines.append(f"  {action:<15} EU = {eu:>8.2f}{marker}")

        lines += [
            "-" * 60,
            f"Decision: {best_action.upper()}  (EU = {best_eu:.2f})",
            "=" * 60,
        ]
        return "\n".join(lines)

    def all_utilities(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """Trả về dict EU của tất cả actions"""
        safety_dist = self.bn.query("FlightSafety", evidence)
        return {a: self.expected_utility(a, safety_dist) for a in ACTIONS}
