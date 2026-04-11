"""
Bayesian Network cho UAV Decision Making — Tự cài đặt, không dùng pgmpy
Lý thuyết: Chương 8 (BN, DAG, CPT, Global Semantics, Markov Blanket)

Cấu trúc DAG:
    Weather ──→ WindSpeed ──→ FlightSafety ──→ PathDecision
                                  ↑
    BatteryLevel ──→ FlightRange ─┘

Global Semantics (Ch8):
    P(x1,...,xn) = Π P(xi | parents(xi))
"""

from typing import Dict, List, Optional, Tuple
import math


# ── Node values (discrete) ─────────────────────────────────────────────── #

WEATHER_VALUES      = ["Clear", "Cloudy", "Rainy", "Stormy"]
WIND_VALUES         = ["Low", "Medium", "High", "Dangerous"]
BATTERY_VALUES      = ["Full", "Medium", "Low", "Critical"]
RANGE_VALUES        = ["Long", "Medium", "Short"]
SAFETY_VALUES       = ["Safe", "Risky", "Dangerous"]
DECISION_VALUES     = ["DirectPath", "SafePath", "ReturnHome", "Land"]


# ── BayesianNode ───────────────────────────────────────────────────────── #

class BayesianNode:
    """
    1 node trong Bayesian Network.

    CPT format:
        - Root node: {"value": prob, ...}
        - Child node: {(parent_val1, parent_val2, ...): {"val": prob, ...}}
    """

    def __init__(self, name: str, values: List[str],
                 parents: List[str], cpt: dict):
        self.name    = name
        self.values  = values
        self.parents = parents   # list of parent node names
        self.cpt     = cpt

    def probability(self, value: str,
                    parent_values: Optional[Tuple] = None) -> float:
        """
        Tra CPT: P(self=value | parents=parent_values)

        Args:
            value        : giá trị cần tra
            parent_values: tuple giá trị của từng parent theo thứ tự self.parents
        """
        if not self.parents:
            # Root node — CPT dạng {value: prob}
            return self.cpt.get(value, 0.0)

        key = parent_values if len(self.parents) > 1 else parent_values[0]
        row = self.cpt.get(key, {})
        return row.get(value, 0.0)


# ── BayesianNetwork ────────────────────────────────────────────────────── #

class BayesianNetwork:
    """
    Bayesian Network tự cài đặt — Ch8.

    Inference bằng Enumeration (exact inference cho BN nhỏ):
        P(X|e) = α * Σ_{y} P(X, e, y)

    Markov Blanket (Ch8):
        MB(X) = parents(X) ∪ children(X) ∪ parents_of_children(X)
        X ⊥ tất cả node khác | MB(X)
    """

    def __init__(self):
        self.nodes: Dict[str, BayesianNode] = {}
        self._topo_order: List[str] = []
        self._build_network()

    # ------------------------------------------------------------------ #
    #  Build DAG + CPT                                                    #
    # ------------------------------------------------------------------ #

    def _build_network(self):
        # ── P(Weather) — root ──
        self.nodes["Weather"] = BayesianNode(
            name="Weather", values=WEATHER_VALUES, parents=[],
            cpt={"Clear": 0.40, "Cloudy": 0.30, "Rainy": 0.20, "Stormy": 0.10}
        )

        # ── P(BatteryLevel) — root ──
        self.nodes["BatteryLevel"] = BayesianNode(
            name="BatteryLevel", values=BATTERY_VALUES, parents=[],
            cpt={"Full": 0.35, "Medium": 0.40, "Low": 0.20, "Critical": 0.05}
        )

        # ── P(WindSpeed | Weather) ──
        self.nodes["WindSpeed"] = BayesianNode(
            name="WindSpeed", values=WIND_VALUES, parents=["Weather"],
            cpt={
                "Clear":  {"Low": 0.60, "Medium": 0.30, "High": 0.08, "Dangerous": 0.02},
                "Cloudy": {"Low": 0.30, "Medium": 0.40, "High": 0.20, "Dangerous": 0.10},
                "Rainy":  {"Low": 0.10, "Medium": 0.30, "High": 0.40, "Dangerous": 0.20},
                "Stormy": {"Low": 0.05, "Medium": 0.15, "High": 0.30, "Dangerous": 0.50},
            }
        )

        # ── P(FlightRange | BatteryLevel) ──
        self.nodes["FlightRange"] = BayesianNode(
            name="FlightRange", values=RANGE_VALUES, parents=["BatteryLevel"],
            cpt={
                "Full":     {"Long": 0.80, "Medium": 0.18, "Short": 0.02},
                "Medium":   {"Long": 0.30, "Medium": 0.60, "Short": 0.10},
                "Low":      {"Long": 0.05, "Medium": 0.35, "Short": 0.60},
                "Critical": {"Long": 0.01, "Medium": 0.09, "Short": 0.90},
            }
        )

        # ── P(FlightSafety | WindSpeed, FlightRange) ──
        # Key: (WindSpeed_value, FlightRange_value)
        fs_cpt = {}
        safety_matrix = {
            # (wind, range) → {Safe, Risky, Dangerous}
            ("Low",       "Long"):   {"Safe": 0.90, "Risky": 0.08, "Dangerous": 0.02},
            ("Low",       "Medium"): {"Safe": 0.80, "Risky": 0.15, "Dangerous": 0.05},
            ("Low",       "Short"):  {"Safe": 0.60, "Risky": 0.30, "Dangerous": 0.10},
            ("Medium",    "Long"):   {"Safe": 0.70, "Risky": 0.25, "Dangerous": 0.05},
            ("Medium",    "Medium"): {"Safe": 0.50, "Risky": 0.38, "Dangerous": 0.12},
            ("Medium",    "Short"):  {"Safe": 0.30, "Risky": 0.45, "Dangerous": 0.25},
            ("High",      "Long"):   {"Safe": 0.30, "Risky": 0.50, "Dangerous": 0.20},
            ("High",      "Medium"): {"Safe": 0.15, "Risky": 0.45, "Dangerous": 0.40},
            ("High",      "Short"):  {"Safe": 0.05, "Risky": 0.25, "Dangerous": 0.70},
            ("Dangerous", "Long"):   {"Safe": 0.05, "Risky": 0.25, "Dangerous": 0.70},
            ("Dangerous", "Medium"): {"Safe": 0.02, "Risky": 0.13, "Dangerous": 0.85},
            ("Dangerous", "Short"):  {"Safe": 0.01, "Risky": 0.04, "Dangerous": 0.95},
        }
        self.nodes["FlightSafety"] = BayesianNode(
            name="FlightSafety", values=SAFETY_VALUES,
            parents=["WindSpeed", "FlightRange"],
            cpt=safety_matrix
        )

        # ── P(PathDecision | FlightSafety) ──
        self.nodes["PathDecision"] = BayesianNode(
            name="PathDecision", values=DECISION_VALUES, parents=["FlightSafety"],
            cpt={
                "Safe":      {"DirectPath": 0.70, "SafePath": 0.20, "ReturnHome": 0.05, "Land": 0.05},
                "Risky":     {"DirectPath": 0.10, "SafePath": 0.50, "ReturnHome": 0.30, "Land": 0.10},
                "Dangerous": {"DirectPath": 0.00, "SafePath": 0.10, "ReturnHome": 0.30, "Land": 0.60},
            }
        )

        # Topological order (ancestors trước)
        self._topo_order = [
            "Weather", "BatteryLevel",
            "WindSpeed", "FlightRange",
            "FlightSafety", "PathDecision"
        ]

    # ------------------------------------------------------------------ #
    #  Inference — Enumeration                                           #
    # ------------------------------------------------------------------ #

    def query(self, target: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Exact inference bằng Enumeration:
            P(target | evidence) = α * Σ_{hidden} P(target, evidence, hidden)

        Args:
            target  : tên node cần tính, vd "FlightSafety"
            evidence: {"Weather": "Rainy", "BatteryLevel": "Low"}

        Returns: {"Safe": 0.x, "Risky": 0.y, "Dangerous": 0.z}
        """
        target_node = self.nodes[target]
        result = {}

        for val in target_node.values:
            # P(target=val, evidence) bằng cách enumerate tất cả hidden vars
            full_evidence = dict(evidence)
            full_evidence[target] = val
            result[val] = self._enumerate_all(self._topo_order, full_evidence)

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total, 6) for k, v in result.items()}

        return result

    def _enumerate_all(self, variables: List[str],
                       evidence: Dict[str, str]) -> float:
        """
        Đệ quy tính xác suất joint distribution.
        Base case: không còn biến ẩn → trả về 1.0
        """
        if not variables:
            return 1.0

        var = variables[0]
        rest = variables[1:]
        node = self.nodes[var]

        if var in evidence:
            # Biến đã có giá trị (observed)
            prob = self._node_probability(node, evidence[var], evidence)
            return prob * self._enumerate_all(rest, evidence)
        else:
            # Biến ẩn — sum out
            total = 0.0
            for val in node.values:
                ev_copy = dict(evidence)
                ev_copy[var] = val
                prob = self._node_probability(node, val, ev_copy)
                total += prob * self._enumerate_all(rest, ev_copy)
            return total

    def _node_probability(self, node: BayesianNode,
                          value: str, evidence: Dict[str, str]) -> float:
        """Tra CPT của node với parent values từ evidence"""
        if not node.parents:
            return node.probability(value)

        parent_vals = tuple(evidence.get(p, node.values[0]) for p in node.parents)
        return node.probability(value, parent_vals)

    # ------------------------------------------------------------------ #
    #  Markov Blanket                                                     #
    # ------------------------------------------------------------------ #

    def markov_blanket(self, node_name: str) -> set:
        """
        Markov Blanket — Ch8:
        MB(X) = parents(X) ∪ children(X) ∪ parents_of_children(X)

        X ⊥ tất cả node còn lại | MB(X)
        """
        node = self.nodes[node_name]
        mb = set()

        # Parents
        for p in node.parents:
            mb.add(p)

        # Children + parents of children
        for name, n in self.nodes.items():
            if node_name in n.parents:          # n là con của X
                mb.add(name)
                for p in n.parents:             # parents của con
                    if p != node_name:
                        mb.add(p)

        return mb

    # ------------------------------------------------------------------ #
    #  Marginal probability P(node) — sum over all parents               #
    # ------------------------------------------------------------------ #

    def marginal(self, node_name: str) -> Dict[str, float]:
        """
        Tính P(node) marginalized qua tất cả parents.
        Dùng Enumeration với evidence rỗng.
        """
        return self.query(node_name, {})

    # ------------------------------------------------------------------ #
    #  Summary                                                           #
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        lines = ["Bayesian Network — UAV Domain", "=" * 40]
        for name in self._topo_order:
            node = self.nodes[name]
            pstr = f" | {', '.join(node.parents)}" if node.parents else " (root)"
            lines.append(f"  {name}{pstr}")
            lines.append(f"    values: {node.values}")
        return "\n".join(lines)
