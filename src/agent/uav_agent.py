"""
UAV Agent — Tích hợp tất cả 7 module thành hệ thống thống nhất
Lý thuyết: Ch2 (Learning Agent Architecture)

Vòng lặp chính:
    PERCEIVE → REASON → PLAN → ACT → LEARN

Components (Ch2):
    Performance Element : A* + GA/SA  (GĐ2, GĐ3)
    Critic              : so sánh expected vs actual
    Learning Element    : cập nhật Decision Tree, Q-table  (GĐ6)
    Problem Generator   : thử đường mới khi gặp thất bại
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# GĐ1: Environment
from src.environment.grid_world import GridWorld3D, CellType
from src.environment.obstacles import ObstacleManager

# GĐ2: Search
from src.search.a_star import a_star_search
from src.search.heuristics import euclidean_3d, energy_based, HEURISTICS
from src.search.greedy_bfs import greedy_bfs

# GĐ3: Optimizer
from src.optimizer.simulated_annealing import SimulatedAnnealing

# GĐ4: Knowledge
from src.knowledge.kb_agent import KnowledgeBase, KBAgent
from src.knowledge.propositional import evaluate_rules, explain_action
from src.knowledge.fol_rules import FOLKnowledgeBase

# GĐ5: Bayesian
from src.bayesian.bayesian_network import BayesianNetwork
from src.bayesian.meu_decision import MEUDecisionMaker
from src.bayesian.inference import BayesianInference

# GĐ6: Learning
from src.learning.decision_tree import FlightConditionClassifier
from src.learning.regression import FlightTimePredictor
from src.learning.reinforcement import QLearningAgent


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass
class MissionResult:
    success:       bool
    reached_goal:  bool
    path:          List[tuple]
    battery_used:  float
    time_taken_ms: float
    steps:         int
    replans:       int
    events:        List[str]
    decision_log:  List[dict]

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{status}] reached={self.reached_goal} | "
            f"steps={self.steps} | battery_used={self.battery_used:.1f}% | "
            f"replans={self.replans} | time={self.time_taken_ms:.1f}ms"
        )


# ── UAVAgent ───────────────────────────────────────────────────────────── #

class UAVAgent:
    """
    Learning Agent theo Ch2 — tích hợp GĐ1~GĐ6.

    PERCEIVE  → đọc sensor (GPS, LIDAR, wind, battery)
    REASON    → KB rules + Bayesian inference + MEU + DT classifier
    PLAN      → A* / Greedy / A*+SA tùy quyết định
    ACT       → di chuyển theo path, tiêu pin, gặp dynamic obstacles
    LEARN     → Critic cập nhật Q-table, log kinh nghiệm
    """

    BATTERY_DRAIN_PER_STEP  = 0.8   # % pin mỗi bước bay ngang
    BATTERY_DRAIN_CLIMB     = 1.5   # % pin mỗi bước leo cao
    BATTERY_LOW_THRESHOLD   = 20.0  # % pin → trigger return home
    WIND_HIGH               = 15.0  # m/s
    WIND_DANGEROUS          = 25.0  # m/s

    def __init__(self, grid: GridWorld3D,
                 wind_speed: float = 5.0,
                 initial_battery: float = 100.0,
                 seed: int = 42):
        self.grid       = grid
        self.wind_speed = wind_speed
        self.battery    = initial_battery
        self.position   = grid.start
        self.goal       = grid.goal
        self.path: List[tuple] = []
        self.flight_log: List[dict] = []
        self._rng = random.Random(seed)

        # GĐ1
        self.obstacle_mgr = ObstacleManager()
        self.fol_kb = FOLKnowledgeBase(grid)
        self.fol_kb.build_from_grid()

        # GĐ4
        self.kb       = KnowledgeBase()
        self.kb_agent = KBAgent(self.kb)

        # GĐ5
        self.bn        = BayesianNetwork()
        self.meu       = MEUDecisionMaker(self.bn)
        self.inference = BayesianInference()

        # GĐ6
        self.classifier = FlightConditionClassifier(max_depth=5)
        self._train_classifier()

        self.predictor = FlightTimePredictor()
        self._train_predictor()

        self.rl_agent = QLearningAgent(grid, alpha=0.2, gamma=0.9, seed=seed)

    # ------------------------------------------------------------------ #
    #  Pre-training GĐ6                                                  #
    # ------------------------------------------------------------------ #

    def _train_classifier(self):
        X, y = self.classifier.generate_training_data(n_samples=300, seed=0)
        self.classifier.train(X, y)

    def _train_predictor(self):
        X, y = self.predictor.generate_training_data(n_samples=200, seed=0)
        self.predictor.train(X, y)

    # ------------------------------------------------------------------ #
    #  PERCEIVE (Ch2)                                                    #
    # ------------------------------------------------------------------ #

    def perceive(self) -> dict:
        """
        Thu thập sensor data, mô phỏng GPS + LIDAR + Wind + Battery.
        """
        # Kiểm tra ô xung quanh có obstacle không (LIDAR)
        neighbors = list(self.grid.get_neighbors(self.position))
        obstacle_nearby = any(
            self.grid.get_cell(nb) == CellType.OBSTACLE
            for nb in neighbors
        )
        in_nfz = (self.grid.get_cell(self.position) == CellType.NO_FLY_ZONE)

        # Ước lượng obstacle density trong vùng lân cận
        obs_count = sum(
            1 for nb in neighbors
            if self.grid.get_cell(nb) in (CellType.OBSTACLE, CellType.NO_FLY_ZONE)
        )
        obs_density = obs_count / max(1, len(neighbors))

        return {
            "position":          self.position,
            "battery":           self.battery,
            "wind_speed":        self.wind_speed,
            "obstacle_detected": obstacle_nearby,
            "in_no_fly_zone":    in_nfz,
            "on_path":           (self.position in self.path) if self.path else True,
            "near_charger":      False,
            "obs_density":       obs_density,
            "visibility":        min(100.0, 100.0 - self.wind_speed * 1.5),
        }

    # ------------------------------------------------------------------ #
    #  REASON (Ch2 + GĐ4 + GĐ5 + GĐ6)                                  #
    # ------------------------------------------------------------------ #

    def reason(self, percept: dict) -> dict:
        """
        1. KB rules (GĐ4)        → safety action
        2. Sensor fusion (GĐ5)   → P(obstacle)
        3. BN inference (GĐ5)    → P(FlightSafety)
        4. MEU (GĐ5)             → best action
        5. DT classifier (GĐ6)   → flight condition label
        """
        # GĐ4 — KB + forward chaining
        kb_action = self.kb_agent.agent_function(percept)
        explanation = explain_action(self.kb.facts)

        # GĐ5 — Sensor fusion
        sensor_data = {
            "lidar": percept["obstacle_detected"],
            "wind_sensor": self.wind_speed > self.WIND_HIGH,
        }
        fusion_result = self.inference.sensor_fusion(sensor_data,
                                                      prior_obstacle=percept["obs_density"])

        # GĐ5 — BN evidence
        weather = self._wind_to_weather(self.wind_speed)
        battery_level = self._battery_to_level(self.battery)
        bn_evidence = {"Weather": weather, "BatteryLevel": battery_level}
        meu_action, meu_eu = self.meu.decide(bn_evidence)

        # GĐ6 — Decision Tree
        dt_features = [
            self.wind_speed,
            self.battery,
            percept["visibility"],
            percept["obs_density"],
            float(self.position[2]),
        ]
        dt_label = self.classifier.predict(dt_features)

        # Tổng hợp quyết định cuối
        final_action = self._merge_decisions(kb_action, meu_action, dt_label)

        return {
            "kb_action":    kb_action,
            "meu_action":   meu_action,
            "meu_eu":       meu_eu,
            "dt_label":     dt_label,
            "final_action": final_action,
            "obstacle_prob": fusion_result["obstacle_prob"],
            "alert_level":  fusion_result["alert_level"],
            "explanation":  explanation,
        }

    def _merge_decisions(self, kb_action: str, meu_action: str,
                         dt_label: str) -> str:
        """
        Ưu tiên: KB safety rules > DT label > MEU action
        KB AVOID/EXIT_NFZ/LAND luôn được ưu tiên
        """
        override_actions = {"AVOID", "EXIT_NFZ", "LAND"}
        if kb_action in override_actions:
            return kb_action

        # DT LAND → thực sự nguy hiểm
        if dt_label == "LAND":
            return "land"
        if dt_label == "AVOID":
            return "fly_safe"
        if dt_label == "CAREFUL":
            return "fly_safe" if meu_action in ("fly_direct",) else meu_action

        return meu_action  # default: tin MEU

    # ------------------------------------------------------------------ #
    #  PLAN (GĐ2 + GĐ3)                                                 #
    # ------------------------------------------------------------------ #

    def plan(self, decision: dict) -> List[tuple]:
        """
        Chọn thuật toán tìm đường theo final_action:
        - fly_direct   → A* + euclidean heuristic
        - fly_safe     → A* + energy heuristic + SA optimize
        - return_home  → Greedy BFS (nhanh, pin yếu)
        - land/LAND    → tìm ô gần nhất
        - AVOID        → A* + energy heuristic (đường an toàn)
        """
        action = decision["final_action"]
        start  = self.position
        goal   = self.goal

        if action in ("land", "LAND"):
            return [start]  # đứng yên hạ cánh

        if action in ("return_home", "RETURN_HOME"):
            r = greedy_bfs(self.grid, start, self.grid.start)
            if r.found and len(r.path) > 0:
                return r.path
            return [start]

        if action in ("fly_safe", "AVOID", "EXIT_NFZ"):
            heuristic = HEURISTICS["energy"]
            r = a_star_search(self.grid, start, goal, heuristic)
            if r.found and len(r.path) > 2:
                sa = SimulatedAnnealing(self.grid, T0=50.0, alpha=0.995,
                                        max_iter=1000, seed=0)
                opt = sa.optimize(r.path)
                return opt.path
            return r.path if r.found else [start]

        # fly_direct → A* + euclidean
        r = a_star_search(self.grid, start, goal, euclidean_3d)
        if r.found:
            return r.path
        # Fallback: Greedy BFS
        r2 = greedy_bfs(self.grid, start, goal)
        return r2.path if r2.found else [start]

    # ------------------------------------------------------------------ #
    #  ACT (mô phỏng flight)                                             #
    # ------------------------------------------------------------------ #

    def act(self, path: List[tuple]) -> dict:
        """
        Di chuyển UAV theo path:
        - Tiêu pin theo khoảng cách + leo cao
        - Có thể gặp dynamic obstacle giữa chừng
        """
        if not path or len(path) < 2:
            return {"moved": 0, "battery_used": 0.0, "blocked": False}

        moved        = 0
        battery_used = 0.0
        blocked      = False

        for i in range(1, len(path)):
            wp = path[i]

            # Kiểm tra obstacle tại waypoint
            if self.grid.get_cell(wp) in (CellType.OBSTACLE, CellType.NO_FLY_ZONE):
                blocked = True
                break

            # Tính hao pin
            drain = self.BATTERY_DRAIN_PER_STEP
            if wp[2] > self.position[2]:
                drain = self.BATTERY_DRAIN_CLIMB
            self.battery = max(0.0, self.battery - drain)
            battery_used += drain

            self.position = wp
            moved += 1

            # Dừng nếu hết pin
            if self.battery <= 0:
                break

        return {
            "moved":        moved,
            "battery_used": battery_used,
            "blocked":      blocked,
            "new_position": self.position,
        }

    # ------------------------------------------------------------------ #
    #  LEARN (Critic — Ch2)                                              #
    # ------------------------------------------------------------------ #

    def learn(self, percept: dict, decision: dict, act_result: dict):
        """
        Critic: log kết quả để phân tích sau.
        Q-table update nếu bị block (negative feedback).
        """
        log_entry = {
            "position":    percept["position"],
            "battery":     percept["battery"],
            "action":      decision["final_action"],
            "dt_label":    decision["dt_label"],
            "obstacle_prob": decision["obstacle_prob"],
            "moved":       act_result["moved"],
            "blocked":     act_result["blocked"],
        }
        self.flight_log.append(log_entry)

        # Nếu bị block → Q-table penalty tại vị trí đó
        if act_result["blocked"] and percept["position"] in self.rl_agent.q_table:
            pass  # Q-table sẽ được cập nhật trong run_mission nếu cần

    # ------------------------------------------------------------------ #
    #  run_mission — Vòng lặp chính                                      #
    # ------------------------------------------------------------------ #

    def run_mission(self, start: tuple, goal: tuple,
                    max_steps: int = 200) -> MissionResult:
        """
        PERCEIVE → REASON → PLAN → ACT → LEARN
        Lặp đến khi đến đích, hết pin, hoặc vượt max_steps.
        """
        t0 = time.perf_counter()
        self.position   = start
        self.path       = []
        self.flight_log = []

        steps        = 0
        replans      = 0
        total_battery = 0.0
        events: List[str] = []
        decision_log: List[dict] = []

        while self.position != goal and steps < max_steps and self.battery > 0:
            # PERCEIVE
            percept = self.perceive()

            # REASON
            decision = self.reason(percept)
            decision_log.append({
                "step": steps,
                "pos":  self.position,
                "action": decision["final_action"],
            })

            # Dừng nếu quyết định hạ cánh
            if decision["final_action"] in ("land", "LAND"):
                events.append(f"step {steps}: Emergency land at {self.position}")
                break

            # Dừng nếu pin thấp và quyết định quay về
            if self.battery < self.BATTERY_LOW_THRESHOLD and start != goal:
                if decision["final_action"] in ("return_home", "RETURN_HOME"):
                    events.append(f"step {steps}: Low battery, returning home")
                    break

            # PLAN
            old_path = list(self.path)
            self.path = self.plan(decision)
            if self.path != old_path:
                replans += 1

            # ACT
            act_result = self.act(self.path)
            total_battery += act_result["battery_used"]

            if act_result["blocked"]:
                events.append(f"step {steps}: Blocked at {self.position}, replanning")

            # LEARN
            self.learn(percept, decision, act_result)

            # Nếu không di chuyển được (stuck)
            if act_result["moved"] == 0:
                events.append(f"step {steps}: Stuck at {self.position}")
                break

            steps += 1

        reached = (self.position == goal)
        success = reached and self.battery > 0

        return MissionResult(
            success       = success,
            reached_goal  = reached,
            path          = [log["pos"] for log in decision_log],
            battery_used  = total_battery,
            time_taken_ms = (time.perf_counter() - t0) * 1000,
            steps         = steps,
            replans       = replans,
            events        = events,
            decision_log  = decision_log,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wind_to_weather(wind: float) -> str:
        if wind <= 5:   return "Clear"
        if wind <= 12:  return "Cloudy"
        if wind <= 20:  return "Rainy"
        return "Stormy"

    @staticmethod
    def _battery_to_level(battery: float) -> str:
        if battery >= 70: return "Full"
        if battery >= 40: return "Medium"
        if battery >= 15: return "Low"
        return "Critical"
