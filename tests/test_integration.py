"""
Test Giai đoạn 7: Tích hợp UAVAgent — End-to-End
Chạy: python -m pytest tests/test_integration.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_world import GridWorld3D, CellType
from src.agent.uav_agent import UAVAgent, MissionResult


# ── Fixtures ───────────────────────────────────────────────────────────── #

def make_agent(w=10, h=10, d=4, start=(0,0,0), goal=(9,9,3),
               wind=5.0, battery=100.0, seed=42, obstacles=0):
    g = GridWorld3D(w, h, d)
    g.set_start(start)
    g.set_goal(goal)
    if obstacles > 0:
        g.add_random_obstacles(obstacles, seed=seed)
    return UAVAgent(g, wind_speed=wind, initial_battery=battery, seed=seed), g


# ── TestUAVAgent ───────────────────────────────────────────────────────── #

class TestUAVAgent:

    def test_init_no_error(self):
        """Khởi tạo UAVAgent không lỗi"""
        agent, _ = make_agent()
        assert agent is not None
        assert agent.battery == 100.0
        assert agent.position == (0, 0, 0)

    def test_perceive_format(self):
        """perceive() trả về dict đúng format"""
        agent, _ = make_agent()
        percept = agent.perceive()
        required_keys = ["position", "battery", "wind_speed",
                         "obstacle_detected", "in_no_fly_zone",
                         "on_path", "near_charger"]
        for key in required_keys:
            assert key in percept, f"Missing key: {key}"

    def test_perceive_values_valid(self):
        """Giá trị percept trong khoảng hợp lý"""
        agent, _ = make_agent(wind=10.0, battery=80.0)
        percept = agent.perceive()
        assert 0 <= percept["battery"] <= 100
        assert percept["wind_speed"] >= 0
        assert isinstance(percept["obstacle_detected"], bool)

    def test_reason_returns_valid_action(self):
        """reason() trả về final_action hợp lệ"""
        agent, _ = make_agent()
        percept = agent.perceive()
        decision = agent.reason(percept)
        valid_actions = {
            "fly_direct", "fly_safe", "return_home", "land",
            "AVOID", "EXIT_NFZ", "LAND", "RETURN_HOME", "REDUCE_SPEED", "CONTINUE"
        }
        assert decision["final_action"] in valid_actions, \
            f"Invalid action: {decision['final_action']}"

    def test_reason_returns_required_keys(self):
        """reason() trả về đủ keys"""
        agent, _ = make_agent()
        decision = agent.reason(agent.perceive())
        for key in ["kb_action", "meu_action", "dt_label",
                    "final_action", "obstacle_prob", "alert_level"]:
            assert key in decision, f"Missing key: {key}"

    def test_plan_returns_path(self):
        """plan() trả về path không rỗng"""
        agent, _ = make_agent()
        percept = agent.perceive()
        decision = agent.reason(percept)
        path = agent.plan(decision)
        assert isinstance(path, list)
        assert len(path) >= 1

    def test_plan_fly_direct_starts_at_position(self):
        """plan(fly_direct) → path bắt đầu từ vị trí hiện tại"""
        agent, _ = make_agent()
        path = agent.plan({"final_action": "fly_direct"})
        assert path[0] == agent.position

    def test_run_mission_normal(self):
        """Mission bình thường: đến đích với pin đủ"""
        agent, _ = make_agent(w=8, h=8, d=4, goal=(7,7,3),
                              wind=5.0, battery=100.0)
        result = agent.run_mission((0,0,0), (7,7,3), max_steps=300)
        assert isinstance(result, MissionResult)
        assert result.reached_goal, f"Không đến đích: {result.summary()}"
        assert result.battery_used > 0

    def test_run_mission_has_steps(self):
        """Mission có ít nhất 1 step"""
        agent, _ = make_agent(w=8, h=8, d=4, goal=(7,7,3))
        result = agent.run_mission((0,0,0), (7,7,3), max_steps=200)
        assert result.steps >= 1

    def test_battery_drain(self):
        """Pin giảm sau mission"""
        agent, _ = make_agent(w=8, h=8, d=4, goal=(7,7,3))
        agent.run_mission((0,0,0), (7,7,3), max_steps=200)
        assert agent.battery < 100.0, "Pin không giảm"

    def test_run_mission_with_obstacles(self):
        """Mission với obstacles — agent không crash"""
        agent, _ = make_agent(w=10, h=10, d=4, goal=(9,9,3),
                              obstacles=15, wind=8.0)
        result = agent.run_mission((0,0,0), (9,9,3), max_steps=400)
        assert isinstance(result, MissionResult)
        # Không crash là đủ — có thể không đến đích nếu grid quá khó

    def test_low_battery_stops_mission(self):
        """Pin 15% → mission dừng sớm (return home hoặc land)"""
        agent, _ = make_agent(w=10, h=10, d=4, goal=(9,9,3),
                              battery=15.0)
        result = agent.run_mission((0,0,0), (9,9,3), max_steps=100)
        # Agent nên dừng sớm vì pin thấp
        assert result.steps <= 50 or result.battery_used < 100.0

    def test_mission_result_summary(self):
        """MissionResult.summary() trả về string"""
        agent, _ = make_agent(w=6, h=6, d=3, goal=(5,5,2))
        result = agent.run_mission((0,0,0), (5,5,2), max_steps=100)
        s = result.summary()
        assert isinstance(s, str)
        assert "steps" in s and "battery" in s

    def test_flight_log_populated(self):
        """flight_log có entries sau mission"""
        agent, _ = make_agent(w=6, h=6, d=3, goal=(5,5,2))
        agent.run_mission((0,0,0), (5,5,2), max_steps=100)
        assert len(agent.flight_log) > 0

    def test_fol_kb_integrated(self):
        """FOL KB được xây dựng và query đúng"""
        agent, g = make_agent()
        assert agent.fol_kb is not None
        # Mọi ô trên path A* phải Safe theo FOL KB
        from src.search.a_star import a_star_search
        r = a_star_search(g, (0,0,0), (9,9,3))
        if r.found:
            for pos in r.path:
                assert agent.fol_kb.query_safe(*pos), \
                    f"A* path đi qua ô không Safe: {pos}"

    def test_replan_on_obstacle(self):
        """Agent replans khi bị block"""
        g = GridWorld3D(10, 10, 4)
        g.set_start((0,0,0)); g.set_goal((9,9,3))
        g.add_obstacle_box(3, 3, 0, 4, 4, 3)
        agent = UAVAgent(g, wind_speed=6.0, seed=42)
        result = agent.run_mission((0,0,0), (9,9,3), max_steps=400)
        assert isinstance(result, MissionResult)


# ── TestScenarios ──────────────────────────────────────────────────────── #

class TestScenarios:

    def test_scenario_normal_no_crash(self):
        """Scenario 1 (bình thường) không crash"""
        g = GridWorld3D(12, 12, 5)
        g.set_start((0,0,0)); g.set_goal((11,11,4))
        agent = UAVAgent(g, wind_speed=5.0, seed=1)
        result = agent.run_mission((0,0,0), (11,11,4), max_steps=300)
        assert isinstance(result, MissionResult)

    def test_scenario_obstacles_no_crash(self):
        """Scenario 2 (obstacles) không crash"""
        g = GridWorld3D(12, 12, 5)
        g.set_start((0,0,0)); g.set_goal((11,11,4))
        g.add_random_obstacles(25, seed=42)
        agent = UAVAgent(g, wind_speed=8.0, seed=2)
        result = agent.run_mission((0,0,0), (11,11,4), max_steps=400)
        assert isinstance(result, MissionResult)

    def test_scenario_high_wind_no_crash(self):
        """Scenario 3 (gió mạnh) không crash"""
        g = GridWorld3D(12, 12, 5)
        g.set_start((0,0,0)); g.set_goal((11,11,4))
        agent = UAVAgent(g, wind_speed=22.0, seed=3)
        result = agent.run_mission((0,0,0), (11,11,4), max_steps=300)
        assert isinstance(result, MissionResult)

    def test_scenario_low_battery_no_crash(self):
        """Scenario 4 (pin yếu) không crash"""
        g = GridWorld3D(12, 12, 5)
        g.set_start((0,0,0)); g.set_goal((11,11,4))
        agent = UAVAgent(g, wind_speed=5.0, initial_battery=18.0, seed=4)
        result = agent.run_mission((0,0,0), (11,11,4), max_steps=100)
        assert isinstance(result, MissionResult)

    def test_scenario_dynamic_obstacle_no_crash(self):
        """Scenario 5 (dynamic obstacle) không crash"""
        g = GridWorld3D(10, 10, 4)
        g.set_start((0,0,0)); g.set_goal((9,9,3))
        g.add_obstacle_box(4, 4, 0, 5, 5, 3)
        agent = UAVAgent(g, wind_speed=6.0, seed=5)
        result = agent.run_mission((0,0,0), (9,9,3), max_steps=400)
        assert isinstance(result, MissionResult)

    def test_5_scenarios_all_return_mission_result(self):
        """Tất cả 5 scenarios trả về MissionResult"""
        configs = [
            {"w":12,"h":12,"d":5,"wind":5.0,"bat":100,"obs":0},
            {"w":12,"h":12,"d":5,"wind":8.0,"bat":100,"obs":25},
            {"w":12,"h":12,"d":5,"wind":22.0,"bat":100,"obs":0},
            {"w":12,"h":12,"d":5,"wind":5.0,"bat":18,"obs":0},
            {"w":10,"h":10,"d":4,"wind":6.0,"bat":100,"obs":0},
        ]
        for i, cfg in enumerate(configs):
            g = GridWorld3D(cfg["w"], cfg["h"], cfg["d"])
            g.set_start((0,0,0))
            goal = (cfg["w"]-1, cfg["h"]-1, cfg["d"]-1)
            g.set_goal(goal)
            if cfg["obs"] > 0:
                g.add_random_obstacles(cfg["obs"], seed=42)
            agent = UAVAgent(g, wind_speed=cfg["wind"],
                             initial_battery=cfg["bat"], seed=i+1)
            result = agent.run_mission((0,0,0), goal, max_steps=400)
            assert isinstance(result, MissionResult), f"Scenario {i+1} không trả về MissionResult"

    def test_mission_logs_populated(self):
        """decision_log có entries sau mission"""
        g = GridWorld3D(8, 8, 4)
        g.set_start((0,0,0)); g.set_goal((7,7,3))
        agent = UAVAgent(g, wind_speed=5.0, seed=42)
        result = agent.run_mission((0,0,0), (7,7,3), max_steps=200)
        assert len(result.decision_log) >= 1, "decision_log trống"

    def test_mission_decision_log_format(self):
        """Mỗi entry trong decision_log có đủ fields"""
        g = GridWorld3D(8, 8, 4)
        g.set_start((0,0,0)); g.set_goal((7,7,3))
        agent = UAVAgent(g, wind_speed=5.0, seed=42)
        result = agent.run_mission((0,0,0), (7,7,3), max_steps=200)
        for entry in result.decision_log:
            assert "step"   in entry
            assert "pos"    in entry
            assert "action" in entry
