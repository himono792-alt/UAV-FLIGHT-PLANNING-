"""
Test Giai đoạn 4: KB-Agent + Propositional Logic + FOL
Chạy: python -m pytest tests/test_knowledge.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_world import GridWorld3D, CellType
from src.search.a_star import a_star_search
from src.knowledge.propositional import (
    evaluate_rules, explain_action, get_rule_summary,
    ACTION_AVOID, ACTION_EXIT_NFZ, ACTION_LAND,
    ACTION_RETURN_HOME, ACTION_REDUCE_SPEED, ACTION_CONTINUE,
    BATTERY_LOW_THRESHOLD, WIND_HIGH_THRESHOLD, WIND_DANGEROUS_THRESHOLD
)
from src.knowledge.kb_agent import KnowledgeBase, KBAgent
from src.knowledge.fol_rules import FOLKnowledgeBase


# ── Helpers ────────────────────────────────────────────────────────────── #

def make_clear_percept(**overrides):
    base = {
        "position":          (1, 1, 0),
        "battery":           80.0,
        "wind_speed":        5.0,
        "obstacle_detected": False,
        "in_no_fly_zone":    False,
        "on_path":           True,
        "near_charger":      False,
    }
    base.update(overrides)
    return base


def make_grid_with_obstacle():
    g = GridWorld3D(8, 8, 4)
    g.set_start((0, 0, 0))
    g.set_goal((7, 7, 3))
    g.add_obstacle_box(3, 3, 0, 3, 3, 3)  # column obstacle
    return g


# ── TestKnowledgeBase ──────────────────────────────────────────────────── #

class TestKnowledgeBase:

    def test_tell_ask_basic(self):
        """TELL fact → ASK trả về đúng giá trị"""
        kb = KnowledgeBase()
        kb.tell("obstacle_detected", True)
        assert kb.ask("obstacle_detected") is True

    def test_tell_overwrite(self):
        """TELL fact hai lần → giá trị mới nhất được giữ"""
        kb = KnowledgeBase()
        kb.tell("battery", 80)
        kb.tell("battery", 15)
        assert kb.ask("battery") == 15

    def test_retract(self):
        """RETRACT fact → ASK trả về None"""
        kb = KnowledgeBase()
        kb.tell("obstacle_detected", True)
        kb.retract("obstacle_detected")
        assert kb.ask("obstacle_detected") is None

    def test_retract_nonexistent(self):
        """RETRACT key không tồn tại không raise exception"""
        kb = KnowledgeBase()
        kb.retract("nonexistent_key")  # không nên raise

    def test_tell_percept_derive_battery_low(self):
        """tell_percept tự derive battery_low từ battery value"""
        kb = KnowledgeBase()
        kb.tell_percept({"battery": 15.0})
        assert kb.ask("battery_low") is True        # 15 < 20 → True
        assert kb.ask("battery_critical") is False  # 15 >= 10 → False

    def test_tell_percept_derive_battery_ok(self):
        kb = KnowledgeBase()
        kb.tell_percept({"battery": 80.0})
        assert kb.ask("battery_low") is False
        assert kb.ask("battery_critical") is False

    def test_tell_percept_derive_wind(self):
        kb = KnowledgeBase()
        kb.tell_percept({"wind_speed": 20.0})
        assert kb.ask("wind_high") is True
        assert kb.ask("wind_dangerous") is False

    def test_tell_percept_derive_wind_dangerous(self):
        kb = KnowledgeBase()
        kb.tell_percept({"wind_speed": 30.0})
        assert kb.ask("wind_high") is True
        assert kb.ask("wind_dangerous") is True

    def test_ask_action_returns_string(self):
        kb = KnowledgeBase()
        kb.tell_percept(make_clear_percept())
        action = kb.ask("action")
        assert isinstance(action, str)

    def test_summary_format(self):
        kb = KnowledgeBase()
        kb.tell("battery", 80)
        s = kb.summary()
        assert "KB" in s and "battery" in s


# ── TestKBAgent ────────────────────────────────────────────────────────── #

class TestKBAgent:

    def test_obstacle_detected(self):
        """Percept obstacle → AVOID"""
        agent = KBAgent()
        p = make_clear_percept(obstacle_detected=True)
        assert agent.agent_function(p) == ACTION_AVOID

    def test_battery_low_return_home(self):
        """Battery 15%, không gần charger → RETURN_HOME"""
        agent = KBAgent()
        p = make_clear_percept(battery=15.0, near_charger=False)
        assert agent.agent_function(p) == ACTION_RETURN_HOME

    def test_battery_low_near_charger(self):
        """Battery thấp nhưng gần charger → không RETURN_HOME"""
        agent = KBAgent()
        p = make_clear_percept(battery=15.0, near_charger=True)
        action = agent.agent_function(p)
        assert action != ACTION_RETURN_HOME

    def test_wind_high(self):
        """Wind 20 m/s → REDUCE_SPEED"""
        agent = KBAgent()
        p = make_clear_percept(wind_speed=20.0)
        assert agent.agent_function(p) == ACTION_REDUCE_SPEED

    def test_wind_dangerous(self):
        """Wind 30 m/s → LAND"""
        agent = KBAgent()
        p = make_clear_percept(wind_speed=30.0)
        assert agent.agent_function(p) == ACTION_LAND

    def test_no_fly_zone(self):
        """in_no_fly_zone = True → EXIT_NFZ"""
        agent = KBAgent()
        p = make_clear_percept(in_no_fly_zone=True)
        assert agent.agent_function(p) == ACTION_EXIT_NFZ

    def test_all_clear(self):
        """Mọi điều kiện OK, on_path = True → CONTINUE"""
        agent = KBAgent()
        p = make_clear_percept()
        assert agent.agent_function(p) == ACTION_CONTINUE

    def test_priority_obstacle_over_battery(self):
        """Obstacle + battery_low cùng lúc → AVOID (priority cao hơn)"""
        agent = KBAgent()
        p = make_clear_percept(obstacle_detected=True, battery=15.0, near_charger=False)
        assert agent.agent_function(p) == ACTION_AVOID

    def test_priority_nfz_over_battery(self):
        """NoFlyZone + battery_low → EXIT_NFZ (priority cao hơn)"""
        agent = KBAgent()
        p = make_clear_percept(in_no_fly_zone=True, battery=15.0, near_charger=False)
        assert agent.agent_function(p) == ACTION_EXIT_NFZ

    def test_timestep_increments(self):
        """Timestep tăng sau mỗi lần gọi agent_function"""
        agent = KBAgent()
        agent.agent_function(make_clear_percept())
        agent.agent_function(make_clear_percept())
        assert agent.t == 2

    def test_action_log(self):
        """Action log ghi lại đúng số bước"""
        agent = KBAgent()
        agent.agent_function(make_clear_percept())
        agent.agent_function(make_clear_percept())
        log = agent.get_action_history()
        assert len(log) == 2

    def test_reset(self):
        """Reset agent xóa facts và log"""
        agent = KBAgent()
        agent.agent_function(make_clear_percept())
        agent.reset()
        assert agent.t == 0
        assert len(agent.action_log) == 0

    def test_last_action_stored(self):
        """KB lưu last_action sau mỗi bước"""
        agent = KBAgent()
        action = agent.agent_function(make_clear_percept(obstacle_detected=True))
        assert agent.kb.ask("last_action") == ACTION_AVOID


# ── TestFOL ────────────────────────────────────────────────────────────── #

class TestFOL:

    def test_safe_free_cell(self):
        """Ô FREE → Safe(x,y,z) = True"""
        g = GridWorld3D(5, 5, 3)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_safe(2, 2, 1) is True

    def test_safe_obstacle_cell(self):
        """Ô obstacle → Safe = False"""
        g = GridWorld3D(5, 5, 3)
        g.add_obstacle_box(2, 2, 0, 2, 2, 2)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_safe(2, 2, 1) is False

    def test_safe_no_fly_zone(self):
        """Ô NFZ → Safe = False"""
        g = GridWorld3D(5, 5, 3)
        g.set_cell((1, 1, 1), CellType.NO_FLY_ZONE)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_safe(1, 1, 1) is False

    def test_get_safe_neighbors_no_obstacle(self):
        """safe_neighbors không trả về ô obstacle"""
        g = make_grid_with_obstacle()
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        # ô (2,3,0) kề obstacle column tại (3,3,0)
        neighbors = fol.get_safe_neighbors((2, 3, 0))
        for nb in neighbors:
            assert fol.query_safe(*nb), f"Neighbor {nb} không phải Safe"

    def test_can_fly_safe_adjacent(self):
        """CanFly: target an toàn và kề → True"""
        g = GridWorld3D(5, 5, 3)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_can_fly((0, 0, 0), (1, 0, 0)) is True

    def test_can_fly_obstacle(self):
        """CanFly: target là obstacle → False"""
        g = GridWorld3D(5, 5, 3)
        g.add_obstacle_box((1, 0, 0), (1, 0, 0))
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_can_fly((0, 0, 0), (1, 0, 0)) is False

    def test_add_predicate_obstacle(self):
        """add_predicate tự động cập nhật Safe"""
        g = GridWorld3D(5, 5, 3)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_safe(2, 2, 1) is True
        fol.add_predicate("Obstacle", 2, 2, 1)
        assert fol.query_safe(2, 2, 1) is False

    def test_remove_predicate_obstacle(self):
        """remove_predicate → ô có thể Safe trở lại"""
        g = GridWorld3D(5, 5, 3)
        g.add_obstacle_box((2, 2, 1), (2, 2, 1))
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.query_safe(2, 2, 1) is False
        fol.remove_predicate("Obstacle", 2, 2, 1)
        assert fol.query_safe(2, 2, 1) is True

    def test_find_nearest_charger(self):
        """find_nearest_charger trả về charger gần nhất"""
        g = GridWorld3D(10, 10, 5)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        fol.add_charger(9, 9, 4)
        fol.add_charger(1, 0, 0)
        nearest = fol.find_nearest_charger((0, 0, 0))
        assert nearest == (1, 0, 0), f"Expected (1,0,0), got {nearest}"

    def test_find_nearest_charger_none(self):
        """Không có charger → trả về None"""
        g = GridWorld3D(5, 5, 3)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        assert fol.find_nearest_charger((0, 0, 0)) is None

    def test_summary_format(self):
        g = GridWorld3D(5, 5, 3)
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()
        s = fol.summary()
        assert "Safe" in s and "Obstacle" in s


# ── TestTichHop ────────────────────────────────────────────────────────── #

class TestTichHop:

    def test_kb_loc_a_star(self):
        """A* với safe_neighbors từ FOL KB không đi qua obstacle"""
        g = make_grid_with_obstacle()
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()

        # A* bình thường
        r = a_star_search(g, (0, 0, 0), (7, 7, 3))
        if not r.found:
            return

        # Kiểm tra mọi bước trên path đều Safe theo KB
        for pos in r.path:
            assert fol.query_safe(*pos), f"Path đi qua ô không safe: {pos}"

    def test_replan_on_new_obstacle(self):
        """
        Phát hiện obstacle mới → TELL KB → A* chạy lại
        """
        g = GridWorld3D(8, 8, 4)
        g.set_start((0, 0, 0))
        g.set_goal((7, 7, 3))
        fol = FOLKnowledgeBase(g)
        fol.build_from_grid()

        # A* lần 1 trên grid sạch
        r1 = a_star_search(g, (0, 0, 0), (7, 7, 3))
        assert r1.found

        # Obstacle xuất hiện trên path
        if len(r1.path) > 2:
            mid = r1.path[len(r1.path) // 2]
            # TELL KB: obstacle mới
            fol.add_predicate("Obstacle", *mid)
            g.set_cell(mid, CellType.OBSTACLE)

            # A* lần 2 — replanning
            r2 = a_star_search(g, (0, 0, 0), (7, 7, 3))
            if r2.found:
                # Path mới không đi qua obstacle
                assert mid not in r2.path, "Replan path vẫn đi qua obstacle mới"

    def test_kbagent_sequence(self):
        """KB Agent xử lý chuỗi percepts đúng thứ tự"""
        agent = KBAgent()
        actions = []
        for percept in [
            make_clear_percept(obstacle_detected=True),    # AVOID
            make_clear_percept(wind_speed=20.0),           # REDUCE_SPEED
            make_clear_percept(),                          # CONTINUE
        ]:
            actions.append(agent.agent_function(percept))

        assert actions[0] == ACTION_AVOID
        assert actions[1] == ACTION_REDUCE_SPEED
        assert actions[2] == ACTION_CONTINUE
        assert agent.t == 3

    def test_explain_action(self):
        """explain_action trả về dict với đầy đủ fields"""
        facts = {"obstacle_detected": True, "on_path": True}
        result = explain_action(facts)
        assert "action" in result
        assert "rule_id" in result
        assert result["action"] == ACTION_AVOID
        assert result["rule_id"] == "R1"
