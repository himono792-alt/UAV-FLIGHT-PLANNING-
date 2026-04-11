"""
Test Giai đoạn 5: Bayesian Inference + Bayesian Network + MEU
Chạy: python -m pytest tests/test_bayesian.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bayesian.inference import BayesianInference
from src.bayesian.bayesian_network import BayesianNetwork
from src.bayesian.meu_decision import MEUDecisionMaker, ACTIONS


# ── TestBayesianInference ──────────────────────────────────────────────── #

class TestBayesianInference:

    def test_bayes_update_detect_increases_belief(self):
        """P(Obstacle|detect) > P(Obstacle) khi sensor detect"""
        bi = BayesianInference(prior_obstacle=0.1)
        updated = bi.update_belief(0.1, observation=True, sensor_type="lidar")
        assert updated > 0.1, f"Expected > 0.1, got {updated}"

    def test_bayes_update_no_detect_decreases_belief(self):
        """P(Obstacle|no_detect) < P(Obstacle) khi sensor không detect"""
        bi = BayesianInference(prior_obstacle=0.5)
        updated = bi.update_belief(0.5, observation=False, sensor_type="lidar")
        assert updated < 0.5, f"Expected < 0.5, got {updated}"

    def test_bayes_update_in_range(self):
        """P luôn trong [0,1]"""
        bi = BayesianInference()
        for obs in [True, False]:
            for sensor in ["gps", "lidar", "wind_sensor"]:
                p = bi.update_belief(0.3, obs, sensor)
                assert 0.0 <= p <= 1.0, f"Out of range: {p}"

    def test_sensor_fusion_agree_increases_prob(self):
        """2 sensor đồng thuận detect → probability cao hơn 1 sensor"""
        bi = BayesianInference(prior_obstacle=0.1)
        r1 = bi.sensor_fusion({"lidar": True})
        r2 = bi.sensor_fusion({"lidar": True, "gps": True})
        assert r2["obstacle_prob"] >= r1["obstacle_prob"], \
            f"2-sensor {r2['obstacle_prob']:.4f} < 1-sensor {r1['obstacle_prob']:.4f}"

    def test_sensor_fusion_disagree_intermediate(self):
        """2 sensor xung đột → probability trung gian"""
        bi = BayesianInference(prior_obstacle=0.5)
        r_agree   = bi.sensor_fusion({"lidar": True,  "gps": True})
        r_disagree= bi.sensor_fusion({"lidar": True,  "gps": False})
        r_no_obs  = bi.sensor_fusion({"lidar": False, "gps": False})
        assert r_no_obs["obstacle_prob"] < r_disagree["obstacle_prob"] < r_agree["obstacle_prob"], \
            f"no={r_no_obs['obstacle_prob']:.4f} disagree={r_disagree['obstacle_prob']:.4f} agree={r_agree['obstacle_prob']:.4f}"

    def test_sensor_fusion_all_false_low_prob(self):
        """Tất cả sensor báo không có → probability thấp"""
        bi = BayesianInference(prior_obstacle=0.5)
        r = bi.sensor_fusion({"lidar": False, "gps": False, "wind_sensor": False})
        assert r["obstacle_prob"] < 0.5

    def test_sensor_fusion_returns_alert_level(self):
        """sensor_fusion trả về alert_level hợp lệ"""
        bi = BayesianInference()
        r = bi.sensor_fusion({"lidar": True})
        assert r["alert_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_alert_level_critical_when_high_prob(self):
        """Probability cao → alert CRITICAL"""
        bi = BayesianInference(prior_obstacle=0.9)
        r = bi.sensor_fusion({"lidar": True, "gps": True})
        assert r["alert_level"] in ["HIGH", "CRITICAL"]

    def test_marginal_evidence_sums_correctly(self):
        """P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H) phải hợp lý"""
        bi = BayesianInference()
        pe = bi.marginal_evidence(0.3, "lidar", True)
        assert 0.0 < pe <= 1.0

    def test_summary_format(self):
        bi = BayesianInference()
        r = bi.sensor_fusion({"lidar": True})
        s = bi.summary(r)
        assert "P(Obstacle)" in s and "Alert" in s


# ── TestBayesianNetwork ────────────────────────────────────────────────── #

class TestBayesianNetwork:

    def test_weather_marginal_sums_to_one(self):
        """P(Weather) sum = 1"""
        bn = BayesianNetwork()
        marg = bn.marginal("Weather")
        total = sum(marg.values())
        assert abs(total - 1.0) < 1e-4, f"Sum = {total}"

    def test_wind_marginal_sums_to_one(self):
        """P(WindSpeed) marginal sum = 1"""
        bn = BayesianNetwork()
        marg = bn.marginal("WindSpeed")
        total = sum(marg.values())
        assert abs(total - 1.0) < 1e-4, f"Sum = {total}"

    def test_flight_safety_conditional_stormy(self):
        """P(FlightSafety | Weather=Stormy, BatteryLevel=Critical) → Dangerous cao"""
        bn = BayesianNetwork()
        dist = bn.query("FlightSafety",
                        {"Weather": "Stormy", "BatteryLevel": "Critical"})
        assert dist["Dangerous"] > dist["Safe"], \
            f"Expected Dangerous > Safe: {dist}"

    def test_flight_safety_conditional_clear(self):
        """P(FlightSafety | Weather=Clear, BatteryLevel=Full) → Safe cao"""
        bn = BayesianNetwork()
        dist = bn.query("FlightSafety",
                        {"Weather": "Clear", "BatteryLevel": "Full"})
        assert dist["Safe"] > dist["Dangerous"], \
            f"Expected Safe > Dangerous: {dist}"

    def test_query_sums_to_one(self):
        """Mọi query đều sum = 1"""
        bn = BayesianNetwork()
        for target in ["FlightSafety", "PathDecision", "WindSpeed"]:
            dist = bn.query(target, {"Weather": "Rainy", "BatteryLevel": "Medium"})
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-3, f"{target} sum = {total}"

    def test_path_decision_stormy_prefers_land(self):
        """P(PathDecision | Weather=Stormy, BatteryLevel=Critical) → Land cao nhất"""
        bn = BayesianNetwork()
        dist = bn.query("PathDecision",
                        {"Weather": "Stormy", "BatteryLevel": "Critical"})
        assert dist["Land"] > dist["DirectPath"], f"{dist}"

    def test_markov_blanket_flight_safety(self):
        """MB(FlightSafety) = {WindSpeed, FlightRange, PathDecision}"""
        bn = BayesianNetwork()
        mb = bn.markov_blanket("FlightSafety")
        assert "WindSpeed"   in mb, "WindSpeed phải trong MB"
        assert "FlightRange" in mb, "FlightRange phải trong MB"
        assert "PathDecision" in mb, "PathDecision phải trong MB"
        assert "FlightSafety" not in mb, "Node không nằm trong MB của chính nó"

    def test_markov_blanket_weather(self):
        """MB(Weather) = {WindSpeed} — chỉ có con, không có cha"""
        bn = BayesianNetwork()
        mb = bn.markov_blanket("Weather")
        assert "WindSpeed" in mb

    def test_all_node_values_positive(self):
        """Mọi giá trị của node root đều > 0"""
        bn = BayesianNetwork()
        marg = bn.marginal("Weather")
        for val, prob in marg.items():
            assert prob > 0, f"P(Weather={val}) = 0"

    def test_describe_format(self):
        bn = BayesianNetwork()
        desc = bn.describe()
        assert "Weather" in desc and "FlightSafety" in desc


# ── TestMEU ────────────────────────────────────────────────────────────── #

class TestMEU:

    def test_meu_good_weather_fly_direct(self):
        """Evidence tốt → fly_direct hoặc fly_safe (không return_home/land)"""
        meu = MEUDecisionMaker()
        action, eu = meu.decide({"Weather": "Clear", "BatteryLevel": "Full"})
        assert action in ("fly_direct", "fly_safe"), \
            f"Điều kiện tốt không nên chọn {action}"

    def test_meu_bad_weather_land(self):
        """Stormy + Critical battery → không chọn fly_direct"""
        meu = MEUDecisionMaker()
        action, eu = meu.decide({"Weather": "Stormy", "BatteryLevel": "Critical"})
        assert action != "fly_direct", \
            f"Điều kiện nguy hiểm không nên chọn fly_direct, got {action}"

    def test_meu_risky_fly_safe(self):
        """Rainy + Low battery → fly_safe hoặc return_home (không fly_direct)"""
        meu = MEUDecisionMaker()
        action, eu = meu.decide({"Weather": "Rainy", "BatteryLevel": "Low"})
        assert action != "fly_direct", \
            f"fly_direct không nên được chọn trong điều kiện xấu, got {action}"

    def test_eu_land_better_than_fly_direct_dangerous(self):
        """Trong điều kiện nguy hiểm: EU(land) > EU(fly_direct)"""
        meu = MEUDecisionMaker()
        safety_dist = {"Safe": 0.02, "Risky": 0.08, "Dangerous": 0.90}
        eu_land   = meu.expected_utility("land",       safety_dist)
        eu_direct = meu.expected_utility("fly_direct", safety_dist)
        assert eu_land > eu_direct, \
            f"EU(land)={eu_land:.2f} should > EU(fly_direct)={eu_direct:.2f}"

    def test_eu_fly_direct_better_than_land_safe(self):
        """Trong điều kiện an toàn: EU(fly_direct) > EU(land)"""
        meu = MEUDecisionMaker()
        safety_dist = {"Safe": 0.90, "Risky": 0.08, "Dangerous": 0.02}
        eu_direct = meu.expected_utility("fly_direct", safety_dist)
        eu_land   = meu.expected_utility("land",       safety_dist)
        assert eu_direct > eu_land, \
            f"EU(fly_direct)={eu_direct:.2f} should > EU(land)={eu_land:.2f}"

    def test_decide_returns_valid_action(self):
        """decide() luôn trả về 1 trong các actions hợp lệ"""
        meu = MEUDecisionMaker()
        for ev in [
            {"Weather": "Clear",  "BatteryLevel": "Full"},
            {"Weather": "Cloudy", "BatteryLevel": "Medium"},
            {"Weather": "Stormy", "BatteryLevel": "Critical"},
        ]:
            action, eu = meu.decide(ev)
            assert action in ACTIONS, f"Invalid action: {action}"

    def test_all_utilities_all_actions(self):
        """all_utilities trả về EU cho đúng 4 actions"""
        meu = MEUDecisionMaker()
        utils = meu.all_utilities({"Weather": "Cloudy", "BatteryLevel": "Medium"})
        assert set(utils.keys()) == set(ACTIONS)

    def test_explain_decision_format(self):
        """explain_decision trả về string với các key fields"""
        meu = MEUDecisionMaker()
        explanation = meu.explain_decision({"Weather": "Rainy",
                                            "BatteryLevel": "Low"})
        assert "MEU Decision" in explanation
        assert "FlightSafety" in explanation
        assert "CHOSEN" in explanation

    def test_meu_consistency(self):
        """decide() nhất quán với expected_utility thủ công"""
        meu = MEUDecisionMaker()
        evidence = {"Weather": "Cloudy", "BatteryLevel": "Medium"}
        best_action, best_eu = meu.decide(evidence)
        utils = meu.all_utilities(evidence)
        manual_best = max(utils, key=utils.__getitem__)
        assert best_action == manual_best, \
            f"decide={best_action} vs manual={manual_best}"
        assert abs(best_eu - utils[best_action]) < 1e-4
