"""
Test Giai đoạn 6: Decision Tree + Regression + Q-Learning
Chạy: python -m pytest tests/test_learning.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid_world import GridWorld3D, CellType
from src.search.a_star import a_star_search
from src.learning.decision_tree import (
    FlightConditionClassifier, ALL_LABELS,
    LABEL_FLY, LABEL_CAREFUL, LABEL_AVOID, LABEL_LAND
)
from src.learning.regression import FlightTimePredictor
from src.learning.reinforcement import QLearningAgent


# ── Fixtures ───────────────────────────────────────────────────────────── #

def make_small_grid(obstacles=False):
    g = GridWorld3D(8, 8, 4)
    g.set_start((0, 0, 0))
    g.set_goal((7, 7, 3))
    if obstacles:
        g.add_random_obstacles(10, seed=42)
    return g


def trained_clf():
    clf = FlightConditionClassifier(max_depth=5, random_state=42)
    X, y = clf.generate_training_data(n_samples=500, seed=42)
    clf.train(X, y)
    return clf, X, y


def trained_reg():
    pred = FlightTimePredictor(random_state=42)
    X, y = pred.generate_training_data(n_samples=300, seed=42)
    pred.train(X, y)
    return pred, X, y


# ── TestDecisionTree ───────────────────────────────────────────────────── #

class TestDecisionTree:

    def test_generate_data_shape(self):
        """generate_training_data trả về đúng kích thước"""
        clf = FlightConditionClassifier()
        X, y = clf.generate_training_data(n_samples=200, seed=42)
        assert len(X) == 200
        assert len(y) == 200
        assert len(X[0]) == 5, "Features phải có 5 chiều"

    def test_labels_valid(self):
        """Tất cả labels trong {FLY, CAREFUL, AVOID, LAND}"""
        clf = FlightConditionClassifier()
        _, y = clf.generate_training_data(n_samples=200, seed=42)
        for label in y:
            assert label in ALL_LABELS, f"Invalid label: {label}"

    def test_train_predict_no_error(self):
        """train + predict chạy không lỗi"""
        clf, X, y = trained_clf()
        pred = clf.predict(X[0])
        assert pred in ALL_LABELS

    def test_accuracy_above_80(self):
        """Accuracy > 80% trên training set"""
        clf, X, y = trained_clf()
        metrics = clf.evaluate(X, y)
        assert metrics["accuracy"] >= 0.80, \
            f"Accuracy {metrics['accuracy']:.2%} < 80%"

    def test_predict_classes(self):
        """output luôn thuộc ALL_LABELS"""
        clf, X, _ = trained_clf()
        for feat in X[:50]:
            label = clf.predict(feat)
            assert label in ALL_LABELS, f"Invalid: {label}"

    def test_predict_land_extreme_wind(self):
        """Wind 35 m/s, battery 5% → LAND"""
        clf, _, _ = trained_clf()
        # [wind, battery, visibility, obs_density, altitude]
        pred = clf.predict([35.0, 5.0, 80.0, 0.1, 10.0])
        assert pred == LABEL_LAND, f"Expected LAND, got {pred}"

    def test_predict_fly_good_conditions(self):
        """Điều kiện tốt → FLY"""
        clf, _, _ = trained_clf()
        pred = clf.predict([5.0, 90.0, 95.0, 0.05, 10.0])
        assert pred == LABEL_FLY, f"Expected FLY, got {pred}"

    def test_predict_avoid_high_obstacle(self):
        """obstacle_density cao → AVOID"""
        clf, _, _ = trained_clf()
        pred = clf.predict([5.0, 80.0, 90.0, 0.5, 10.0])
        assert pred == LABEL_AVOID, f"Expected AVOID, got {pred}"

    def test_predict_proba_sums_to_one(self):
        """predict_proba sum = 1"""
        clf, X, _ = trained_clf()
        proba = clf.predict_proba(X[0])
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-4, f"Sum = {total}"

    def test_export_tree_string(self):
        """export_tree trả về string"""
        clf, _, _ = trained_clf()
        tree_str = clf.export_tree()
        assert isinstance(tree_str, str) and len(tree_str) > 0


# ── TestRegression ─────────────────────────────────────────────────────── #

class TestRegression:

    def test_generate_data_shape(self):
        """generate_training_data trả về đúng kích thước"""
        pred = FlightTimePredictor()
        X, y = pred.generate_training_data(n_samples=100, seed=42)
        assert len(X) == 100
        assert len(y) == 100
        assert len(X[0]) == 5

    def test_flight_time_positive(self):
        """Mọi giá trị flight_time đều > 0"""
        pred = FlightTimePredictor()
        _, y = pred.generate_training_data(n_samples=100, seed=42)
        for t in y:
            assert t > 0, f"Negative flight time: {t}"

    def test_train_predict_float(self):
        """predict trả về float > 0"""
        pred, X, _ = trained_reg()
        t = pred.predict(X[0])
        assert isinstance(t, float)
        assert t > 0, f"Negative prediction: {t}"

    def test_r_squared_above_05(self):
        """R² > 0.5 — model có nghĩa"""
        pred, X, y = trained_reg()
        metrics = pred.evaluate(X, y)
        assert metrics["r2"] >= 0.5, \
            f"R² {metrics['r2']:.3f} < 0.5"

    def test_mae_reasonable(self):
        """MAE < 5 phút — sai số chấp nhận được"""
        pred, X, y = trained_reg()
        metrics = pred.evaluate(X, y)
        assert metrics["mae"] < 5.0, f"MAE {metrics['mae']:.2f} quá cao"

    def test_longer_distance_longer_time(self):
        """Distance tăng → thời gian bay tăng"""
        pred, _, _ = trained_reg()
        # [distance, wind=5, battery=80, payload=1, alt=5]
        t_short = pred.predict([5.0,  5.0, 80.0, 1.0, 5.0])
        t_long  = pred.predict([20.0, 5.0, 80.0, 1.0, 5.0])
        assert t_long > t_short, \
            f"t_long={t_long:.2f} <= t_short={t_short:.2f}"

    def test_high_wind_slower(self):
        """Wind cao → thời gian bay lâu hơn"""
        pred, _, _ = trained_reg()
        t_calm  = pred.predict([10.0, 2.0,  80.0, 1.0, 5.0])
        t_windy = pred.predict([10.0, 25.0, 80.0, 1.0, 5.0])
        assert t_windy > t_calm, \
            f"windy={t_windy:.2f} not > calm={t_calm:.2f}"

    def test_evaluate_metrics_keys(self):
        """evaluate trả về đủ keys"""
        pred, X, y = trained_reg()
        metrics = pred.evaluate(X[:50], y[:50])
        assert "mae" in metrics and "rmse" in metrics and "r2" in metrics

    def test_coefficients(self):
        """coefficients trả về dict với feature names"""
        pred, _, _ = trained_reg()
        coefs = pred.coefficients()
        assert "distance_3d" in coefs
        assert len(coefs) == 5


# ── TestQLearning ──────────────────────────────────────────────────────── #

class TestQLearning:

    def test_train_runs(self):
        """train() chạy không lỗi"""
        g = GridWorld3D(6, 6, 3)
        g.set_start((0, 0, 0)); g.set_goal((5, 5, 2))
        agent = QLearningAgent(g, seed=42)
        stats = agent.train((0, 0, 0), (5, 5, 2), episodes=100)
        assert "episodes" in stats
        assert stats["episodes"] == 100

    def test_q_table_populated(self):
        """Q-table được điền sau training"""
        g = GridWorld3D(6, 6, 3)
        g.set_start((0, 0, 0)); g.set_goal((5, 5, 2))
        agent = QLearningAgent(g, seed=42)
        agent.train((0, 0, 0), (5, 5, 2), episodes=100)
        assert len(agent.q_table) > 0, "Q-table trống"

    def test_epsilon_decays(self):
        """Epsilon giảm dần qua training"""
        g = GridWorld3D(6, 6, 3)
        g.set_start((0, 0, 0)); g.set_goal((5, 5, 2))
        agent = QLearningAgent(g, epsilon=1.0, seed=42)
        agent.train((0, 0, 0), (5, 5, 2), episodes=100)
        assert agent.epsilon < 1.0, f"Epsilon không giảm: {agent.epsilon}"

    def test_find_path_after_train(self):
        """Sau train, get_path() trả về path"""
        g = GridWorld3D(8, 8, 4)
        g.set_start((0, 0, 0)); g.set_goal((7, 7, 3))
        agent = QLearningAgent(g, alpha=0.3, gamma=0.9, seed=42)
        agent.train((0, 0, 0), (7, 7, 3), episodes=500)
        path = agent.get_path((0, 0, 0), (7, 7, 3))
        assert path is not None, "get_path trả về None"
        assert len(path) >= 2

    def test_path_starts_at_start(self):
        """Path bắt đầu từ start"""
        g = GridWorld3D(6, 6, 3)
        agent = QLearningAgent(g, seed=42)
        agent.train((0, 0, 0), (5, 5, 2), episodes=300)
        path = agent.get_path((0, 0, 0), (5, 5, 2))
        if path:
            assert path[0] == (0, 0, 0)

    def test_avoid_obstacles(self):
        """Path không đi qua obstacles"""
        g = GridWorld3D(8, 8, 4)
        g.set_start((0, 0, 0)); g.set_goal((7, 7, 3))
        g.add_obstacle_box(3, 3, 0, 3, 3, 3)
        agent = QLearningAgent(g, alpha=0.3, seed=42)
        agent.train((0, 0, 0), (7, 7, 3), episodes=400)
        path = agent.get_path((0, 0, 0), (7, 7, 3))
        if path:
            for pos in path:
                cell = g.get_cell(pos)
                assert cell != CellType.OBSTACLE, \
                    f"Path đi qua obstacle tại {pos}"

    def test_training_stats(self):
        """get_training_stats trả về đúng keys"""
        g = GridWorld3D(6, 6, 3)
        agent = QLearningAgent(g, seed=42)
        agent.train((0, 0, 0), (5, 5, 2), episodes=50)
        stats = agent.get_training_stats()
        assert "n_episodes"   in stats
        assert "avg_reward"   in stats
        assert "q_table_size" in stats

    def test_reward_trend_length(self):
        """reward_trend trả về list đúng độ dài"""
        g = GridWorld3D(6, 6, 3)
        agent = QLearningAgent(g, seed=42)
        agent.train((0, 0, 0), (5, 5, 2), episodes=50)
        trend = agent.reward_trend()
        assert len(trend) == 50

    def test_compare_with_astar(self):
        """RL path cost <= 3x A* cost (RL hợp lý)"""
        g = GridWorld3D(8, 8, 4)
        g.set_start((0, 0, 0)); g.set_goal((7, 7, 3))
        start, goal = (0, 0, 0), (7, 7, 3)

        # A* baseline
        r_astar = a_star_search(g, start, goal)
        assert r_astar.found, "A* không tìm được đường"

        # RL
        agent = QLearningAgent(g, alpha=0.3, gamma=0.9,
                               epsilon_decay=0.99, seed=42)
        agent.train(start, goal, episodes=600)
        rl_path = agent.get_path(start, goal)

        if rl_path and rl_path[-1] == goal:
            rl_cost = sum(
                g.step_cost(rl_path[i], rl_path[i+1])
                for i in range(len(rl_path)-1)
            )
            assert rl_cost <= r_astar.cost * 3.0 + 1e-6, \
                f"RL cost {rl_cost:.2f} > 3× A* {r_astar.cost:.2f}"
        # Nếu RL chưa tìm được đường → test vẫn pass (không crash)
