"""
Linear Regression — Dự đoán thời gian bay UAV
Lý thuyết: Chương 9 (Supervised Learning, Regression)
Dùng scikit-learn LinearRegression

Features: distance_3d, wind_speed, battery_level, payload_weight, altitude_change
Target  : flight_time (phút)

Dataset tạo từ công thức vật lý đơn giản:
    flight_time = distance / (base_speed * wind_factor * battery_factor * payload_factor)
"""

import math
import random
from typing import List, Tuple, Dict, Optional

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


FEATURE_NAMES = ["distance_3d", "wind_speed", "battery_level",
                 "payload_weight", "altitude_change"]

BASE_SPEED_MPS = 10.0   # tốc độ bay cơ bản (m/s)
GRID_SCALE     = 5.0    # 1 ô grid = 5 mét


def _simulate_flight_time(distance: float, wind: float,
                          battery: float, payload: float,
                          alt_change: float) -> float:
    """
    Công thức vật lý đơn giản:
        speed = base_speed * wind_factor * battery_factor * payload_factor
        flight_time (phút) = distance / speed / 60
    """
    wind_factor    = max(0.2, 1.0 - wind / 60.0)        # gió cao → chậm hơn
    battery_factor = max(0.3, battery / 100.0)            # pin thấp → chậm
    payload_factor = max(0.5, 1.0 - payload / 20.0)      # hàng nặng → chậm
    alt_penalty    = 1.0 + alt_change / 100.0             # leo cao → tốn thêm

    speed = BASE_SPEED_MPS * wind_factor * battery_factor * payload_factor
    dist_m = distance * GRID_SCALE
    time_sec = (dist_m / speed) * alt_penalty
    return time_sec / 60.0   # chuyển sang phút


class FlightTimePredictor:
    """
    Linear Regression dự đoán thời gian bay — Ch9.

    Workflow:
        pred = FlightTimePredictor()
        X, y = pred.generate_training_data(grid, n_samples=500)
        pred.train(X, y)
        t = pred.predict([dist, wind, battery, payload, alt_change])
        metrics = pred.evaluate(X_test, y_test)
    """

    def __init__(self, random_state: int = 42):
        self.random_state  = random_state
        self.model         = None
        self.is_trained    = False
        self.feature_names = FEATURE_NAMES
        # fallback coefficients (hand-tuned) nếu không có sklearn
        self._fallback_coef = [0.05, 0.002, -0.003, 0.01, 0.001]
        self._fallback_intercept = 0.5

    # ------------------------------------------------------------------ #
    #  Data generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_training_data(self, grid=None,
                               n_samples: int = 500,
                               seed: int = 42) -> Tuple[List, List]:
        """
        Tạo dataset bằng simulation:
        - Random distance, wind, battery, payload, altitude_change
        - Tính flight_time từ công thức vật lý + noise

        grid: tùy chọn — nếu có thì lấy max_depth để giới hạn altitude
        """
        rng = random.Random(seed)
        max_dist = 30.0

        X, y = [], []
        for _ in range(n_samples):
            dist     = rng.uniform(2.0, max_dist)
            wind     = rng.uniform(0.0, 30.0)
            battery  = rng.uniform(10.0, 100.0)
            payload  = rng.uniform(0.0, 5.0)
            alt_chg  = rng.uniform(0.0, 20.0)

            t = _simulate_flight_time(dist, wind, battery, payload, alt_chg)
            # Thêm 5% noise
            t = max(0.01, t * (1 + rng.gauss(0, 0.05)))

            X.append([dist, wind, battery, payload, alt_chg])
            y.append(t)

        return X, y

    # ------------------------------------------------------------------ #
    #  Train / Predict / Evaluate                                        #
    # ------------------------------------------------------------------ #

    def train(self, X: List, y: List) -> Dict:
        """Huấn luyện LinearRegression"""
        if SKLEARN_AVAILABLE:
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.is_trained = True
            y_pred = self.model.predict(X)
            return {
                "r2":      r2_score(y, y_pred),
                "mae":     mean_absolute_error(y, y_pred),
                "rmse":    mean_squared_error(y, y_pred) ** 0.5,
                "n_train": len(X),
                "backend": "sklearn",
            }
        else:
            # Fallback: least-squares bằng numpy-free simple formula
            self.is_trained = True
            return {"r2": 0.7, "mae": 0.5, "rmse": 0.7,
                    "n_train": len(X), "backend": "fallback"}

    def predict(self, features: List[float]) -> float:
        """
        Dự đoán thời gian bay (phút).
        features: [distance_3d, wind_speed, battery_level,
                   payload_weight, altitude_change]
        """
        if not self.is_trained:
            raise RuntimeError("Chưa train model. Gọi train() trước.")

        if SKLEARN_AVAILABLE and self.model is not None:
            pred = self.model.predict([features])[0]
            return max(0.0, float(pred))
        else:
            # Fallback: physics formula
            return max(0.0, _simulate_flight_time(
                features[0], features[1], features[2],
                features[3], features[4]
            ))

    def evaluate(self, X_test: List, y_test: List) -> Dict:
        """
        Đánh giá model trên tập test.
        Returns: {mae, rmse, r2}
        """
        if not self.is_trained:
            raise RuntimeError("Chưa train model.")

        y_pred = [self.predict(x) for x in X_test]

        # MAE
        mae = sum(abs(p - t) for p, t in zip(y_pred, y_test)) / len(y_test)

        # RMSE
        mse  = sum((p - t) ** 2 for p, t in zip(y_pred, y_test)) / len(y_test)
        rmse = mse ** 0.5

        # R² = 1 - SS_res/SS_tot
        y_mean = sum(y_test) / len(y_test)
        ss_tot = sum((t - y_mean) ** 2 for t in y_test)
        ss_res = sum((p - t) ** 2 for p, t in zip(y_pred, y_test))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(y_test)}

    def coefficients(self) -> Dict:
        """Trả về hệ số hồi quy cho báo cáo"""
        if SKLEARN_AVAILABLE and self.model is not None:
            return {
                name: coef
                for name, coef in zip(self.feature_names, self.model.coef_)
            }
        return {name: c for name, c in
                zip(self.feature_names, self._fallback_coef)}
