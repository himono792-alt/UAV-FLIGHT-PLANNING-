"""
Bayesian Inference — Bayes' Rule + Sensor Fusion cho UAV
Lý thuyết: Chương 7 (Probability, Bayes' Rule, Uncertainty)

P(H|E) = P(E|H) * P(H) / P(E)

Ứng dụng: kết hợp GPS + LIDAR + Wind Sensor để tính xác suất
chướng ngại vật, độ chính xác vị trí, mức gió thực tế.
"""

from typing import Dict, Optional


class BayesianInference:
    """
    Bayes' Rule theo Ch7 — Sensor Fusion cho UAV.

    Sensor reliability (P(detect | obstacle)):
        GPS       : 0.95 (false positive rate 0.05)
        LIDAR     : 0.90 (false positive rate 0.10)
        wind_sensor: 0.85

    Quy ước:
        likelihood    = P(sensor detects | obstacle exists) = reliability
        false_pos_rate= P(sensor detects | no obstacle)    = 1 - reliability
    """

    # P(sensor=True | H=True) — true positive rate
    SENSOR_RELIABILITY: Dict[str, float] = {
        "gps":         0.95,
        "lidar":       0.90,
        "wind_sensor": 0.85,
        "camera":      0.88,
    }

    # P(sensor=True | H=False) — false positive rate
    SENSOR_FALSE_POS: Dict[str, float] = {
        "gps":         0.05,
        "lidar":       0.10,
        "wind_sensor": 0.15,
        "camera":      0.12,
    }

    def __init__(self, prior_obstacle: float = 0.1):
        """
        prior_obstacle: P(Obstacle) trước khi quan sát sensor
                        Thường ước từ obstacle density của grid
        """
        self.prior: Dict[str, float] = {
            "obstacle":   prior_obstacle,
            "wind_high":  0.2,
            "position_ok": 0.9,
        }

    # ------------------------------------------------------------------ #
    #  Bayes' Rule cơ bản                                                 #
    # ------------------------------------------------------------------ #

    def bayes_update(self, prior: float, likelihood: float,
                     evidence_prob: float) -> float:
        """
        P(H|E) = P(E|H) * P(H) / P(E)

        Args:
            prior         : P(H)         — xác suất prior
            likelihood    : P(E|H)       — xác suất sensor detect nếu H đúng
            evidence_prob : P(E)         — xác suất sensor detect tổng thể

        Returns: P(H|E) đã normalize
        """
        if evidence_prob <= 0:
            return prior
        posterior = (likelihood * prior) / evidence_prob
        return max(0.0, min(1.0, posterior))

    def update_belief(self, belief: float, observation: bool,
                      sensor_type: str) -> float:
        """
        Cập nhật belief P(Obstacle) sau 1 quan sát từ sensor.

        Áp dụng Bayes' Rule:
            P(O|detect)    = P(detect|O)    * P(O)    / P(detect)
            P(O|no_detect) = P(no_detect|O) * P(O)    / P(no_detect)

        P(detect) = P(detect|O)*P(O) + P(detect|¬O)*P(¬O)
        """
        tp = self.SENSOR_RELIABILITY.get(sensor_type, 0.85)   # true positive
        fp = self.SENSOR_FALSE_POS.get(sensor_type, 0.15)      # false positive

        if observation:
            # Sensor báo có obstacle
            p_e  = tp * belief + fp * (1 - belief)     # P(detect)
            p_e  = max(p_e, 1e-9)
            posterior = (tp * belief) / p_e
        else:
            # Sensor báo không có obstacle
            fn = 1 - tp                                  # false negative rate
            tn = 1 - fp                                  # true negative rate
            p_e = fn * belief + tn * (1 - belief)       # P(no_detect)
            p_e = max(p_e, 1e-9)
            posterior = (fn * belief) / p_e

        return max(0.0, min(1.0, posterior))

    # ------------------------------------------------------------------ #
    #  Sensor Fusion                                                      #
    # ------------------------------------------------------------------ #

    def sensor_fusion(self, sensors_data: Dict[str, bool],
                      prior_obstacle: Optional[float] = None) -> Dict[str, float]:
        """
        Kết hợp nhiều sensor bằng Bayes' Rule liên tiếp (sequential update).

        Thứ tự: GPS → LIDAR → wind_sensor → camera (nếu có)
        P(Obstacle | GPS, LIDAR) = Bayes cập nhật lần lượt

        Args:
            sensors_data: {"gps": True/False, "lidar": True/False, ...}
            prior_obstacle: ghi đè prior mặc định

        Returns: {
            "obstacle_prob":       float,   # P(Obstacle | all sensors)
            "position_confidence": float,   # P(GPS đúng)
            "wind_confidence":     float,   # P(wind sensor đúng)
            "alert_level":         str,     # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        }
        """
        belief = prior_obstacle if prior_obstacle is not None \
                 else self.prior["obstacle"]

        # Sequential Bayesian update theo thứ tự ưu tiên
        update_order = ["lidar", "camera", "gps", "wind_sensor"]

        for sensor in update_order:
            if sensor in sensors_data:
                belief = self.update_belief(belief, sensors_data[sensor], sensor)

        # Position confidence — chỉ GPS ảnh hưởng
        pos_conf = self.prior["position_ok"]
        if "gps" in sensors_data:
            pos_conf = self.update_belief(pos_conf, sensors_data["gps"], "gps")

        # Wind confidence
        wind_conf = self.prior["wind_high"]
        if "wind_sensor" in sensors_data:
            wind_conf = self.update_belief(wind_conf, sensors_data["wind_sensor"],
                                           "wind_sensor")

        alert = self._alert_level(belief)

        return {
            "obstacle_prob":       round(belief, 4),
            "position_confidence": round(pos_conf, 4),
            "wind_confidence":     round(wind_conf, 4),
            "alert_level":         alert,
        }

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _alert_level(prob: float) -> str:
        if prob < 0.2:   return "LOW"
        if prob < 0.5:   return "MEDIUM"
        if prob < 0.8:   return "HIGH"
        return "CRITICAL"

    def marginal_evidence(self, hypothesis_prior: float,
                          sensor_type: str, observation: bool) -> float:
        """P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H) — evidence probability"""
        tp = self.SENSOR_RELIABILITY.get(sensor_type, 0.85)
        fp = self.SENSOR_FALSE_POS.get(sensor_type, 0.15)
        if observation:
            return tp * hypothesis_prior + fp * (1 - hypothesis_prior)
        fn = 1 - tp
        tn = 1 - fp
        return fn * hypothesis_prior + tn * (1 - hypothesis_prior)

    def summary(self, result: dict) -> str:
        return (
            f"Sensor Fusion:\n"
            f"  P(Obstacle)         = {result['obstacle_prob']:.2%}\n"
            f"  Position confidence = {result['position_confidence']:.2%}\n"
            f"  Wind confidence     = {result['wind_confidence']:.2%}\n"
            f"  Alert level         = {result['alert_level']}"
        )
