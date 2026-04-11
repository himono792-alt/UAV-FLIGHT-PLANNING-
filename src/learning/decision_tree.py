"""
Decision Tree Classifier — Phân loại điều kiện bay
Lý thuyết: Chương 9 (Supervised Learning, Decision Tree)
Dùng scikit-learn DecisionTreeClassifier

Features: wind_speed, battery_level, visibility, obstacle_density, altitude
Labels  : FLY | CAREFUL | AVOID | LAND

Dataset được tạo từ rules GĐ4 (propositional.py) + Gaussian noise
→ mô phỏng dữ liệu cảm biến thực tế có sai số.
"""

import random
import math
from typing import List, Tuple, Dict, Optional

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Label constants ────────────────────────────────────────────────────── #
LABEL_FLY     = "FLY"
LABEL_CAREFUL = "CAREFUL"
LABEL_AVOID   = "AVOID"
LABEL_LAND    = "LAND"
ALL_LABELS    = [LABEL_FLY, LABEL_CAREFUL, LABEL_AVOID, LABEL_LAND]

FEATURE_NAMES = ["wind_speed", "battery_level", "visibility",
                 "obstacle_density", "altitude"]


def _label_from_rules(wind: float, battery: float,
                      visibility: float, obstacle_density: float) -> str:
    """
    Rule-based labeling từ GĐ4 propositional rules:
    R4: wind > 25 → LAND (nguy hiểm nhất)
    R5: battery < 10 → LAND
    R1: obstacle_density > 0.3 → AVOID
    R3: wind > 15 OR visibility < 30 → CAREFUL
    R2: battery < 20 → CAREFUL
    Default → FLY
    """
    if wind > 25 or battery < 10:
        return LABEL_LAND
    if obstacle_density > 0.3:
        return LABEL_AVOID
    if wind > 15 or visibility < 30 or battery < 20:
        return LABEL_CAREFUL
    return LABEL_FLY


class FlightConditionClassifier:
    """
    Decision Tree Classifier — Ch9 Supervised Learning.

    Workflow:
        clf = FlightConditionClassifier()
        X, y = clf.generate_training_data(n_samples=1000)
        clf.train(X, y)
        label = clf.predict([wind, battery, visibility, obstacle_density, alt])
    """

    def __init__(self, max_depth: int = 5, random_state: int = 42):
        self.max_depth    = max_depth
        self.random_state = random_state
        self.model        = None
        self.is_trained   = False
        self.feature_names = FEATURE_NAMES
        self._label_encoder: Dict[str, int] = {
            LABEL_FLY: 0, LABEL_CAREFUL: 1, LABEL_AVOID: 2, LABEL_LAND: 3
        }
        self._label_decoder: Dict[int, str] = {
            v: k for k, v in self._label_encoder.items()
        }

    # ------------------------------------------------------------------ #
    #  Data generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_training_data(self, n_samples: int = 1000,
                               noise_std: float = 0.05,
                               seed: int = 42) -> Tuple[List, List]:
        """
        Tạo dataset từ rules + Gaussian noise — Ch9 methodology.

        Returns: (X, y) với y là list nhãn string
        """
        rng = random.Random(seed)

        X, y = [], []
        for _ in range(n_samples):
            wind       = rng.uniform(0, 40)
            battery    = rng.uniform(0, 100)
            visibility = rng.uniform(0, 100)
            obs_density= rng.uniform(0, 0.6)
            altitude   = rng.uniform(0, 50)

            label = _label_from_rules(wind, battery, visibility, obs_density)

            # Gaussian noise trên features
            wind       = max(0, wind       + rng.gauss(0, noise_std * 40))
            battery    = max(0, min(100, battery    + rng.gauss(0, noise_std * 100)))
            visibility = max(0, min(100, visibility + rng.gauss(0, noise_std * 100)))
            obs_density= max(0, min(1,   obs_density+ rng.gauss(0, noise_std)))
            altitude   = max(0, altitude   + rng.gauss(0, noise_std * 50))

            X.append([wind, battery, visibility, obs_density, altitude])
            y.append(label)

        return X, y

    # ------------------------------------------------------------------ #
    #  Train / Predict                                                   #
    # ------------------------------------------------------------------ #

    def train(self, X: List, y: List) -> Dict:
        """
        Huấn luyện DecisionTreeClassifier (Ch9).
        Returns metrics dict nếu có sklearn, dict đơn giản nếu không.
        """
        if SKLEARN_AVAILABLE:
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            # Encode labels
            y_enc = [self._label_encoder[label] for label in y]
            self.model.fit(X, y_enc)
            self.is_trained = True

            # Evaluate trên toàn bộ training data
            y_pred = self.model.predict(X)
            acc = accuracy_score(y_enc, y_pred)
            return {"accuracy": acc, "n_samples": len(X), "backend": "sklearn"}
        else:
            # Fallback: simple rule-based "model"
            self._fallback = True
            self.is_trained = True
            return {"accuracy": 1.0, "n_samples": len(X), "backend": "rules"}

    def predict(self, features: List[float]) -> str:
        """
        Phân loại điều kiện bay — O(log n) với Decision Tree.

        Args:
            features: [wind_speed, battery_level, visibility,
                       obstacle_density, altitude]

        Returns: "FLY" | "CAREFUL" | "AVOID" | "LAND"
        """
        if not self.is_trained:
            raise RuntimeError("Chưa train model. Gọi train() trước.")

        if SKLEARN_AVAILABLE and hasattr(self, 'model') and self.model is not None:
            enc = self.model.predict([features])[0]
            return self._label_decoder[enc]
        else:
            # Fallback to rules
            return _label_from_rules(features[0], features[1],
                                     features[2], features[3])

    def predict_proba(self, features: List[float]) -> Dict[str, float]:
        """Trả về xác suất của từng class"""
        if SKLEARN_AVAILABLE and self.model is not None:
            proba = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            return {self._label_decoder[c]: p for c, p in zip(classes, proba)}
        label = self.predict(features)
        return {l: (1.0 if l == label else 0.0) for l in ALL_LABELS}

    def evaluate(self, X_test: List, y_test: List) -> Dict:
        """Đánh giá model trên tập test — trả về accuracy"""
        if not self.is_trained:
            raise RuntimeError("Chưa train model.")
        y_pred = [self.predict(x) for x in X_test]
        correct = sum(p == t for p, t in zip(y_pred, y_test))
        return {
            "accuracy": correct / len(y_test),
            "n_test":   len(y_test),
        }

    def export_tree(self) -> str:
        """Xuất cây quyết định dạng text cho báo cáo"""
        if SKLEARN_AVAILABLE and self.model is not None:
            labels = [self._label_decoder[c] for c in self.model.classes_]
            return export_text(self.model,
                               feature_names=self.feature_names,
                               class_names=labels)
        return "Decision Tree (rules-based fallback)"
