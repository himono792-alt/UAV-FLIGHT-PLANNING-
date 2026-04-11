# UAV Flight Planning Agent
**Đồ án Trí tuệ Nhân tạo — PTIT**

Tác tử thông minh lập kế hoạch đường bay UAV trong không gian 3D, tích hợp đầy đủ **9/9 chương lý thuyết TTNT**.

---

## Tính năng

| Module | Lý thuyết | Chức năng |
|--------|-----------|-----------|
| `environment` | Ch2, Ch3 | GridWorld3D, Obstacle management |
| `search` | Ch3, Ch4 | BFS, DFS, A*, Greedy BFS, Heuristics |
| `optimizer` | Ch4 | Simulated Annealing, Genetic Algorithm |
| `knowledge` | Ch5, Ch6 | KB-Agent TELL/ASK, Propositional Logic, FOL |
| `bayesian` | Ch7, Ch8 | Bayesian Network, Sensor Fusion, MEU |
| `learning` | Ch9 | Decision Tree, Regression, Q-Learning |
| `agent` | Ch2 | UAVAgent — tích hợp PERCEIVE→REASON→PLAN→ACT→LEARN |

## Cài đặt

```bash
pip install scikit-learn matplotlib numpy
```

## Chạy demo

```bash
python main.py
```

## Chạy tests

```bash
python -m pytest tests/ -v
```

Kết quả: **175/175 tests PASS**

## Kết quả thực nghiệm

- A* tìm đường optimal (cost 35.5) trong 34ms
- Q-Learning đạt cost = A* sau 800 episodes (ratio 1.00x)
- Decision Tree accuracy: **89.1%**
- Regression R²: **0.9927**

## Cấu trúc

```
src/
├── environment/    GridWorld3D (15×15×6), Obstacles
├── search/         A*, SA, GA, BFS, DFS, Heuristics
├── optimizer/      SA, GA
├── knowledge/      KB-Agent, FOL
├── bayesian/       BN, MEU, Sensor Fusion
├── learning/       DT, Regression, Q-Learning
└── agent/          UAVAgent (tích hợp)
tests/              175 unit tests
docs/               Báo cáo + Mapping lý thuyết
main.py             5 demo scenarios
```

## GitHub

Repository: [UAV-FLIGHT-PLANNING-](https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-)
