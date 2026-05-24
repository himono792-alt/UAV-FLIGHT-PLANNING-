# UAV Flight Planning Agent

**Đồ án Trí tuệ Nhân tạo — PTIT**

Tác tử thông minh lập kế hoạch đường bay UAV trong không gian 3D, tích hợp đầy đủ **9/9 chương lý thuyết TTNT**.

![Tests](https://img.shields.io/badge/tests-193%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Release](https://img.shields.io/badge/release-v1.0--hcm--demo-blue)](https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-/releases/tag/v1.0-hcm-demo)

---

## 🚀 Live Demo

| Demo | Mô tả | Cách chạy |
|---|---|---|
| 🌐 **3D Web (abstract)** | Mô phỏng UAV bay 3D trên grid 12×12×5, UI sci-fi xanh dương | **[Mở ngay](https://himono792-alt.github.io/UAV-FLIGHT-PLANNING-/demo_3d.html)** (Three.js, không cần cài đặt) |
| 🗺️ **HCM 3D Map** | Bản đồ Quận 1 + Thủ Đức từ OSM, building LOD1, 3 tier thuật toán | **[Download HTML](https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-/releases/tag/v1.0-hcm-demo)** rồi mở local |
| 📊 **Visualize Local** | 4 scenario A* tìm đường, render matplotlib 3D xoay được | `python demo_visualize.py` |
| 💻 **CLI Demo** | 5 kịch bản agent đầy đủ (gió, pin yếu, obstacle, replan) | `python main.py` |

📖 **[Hướng dẫn chi tiết các demo →](docs/demo_guide.md)**

---

## Quick Start

```bash
git clone https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-.git
cd UAV-FLIGHT-PLANNING-
pip install -r requirements.txt
python main.py
```

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

## Chạy demo

```bash
python main.py
```

Output mẫu (5 kịch bản thử nghiệm):

```
============================================================
  UAV FLIGHT PLANNING AGENT — Demo
  Đồ án TTNT — PTIT
============================================================

  ✓ Scenario 1: Bay bình thường (gió 5 m/s, pin 100%)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=46.6ms

  ✓ Scenario 2: Nhiều chướng ngại vật (25 obstacles)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=47.6ms

  ✓ Scenario 3: Gió mạnh (22 m/s, BN+MEU quyết định)
     [SUCCESS] reached=True | steps=1 | battery_used=23.6% | replans=1 | time=47.2ms

  ✗ Scenario 4: Pin yếu (18%), KB→RETURN_HOME
     [FAILED] reached=False | steps=1 | battery_used=18.0% | replans=1 | time=44.6ms

  ✓ Scenario 5: Dynamic re-planning (obstacle giữa đường)
     [SUCCESS] reached=True | steps=1 | battery_used=18.9% | replans=1 | time=38.1ms

============================================================
  Kết quả: 4/5 scenarios thành công
============================================================
```

> Lưu ý: Scenario 4 báo "FAILED" là **hành vi đúng** — pin yếu nên KB-Agent ra quyết định quay về thay vì cố bay tiếp đến goal.

## Chạy tests

```bash
python -m pytest tests/ -v
```

Kết quả: **193/193 tests PASS** trong khoảng 8 giây.

Test bao phủ đầy đủ các module: `test_bayesian`, `test_environment`, `test_integration`, `test_knowledge`, `test_learning`, `test_optimizer`, `test_search`.

## Kết quả thực nghiệm

- A* tìm đường optimal (cost 35.5) trong 34ms
- Q-Learning đạt cost = A* sau 800 episodes (ratio 1.00x)
- Decision Tree accuracy: **89.1%**
- Regression R²: **0.9927**

## Cấu trúc dự án

```
UAV-FLIGHT-PLANNING/
├── src/
│   ├── environment/        GridWorld3D, Obstacles, Visualizer, CityGrid3D
│   ├── search/             A*, BFS, DFS, Greedy BFS, Heuristics
│   ├── optimizer/          Simulated Annealing, Genetic Algorithm
│   ├── knowledge/          KB-Agent, Propositional, FOL
│   ├── bayesian/           Bayesian Network, MEU, Inference
│   ├── learning/           Decision Tree, Regression, Q-Learning
│   ├── geodata/            OSM Loader (Overpass → GridWorld)
│   └── agent/              UAVAgent (tích hợp đầy đủ pipeline)
├── tests/                  193 unit tests + integration tests
├── docs/
│   ├── bao_cao.md              Báo cáo đồ án (theo mẫu PTIT)
│   ├── mapping_ly_thuyet.md    Mapping với chương trình PTIT
│   └── demo_guide.md           Hướng dẫn chi tiết 4 demo
├── configs/
│   ├── default_config.yaml     Cấu hình mặc định cho main.py
│   └── hcm_central.yaml        Bbox + cell size cho HCM 3D
├── tools/
│   └── enhance_hcm_building_detail.py   Nâng chi tiết building từ OSM
├── outputs/
│   └── hcm_uav_path.geojson    Xuất quỹ đạo UAV ra GeoJSON
├── main.py                 Entry point — 5 demo scenarios CLI
├── demo_visualize.py       Demo matplotlib 3D — 4 scenarios A*
├── demo_3d.html            Demo 3D web abstract (GitHub Pages)
├── demo_hcm_3d.py          Launcher static server cho demo HCM
└── requirements.txt        Dependencies
```

> 💡 **File `hcm_3d_demo.html` (12 MB)** không nằm trong repo — tải từ [Release v1.0-hcm-demo](https://github.com/himono792-alt/UAV-FLIGHT-PLANNING-/releases/tag/v1.0-hcm-demo), đặt vào `outputs/` rồi chạy `python demo_hcm_3d.py`.

## Yêu cầu môi trường

- Python 3.10+
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- scikit-learn ≥ 1.3
- pytest ≥ 7.4

## License

MIT License — xem file [LICENSE](LICENSE) để biết chi tiết.
