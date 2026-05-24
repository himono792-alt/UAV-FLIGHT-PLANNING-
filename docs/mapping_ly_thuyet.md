# BẢNG MAPPING: LÝ THUYẾT → CODE

## UAV Flight Planning Agent — TTNT (PTIT)

| Chương | Khái niệm lý thuyết | Module | File chính | Class / Function |
|--------|---------------------|--------|-----------|-----------------|
| Ch2 | PEAS Framework | environment | `grid_world.py` | `GridWorld3D` |
| Ch2 | Learning Agent (Critic, Learning Elem.) | agent | `uav_agent.py` | `UAVAgent` |
| Ch2 | Agent loop: Perceive→Reason→Plan→Act→Learn | agent | `uav_agent.py` | `run_mission()` |
| Ch3 | State Space (state, actions, transition) | environment | `grid_world.py` | `get_neighbors()`, `step_cost()` |
| Ch3 | Graph-Search (BFS, DFS) | search | `graph_search.py` | `graph_search()`, `bfs()`, `dfs()` |
| Ch4 | A* Search: f(n)=g(n)+h(n) | search | `a_star.py` | `a_star_search()` |
| Ch4 | Admissible Heuristics | search | `heuristics.py` | `euclidean_3d`, `manhattan_3d`, `energy_based` |
| Ch4 | Greedy Best-First Search | search | `greedy_bfs.py` | `greedy_bfs()` |
| Ch4 | Simulated Annealing: T*=alpha | optimizer | `simulated_annealing.py` | `SimulatedAnnealing.optimize()` |
| Ch4 | Genetic Algorithm: selection/crossover/mutation | optimizer | `genetic_algorithm.py` | `GeneticAlgorithm.evolve()` |
| Ch5 | Knowledge-Based Agent: TELL/ASK/RETRACT | knowledge | `kb_agent.py` | `KnowledgeBase`, `KBAgent` |
| Ch5 | Propositional Logic: forward chaining | knowledge | `propositional.py` | `evaluate_rules()`, `SAFETY_RULES` |
| Ch6 | First-Order Logic: predicates, quantifiers | knowledge | `fol_rules.py` | `FOLKnowledgeBase` |
| Ch6 | FOL Rules: ∀x Safe(x)↔¬Obstacle(x) | knowledge | `fol_rules.py` | `build_from_grid()`, `query_safe()` |
| Ch6 | Markov Blanket | bayesian | `bayesian_network.py` | `markov_blanket()` |
| Ch7 | Bayes' Rule: P(H\|E)=P(E\|H)P(H)/P(E) | bayesian | `inference.py` | `BayesianInference.bayes_update()` |
| Ch7 | Sensor Fusion (sequential Bayes update) | bayesian | `inference.py` | `sensor_fusion()` |
| Ch7 | MEU: argmax_a EU(a\|evidence) | bayesian | `meu_decision.py` | `MEUDecisionMaker.decide()` |
| Ch8 | Bayesian Network: DAG + CPT | bayesian | `bayesian_network.py` | `BayesianNetwork`, `BayesianNode` |
| Ch8 | Global Semantics: P(x₁..xₙ)=ΠP(xᵢ\|parents) | bayesian | `bayesian_network.py` | `_enumerate_all()` |
| Ch8 | Variable Enumeration (exact inference) | bayesian | `bayesian_network.py` | `query()` |
| Ch9 | Decision Tree: supervised learning | learning | `decision_tree.py` | `FlightConditionClassifier` |
| Ch9 | Linear Regression | learning | `regression.py` | `FlightTimePredictor` |
| Ch9 | Q-Learning: Q(s,a)←Q+α[R+γmaxQ'-Q] | learning | `reinforcement.py` | `QLearningAgent` |
| Ch9 | Epsilon-greedy exploration | learning | `reinforcement.py` | `choose_action()` |
| Ch9 | Reward signal (Critic — Ch2) | learning | `reinforcement.py` | `_get_reward()` |

---

## Cấu trúc thư mục

```
uav-flight-planning/
├── src/
│   ├── environment/
│   │   ├── grid_world.py         ← Ch2 PEAS, Ch3 State Space
│   │   ├── obstacles.py          ← Ch2 Environment
│   │   ├── visualizer.py         ← Demo visualization
│   │   └── city_grid.py          ← HCM 3D grid từ dữ liệu địa lý
│   ├── search/
│   │   ├── graph_search.py       ← Ch3 BFS/DFS
│   │   ├── heuristics.py         ← Ch4 Admissible heuristics
│   │   ├── a_star.py             ← Ch4 A* Search
│   │   └── greedy_bfs.py         ← Ch4 Greedy BFS
│   ├── optimizer/
│   │   ├── simulated_annealing.py← Ch4 SA Local Search
│   │   └── genetic_algorithm.py  ← Ch4 GA Population Search
│   ├── knowledge/
│   │   ├── kb_agent.py           ← Ch5 KB-Agent TELL/ASK
│   │   ├── propositional.py      ← Ch5 Propositional Logic
│   │   └── fol_rules.py          ← Ch6 First-Order Logic
│   ├── bayesian/
│   │   ├── inference.py          ← Ch7 Bayes' Rule, Sensor Fusion
│   │   ├── bayesian_network.py   ← Ch8 BN, CPT, Enumeration
│   │   └── meu_decision.py       ← Ch7 MEU Decision Making
│   ├── learning/
│   │   ├── decision_tree.py      ← Ch9 Supervised Learning
│   │   ├── regression.py         ← Ch9 Linear Regression
│   │   └── reinforcement.py      ← Ch9 Q-Learning
│   ├── geodata/
│   │   └── osm_loader.py         ← OpenStreetMap/Overpass loader
│   └── agent/
│       └── uav_agent.py          ← Ch2 Learning Agent tích hợp
├── tests/
│   ├── test_environment.py       ← 21 tests môi trường 3D
│   ├── test_city_geodata.py      ← 7 tests HCM geodata/city grid
│   ├── test_search.py            ← 33 tests tìm kiếm
│   ├── test_optimizer.py         ← 20 tests tối ưu
│   ├── test_knowledge.py         ← 38 tests tri thức/logic
│   ├── test_bayesian.py          ← 29 tests xác suất/MEU
│   ├── test_learning.py          ← 28 tests học máy
│   └── test_integration.py       ← 24 tests tích hợp agent
├── docs/
│   ├── bao_cao.pdf               ← Báo cáo đồ án
│   ├── demo_guide.md             ← Hướng dẫn 4 demo
│   ├── mapping_ly_thuyet.md      ← File này
│   └── phan_cong.pdf             ← Bảng phân công nhiệm vụ
├── configs/
│   ├── default_config.yaml        ← Cấu hình mặc định
│   └── hcm_central.yaml          ← Bbox/cell size cho HCM 3D
├── tools/
│   └── enhance_hcm_building_detail.py ← Build/enhance HCM demo
├── outputs/
│   └── hcm_uav_path.geojson      ← Quỹ đạo UAV mẫu
├── main.py                       ← Entry point, 5 demo scenarios
├── demo_visualize.py             ← Demo matplotlib 3D
├── demo_3d.html                  ← Demo web 3D abstract
├── demo_hcm_3d.py                ← Static server cho HCM demo
└── requirements.txt
```

---

## Tổng kết test

| Module | Tests | Status |
|--------|-------|--------|
| environment | 21 | ✅ PASS |
| city_geodata | 7 | ✅ PASS |
| search | 33 | ✅ PASS |
| optimizer | 20 | ✅ PASS |
| knowledge | 38 | ✅ PASS |
| bayesian | 29 | ✅ PASS |
| learning | 28 | ✅ PASS |
| integration | 24 | ✅ PASS |
| **TỔNG** | **200** | **✅ 200/200 PASS** |
