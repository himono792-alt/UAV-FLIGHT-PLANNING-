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
│   │   └── visualizer.py         ← Demo visualization
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
│   └── agent/
│       └── uav_agent.py          ← Ch2 Learning Agent (tích hợp)
├── tests/
│   ├── test_environment.py       ← 25 tests GĐ1
│   ├── test_search.py            ← 18 tests GĐ2
│   ├── test_optimizer.py         ← 18 tests GĐ3
│   ├── test_knowledge.py         ← 28 tests GĐ4
│   ├── test_bayesian.py          ← 38 tests GĐ5
│   ├── test_learning.py          ← 26 tests GĐ6
│   └── test_integration.py       ← 22 tests GĐ7
├── docs/
│   ├── bao_cao.pdf               ← Báo cáo đồ án
│   ├── demo_guide.md             ← Hướng dẫn 4 demo
│   └── mapping_ly_thuyet.md      ← File này
├── main.py                       ← Entry point, 5 demo scenarios
└── requirements.txt
```

---

## Tổng kết test

| Module | Tests | Status |
|--------|-------|--------|
| environment | 25 | ✅ PASS |
| search | 18 | ✅ PASS |
| optimizer | 18 | ✅ PASS |
| knowledge | 28 | ✅ PASS |
| bayesian | 38 | ✅ PASS |
| learning | 26 | ✅ PASS |
| integration | 22 | ✅ PASS |
| **TỔNG** | **175** | **✅ 175/175 PASS** |
