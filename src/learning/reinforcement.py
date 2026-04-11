"""
Q-Learning Agent — UAV tự học tìm đường tối ưu
Lý thuyết: Chương 9 (Reinforcement Learning, Q-Learning)
           Chương 2 (Learning Agent: Critic, Learning Element, Problem Generator)

Q-Learning update rule (Ch9):
    Q(s, a) ← Q(s, a) + α [R + γ max_a' Q(s', a') − Q(s, a)]

State : (x, y, z) — vị trí rời rạc trên GridWorld3D
Action: 6 hướng (+x/-x, +y/-y, +z/-z)

Reward:
    +100  đến đích
    -1000 va chạm obstacle
    -500  vào no-fly zone
    -1    mỗi bước (khuyến khích path ngắn)
    -5    mỗi bước leo cao (tốn năng lượng)
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from src.environment.grid_world import CellType


# ── Action constants ───────────────────────────────────────────────────── #
ACTIONS = [
    (+1, 0, 0), (-1, 0, 0),
    (0, +1, 0), (0, -1, 0),
    (0, 0, +1), (0, 0, -1),
]

# ── Reward values ──────────────────────────────────────────────────────── #
R_GOAL      = +100
R_OBSTACLE  = -1000
R_NFZ       = -500
R_STEP      = -1
R_CLIMB     = -5     # thêm khi z tăng
MAX_STEPS   = 1000   # giới hạn steps/episode tránh infinite loop


class QLearningAgent:
    """
    Q-Learning Agent theo Ch9.
    Cũng thể hiện Learning Agent architecture (Ch2):
        - Critic    : reward function
        - Learning Element : Q-update rule
        - Problem Generator: epsilon-greedy exploration
        - Performance Element: exploit Q-table
    """

    def __init__(self, grid,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 seed: int = None):
        self.grid          = grid
        self.alpha         = alpha       # learning rate
        self.gamma         = gamma       # discount factor
        self.epsilon       = epsilon     # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple, float] = {}   # {(state, action_idx): Q}

        # Training stats
        self.episode_rewards: List[float] = []
        self.episode_steps:   List[int]   = []
        self.epsilon_history: List[float] = []

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------ #
    #  Q-table helpers                                                    #
    # ------------------------------------------------------------------ #

    def _q(self, state: tuple, action_idx: int) -> float:
        """Lấy Q(s, a), mặc định 0.0"""
        return self.q_table.get((state, action_idx), 0.0)

    def _best_action(self, state: tuple) -> Tuple[int, float]:
        """Trả về (best_action_idx, max_Q) từ Q-table"""
        qs = [self._q(state, i) for i in range(len(ACTIONS))]
        best_idx = max(range(len(ACTIONS)), key=lambda i: qs[i])
        return best_idx, qs[best_idx]

    # ------------------------------------------------------------------ #
    #  Epsilon-greedy (Problem Generator — Ch2)                          #
    # ------------------------------------------------------------------ #

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy — Ch9:
        - Với P = epsilon: chọn ngẫu nhiên (explore)
        - Với P = 1-epsilon: chọn argmax Q(s,a) (exploit)
        """
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        return self._best_action(state)[0]

    # ------------------------------------------------------------------ #
    #  Q-update (Learning Element — Ch2)                                 #
    # ------------------------------------------------------------------ #

    def update(self, state: tuple, action_idx: int,
               reward: float, next_state: tuple):
        """
        Q-Learning update rule (Ch9):
        Q(s,a) ← Q(s,a) + α [R + γ max_a' Q(s',a') − Q(s,a)]
        """
        _, max_next_q = self._best_action(next_state)
        current_q = self._q(state, action_idx)
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[(state, action_idx)] = new_q

    # ------------------------------------------------------------------ #
    #  Reward function (Critic — Ch2)                                    #
    # ------------------------------------------------------------------ #

    def _get_reward(self, state: tuple, prev_state: tuple) -> float:
        """
        Reward theo Ch9 reward signal:
        +100 đích, -1000 obstacle, -500 NFZ, -1 mỗi bước, -5 leo cao
        """
        cell = self.grid.get_cell(state)
        if cell == CellType.OBSTACLE:
            return R_OBSTACLE
        if cell == CellType.NO_FLY_ZONE:
            return R_NFZ
        reward = R_STEP
        # Phạt thêm nếu leo cao (z tăng)
        if state[2] > prev_state[2]:
            reward += R_CLIMB
        return reward

    def _apply_action(self, state: tuple, action_idx: int) -> tuple:
        """Áp dụng action, clamp trong bounds grid"""
        dx, dy, dz = ACTIONS[action_idx]
        nx = max(0, min(self.grid.width  - 1, state[0] + dx))
        ny = max(0, min(self.grid.height - 1, state[1] + dy))
        nz = max(0, min(self.grid.depth  - 1, state[2] + dz))
        return (nx, ny, nz)

    # ------------------------------------------------------------------ #
    #  Training loop (Ch9)                                               #
    # ------------------------------------------------------------------ #

    def train(self, start: tuple, goal: tuple,
              episodes: int = 500) -> Dict:
        """
        Training loop Q-Learning:
        for ep in range(episodes):
            state = start
            while state != goal and steps < MAX_STEPS:
                action = choose_action(state)  # ε-greedy
                next_state = apply(action)
                reward = critic(next_state)
                update(state, action, reward, next_state)
                state = next_state
            epsilon *= decay
        """
        t0 = time.perf_counter()
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.epsilon_history.clear()

        goal_reached_count = 0

        for ep in range(episodes):
            state       = start
            total_reward = 0.0
            steps        = 0

            while state != goal and steps < MAX_STEPS:
                action_idx = self.choose_action(state)
                next_state = self._apply_action(state, action_idx)

                # Goal reward
                if next_state == goal:
                    reward = R_GOAL
                else:
                    reward = self._get_reward(next_state, state)

                self.update(state, action_idx, reward, next_state)

                total_reward += reward
                state = next_state
                steps += 1

                if state == goal:
                    goal_reached_count += 1
                    break

                # Dừng sớm nếu rơi vào obstacle
                if reward <= R_OBSTACLE:
                    break

            # Epsilon decay
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

        return {
            "episodes":      episodes,
            "goal_reached":  goal_reached_count,
            "success_rate":  goal_reached_count / episodes,
            "time_ms":       (time.perf_counter() - t0) * 1000,
            "q_table_size":  len(self.q_table),
            "final_epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------ #
    #  Path extraction (Performance Element — Ch2)                       #
    # ------------------------------------------------------------------ #

    def get_path(self, start: tuple, goal: tuple,
                 max_steps: int = MAX_STEPS) -> Optional[List[tuple]]:
        """
        Greedy extract path từ Q-table đã train.
        Dùng argmax Q(s,a) — không explore.
        """
        state  = start
        path   = [state]
        visited = {state}

        for _ in range(max_steps):
            if state == goal:
                return path

            action_idx, _ = self._best_action(state)
            next_state    = self._apply_action(state, action_idx)

            # Tránh vòng lặp
            if next_state in visited:
                # Thử action khác
                qs = [(self._q(state, i), i) for i in range(len(ACTIONS))]
                qs.sort(reverse=True)
                moved = False
                for _, idx in qs[1:]:
                    ns = self._apply_action(state, idx)
                    if ns not in visited:
                        next_state = ns
                        moved = True
                        break
                if not moved:
                    break

            path.append(next_state)
            visited.add(next_state)
            state = next_state

        return path if path[-1] == goal else None

    # ------------------------------------------------------------------ #
    #  Stats                                                             #
    # ------------------------------------------------------------------ #

    def get_training_stats(self) -> Dict:
        """Trả về thống kê training — dùng cho báo cáo / visualization"""
        if not self.episode_rewards:
            return {}

        n = len(self.episode_rewards)
        # Moving average reward (window=50)
        window = min(50, n)
        recent_avg = sum(self.episode_rewards[-window:]) / window

        return {
            "n_episodes":    n,
            "total_reward":  sum(self.episode_rewards),
            "avg_reward":    sum(self.episode_rewards) / n,
            "recent_avg":    recent_avg,
            "min_steps":     min(self.episode_steps) if self.episode_steps else 0,
            "avg_steps":     sum(self.episode_steps) / n,
            "q_table_size":  len(self.q_table),
            "final_epsilon": self.epsilon,
        }

    def reward_trend(self, window: int = 50) -> List[float]:
        """Moving average của rewards — cho thấy convergence"""
        rewards = self.episode_rewards
        if not rewards:
            return []
        result = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            result.append(sum(rewards[start:i + 1]) / (i - start + 1))
        return result
