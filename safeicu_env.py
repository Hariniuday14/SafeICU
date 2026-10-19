
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SafeICURLEnv(gym.Env):
    """Synthetic ICU environment for educational RL experiments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario="normal",
        safety_enabled=True,
        horizon=24,
    ):
        super().__init__()

        if scenario not in {"normal", "surge", "shortage"}:
            raise ValueError(f"Unknown scenario: {scenario}")

        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        self.scenario = scenario
        self.safety_enabled = safety_enabled
        self.horizon = horizon

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        self.capacity = {
            "normal": (10, 8, 6),
            "surge": (10, 8, 6),
            "shortage": (5, 4, 3),
        }

        self.arrival_rates = {
            "normal": 1.0,
            "surge": 2.2,
            "shortage": 1.5,
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.hour = 0
        self.waiting = [0, 0, 0]
        self.active = []

        self.beds, self.monitors, self.staff = (
            self.capacity[self.scenario]
        )

        self.total_reward = 0.0
        self.violations = 0
        self.interventions = 0
        self.admissions = 0
        self.critical_wait_total = 0

        self._generate_patients()
        return self._get_obs(), {}

    def _generate_patients(self):
        count = self.np_random.poisson(
            self.arrival_rates[self.scenario]
        )

        for _ in range(count):
            category = int(
                self.np_random.choice(
                    3, p=[0.20, 0.35, 0.45]
                )
            )
            self.waiting[category] += 1

    def _available(self):
        used = len(self.active)
        return (
            used < self.beds
            and used < self.monitors
            and used < self.staff
        )

    def _get_obs(self):
        used = len(self.active)

        return np.array(
            [
                min(self.waiting[0] / 20, 1),
                min(self.waiting[1] / 20, 1),
                min(self.waiting[2] / 20, 1),
                (self.beds - used) / self.beds,
                (self.monitors - used) / self.monitors,
                (self.staff - used) / self.staff,
                self.hour / self.horizon,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        if self.hour >= self.horizon:
            raise RuntimeError("Episode ended. Call reset().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        action = int(action)

        self.active = [
            duration - 1
            for duration in self.active
            if duration - 1 > 0
        ]

        self._generate_patients()

        reward = 0.0
        intervention = False
        proposed_action = action
        executed_action = action

        if self.safety_enabled and self._available():
            if self.waiting[0] > 0 and action != 1:
                executed_action = 1
                intervention = True

        if intervention:
            self.interventions += 1

        if executed_action > 0:
            category = executed_action - 1

            if self.waiting[category] > 0 and self._available():
                self.waiting[category] -= 1
                duration = int(self.np_random.integers(2, 7))
                self.active.append(duration)
                self.admissions += 1
                reward += [10, 4, 1][category]
            else:
                reward -= 1

        reward -= 2 * self.waiting[0]
        reward -= 0.5 * self.waiting[1]
        self.critical_wait_total += self.waiting[0]

        self.total_reward += reward
        self.hour += 1

        terminated = False
        truncated = self.hour >= self.horizon

        info = {
            "proposed_action": proposed_action,
            "executed_action": executed_action,
            "shield_intervention": intervention,
            "admissions": self.admissions,
            "critical_wait_total": self.critical_wait_total,
            "total_reward": self.total_reward,
            "executed_violations": self.violations,
        }

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )
