"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import gym
import numpy as np

__author__ = "Islam Elnabarawy"


class DQNEnvWrapper(gym.Wrapper):

    def __init__(self, env):
        self.last_observation = self.observation(env.reset())
        super().__init__(env)
        self.observation_space.shape = (28 * 3,)
        self.action_space.n = 33

    def _reset(self, **kwargs):
        self.env.reset()
        obs, _, _, _ = self.env.step([1, 2, 3] * 3)
        obs = self.observation(obs)
        self.last_observation = obs
        return obs

    def _step(self, action):
        if action >= len(self.env.available_actions):
            return self.last_observation, -1000, True, {}
        action = self.action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        self.last_observation = obs
        return obs, reward, done, info

    def observation(self, obs):
        return np.concatenate([
            np.array(obs['occupied'], dtype=np.float32).reshape([28, 1]),
            np.array(obs['player_owned'], dtype=np.float32).reshape([28, 1]),
            np.array(obs['piece_type'], dtype=np.float32).reshape([28, 1]),
        ], axis=0).reshape(-1)

    def action(self, action):
        return self.env.available_actions[action]

    def reverse_action(self, action):
        return self.env.available_actions.index(action)

    def reward(self, reward):
        return reward[0] + reward[1]
