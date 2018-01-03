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
# noinspection PyUnresolvedReferences
import rps3env
from baselines import deepq

from rps3agents.baselines_dqn.wrapper import DQNEnvWrapper

__author__ = "Islam Elnabarawy"


def main():
    env = DQNEnvWrapper(gym.make('RPS3Game-v0'))
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
