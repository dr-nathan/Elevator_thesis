import logging
import os
from datetime import datetime
from pathlib import Path

import cma
import torch

import config
from agent.continuous_agents.RLAgents_cont import RLAgent
from environment.environment_continuous import DiscreteEvent

file_loc = Path(__file__).parent / 'data' / 'log.log'
logging.basicConfig(filename=file_loc, filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def train_agent(parameters):

    n_elev = 6
    n_floors = 17

    env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors)
    train_env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors)
    val_env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')

    agent = RLAgent(env=env,
                    train_env=train_env,
                    val_env=val_env,
                    training=True,
                    nn_type_assign=config.NN_TYPE_ASSIGN,
                    device=torch.device('cpu'),
                    n_episodes=config.N_EPISODES,
                    buffer_size=10000,
                    lr=config.LR,
                    epsilon_decay=True,
                    epsilon_start=1,
                    epsilon_end=0.1)

    # create save map for agents, might as well keep the trained files obtained during tuning
    path = Path(__file__).parent / 'agent' / 'continuous_agents' / 'data' / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(path):
        os.makedirs(path)

    # train agent
    agent.training_loop(batch_size=32, save_path=path)

    # do one full validation data run
    state, info = val_env.reset()
    terminated = False
    while not terminated:
        action = agent.act(state, info_=info)
        state, reward, terminated, info = val_env.step(action)

    return sum(val_env.episode_data.rewards)


es = cma.CMAEvolutionStrategy(4 * [1], 1, {'popsize': 4, 'timeout': "2 * 60**2"})

es.optimize(train_agent, verb_disp=1)

res = es.result

print(res)

