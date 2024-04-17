import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from agent.continuous_agents.RLAgents_cont import RLAgent
from agent.continuous_agents.RuleBasedAgents import (LeastBusyAgent, ETDAgent)
from environment.environment_continuous import DiscreteEvent

file_loc = Path(__file__).parent / 'log.log'
logging.basicConfig(filename=file_loc, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

n_elev = 6
n_floors = 17
env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors)

tot_reward = defaultdict(list)
tot_wait_time = defaultdict(list)

agent_loc = 'comparing/agents_to_compare_cont/2024-04-02_22-18-01'

for agent_type in [RLAgent, LeastBusyAgent, ETDAgent]:

    if agent_type == RLAgent:

        with open(agent_loc + '/config.txt', 'r') as f:
            # load config file
            configfile = json.load(f)
            config.__dict__.update(configfile)

        agent = agent_type(env=env, nn_type_assign=config.NN_TYPE_ASSIGN, nn_type_zone=config.NN_TYPE_ZONE,
                           training=False)

        if config.LEARN_ZONING:
            agent.load(agent_loc + '/online_network_assign.pt')
            agent.load(agent_loc + '/online_network_zone.pt', load_zone=True)
        else:
            agent.load(agent_loc + '/online_network_assign.pt')
    else:
        agent = agent_type(env)

    tqdm.write(f'\n Running {agent_type} agent...')

    for run in tqdm(range(config.RUNS)):

        state, info = env.reset()

        terminated = False
        while not terminated:
            action = agent.act(state, env, info)
            state, reward, terminated, info = env.step(action)

        tot_reward[agent_type.__name__].append(sum(env.episode_data.rewards))
        tot_wait_time[agent_type.__name__].append(np.mean(env.episode_data.passenger_wait_times))

# plot reward per agent
plt.boxplot(tot_reward.values(), labels=tot_reward.keys())
plt.title(f'Reward per agent. {n_elev} elevators, {n_floors} floors')
plt.xlabel('agent')
plt.ylabel('Total reward')
plt.show()

# plot wait time per agent
plt.boxplot(tot_wait_time.values(), labels=tot_wait_time.keys())
plt.title(f'Avg wait time per agent, {n_elev} elevators, {n_floors} floors')
plt.xlabel('agent')
plt.ylabel('Avg wait time')
plt.show()
