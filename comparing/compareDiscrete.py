import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from agent.discrete_agents.RLAgents_disc import DQNAgent
from agent.discrete_agents.RLagents_multiagents import DDQNAgentMulti
from environment.environment_discrete import DiscreteNocam
from environment.helper_functions import fetch_agent

n_elev = 2
n_floors = 5
env = DiscreteNocam(n_elev=n_elev, n_floors=n_floors)
config.RENDER = False
config.RUN_TYPE = 'multiple_runs'
config.RUNS = 250

# whether to compare with rule based agents as well
run_rule_based = False

# to keep track of total reward per run
tot_reward = defaultdict(list)
tot_wait_time = defaultdict(list)

# run all agents in agents_to_compare map
for filepath in os.listdir(Path(__file__).parent / 'agents_to_compare'):

    if filepath.startswith('multi'):
        agent = DDQNAgentMulti(_env=env)
    elif filepath.startswith('single'):
        agent = DQNAgent(_env=env)
    else:
        agent = None
        raise NotImplementedError(f'Agent type {filepath} not implemented')

    agent.load(Path(__file__).parent / 'agents_to_compare' / filepath)

    tqdm.write(f'Running {filepath} agent...')
    time.sleep(.5)

    for run in tqdm(range(config.RUNS)):

        state, info = env.reset()

        for _ in range(config.STEPS):
            action = agent.act(state, info)
            state, reward, done, info = env.step(action)

        name = filepath.split('_')[-1]
        tot_reward[name].append(sum(env.episode_data.rewards))
        tot_wait_time[name].append(np.mean(env.episode_data.passenger_wait_times))

if run_rule_based:

    for agent_type in config.AGENTS:

        agent = fetch_agent(agent_type, env)

        tqdm.write(f'Running {agent_type} agent...')
        time.sleep(.5)

        for run in tqdm(range(config.RUNS)):

                state, info = env.reset()

                for _ in range(config.STEPS):
                    if agent_type == 'random':
                        action = env.sample_valid_action()
                    else:
                        action = agent.act(state, info)

                    state, reward, done, info = env.step(action)

                tot_reward[agent_type].append(sum(env.episode_data.rewards))
                tot_wait_time[agent_type].append(np.mean(env.episode_data.passenger_wait_times))

# set figure size
plt.rcParams["figure.figsize"] = (10, 5)
# plot reward per agent
plt.boxplot(tot_reward.values(), labels=tot_reward.keys())
plt.title(f'Reward per agent. {n_elev} elevators, {n_floors} floors')
plt.xlabel('agent')
plt.ylabel('Total reward')
plt.xticks(rotation=45)
# prevent clipping of labels
plt.tight_layout()
plt.show()

# plot wait time per agent
plt.boxplot(tot_wait_time.values(), labels=tot_wait_time.keys())
plt.title(f'Avg wait time per agent, {n_elev} elevators, {n_floors} floors')
plt.xlabel('agent')
plt.ylabel('Avg wait time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
