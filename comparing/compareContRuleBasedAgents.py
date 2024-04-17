import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import config
from agent.continuous_agents.RuleBasedAgents import (
    RandomAgent, SectorAgent, ClosestAgent, LeastBusyAgent, ETDAgent)
from environment import environment_continuous as environment

env = environment.DiscreteEvent(6, 17, 'woensdag_donderdag.json')
agents = [ClosestAgent(env), RandomAgent(env), SectorAgent(env), LeastBusyAgent(env), ETDAgent(env),
          ETDAgent(env, set_zoning=True)]

agents[-1].__repr__ = lambda: 'ETDAgent with zoning'  # hack to change the name of the agent

all_rewards = defaultdict(list)
all_waittimes = defaultdict(list)
all_energy_consumption = defaultdict(list)

file_loc = Path(__file__).parent / 'log.log'
logging.basicConfig(filename=file_loc, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for agent in agents:

    # try: re-make env instead of reset()
    config.__dict__.update({'STATE_POSITION': 'distance'})
    config.__dict__.update({'STATE_ETD': 'STA'})
    env = environment.DiscreteEvent(6, 17, 'woensdag_donderdag.json')

    if agent.__repr__() == 'SectorAgent':
        for ix, elev in enumerate(env.building.elevators):
            elev.rest_floor = agent.sectors[ix][0]
        env.building.elevators[0].rest_floor = 1  # ground floor
        print('repr success')

    if agent.__repr__() == 'ETDAgent with zoning':
        config.__dict__.update({'ZONING': True})
        # update env to include zoning steps
        env = environment.DiscreteEvent(n_elev=6, n_floors=17, data='woensdag_donderdag.json')

    tqdm.write(f'\n Running {agent.__repr__()} agent...')

    for run in tqdm(range(20)):

        terminated = False
        state, info = env.reset()
        while not terminated:
            action = agent.act(state, env, info)
            state, reward, terminated, info = env.step(action)

        total_reward = sum(env.episode_data.rewards)
        waittime = np.mean(env.episode_data.passenger_total_times)
        consumptions = sum(env.episode_data.energy_consumption)

        all_rewards[agent.__repr__()].append(total_reward)
        all_waittimes[agent.__repr__()].append(waittime)
        all_energy_consumption[agent.__repr__()].append(consumptions)

# Plotting
sns.set(style='whitegrid')
plt.gcf().subplots_adjust(bottom=0.33, left=0.2)
plt.boxplot(all_rewards.values(), labels=all_rewards.keys())
plt.ylabel('Reward')
plt.xlabel('Agent')
plt.title('Comparison of rule-based agents')
# rotate x-axis labels
plt.xticks(rotation=45)
plt.show()

plt.gcf().subplots_adjust(bottom=0.33)
plt.boxplot(all_waittimes.values(), labels=all_waittimes.keys())
plt.ylabel('Average wait time')
plt.xlabel('Agent')
plt.title('Comparison of rule-based agents')
plt.xticks(rotation=45)
plt.show()

plt.gcf().subplots_adjust(bottom=0.33)
plt.boxplot(all_energy_consumption.values(), labels=all_energy_consumption.keys())
plt.ylabel('Energy consumption')
plt.xlabel('Agent')
plt.title('Comparison of rule-based agents')
plt.xticks(rotation=45)
plt.show()

# fig, ax = plt.subplots(1, len(agents))
#
# for ix, agent in enumerate(agents):
#     ax[ix].boxplot(all_rewards[agent.__class__.__name__])
#     ax[ix].set_title(agent.__class__.__name__)
#     ax[ix].set_xlabel('Episode')
#     ax[ix].set_ylabel('Reward')
#
# plt.show()
#
# print('Results: \n'
#       f'ETDAgent: {np.mean(all_rewards["ETDAgent"])} \n'
#       f'SectorAgent: {np.mean(all_rewards["SectorAgent"])} \n'
#       f'ClosestAgent: {np.mean(all_rewards["ClosestAgent"])} \n'
#       f'LeastBusyAgent: {np.mean(all_rewards["LeastBusyAgent"])} \n'
#       f'RandomAgent: {np.mean(all_rewards["RandomAgent"])} \n')




