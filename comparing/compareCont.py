import json
import logging
import os
from collections import defaultdict
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config
from agent.continuous_agents.RLAgents_cont import RLAgent
from agent.continuous_agents.RuleBasedAgents import ETDAgent
from environment.environment_continuous import DiscreteEvent

file_loc = Path(__file__).parent / 'log.log'
logging.basicConfig(filename=file_loc, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

n_elev = 6
n_floors = 17

config.RENDER = False

rewards = defaultdict(list)
wait_times = defaultdict(list)
consumptions = defaultdict(list)
num_responding = defaultdict(list)

compare_to_RuleBased = True

for filepath in os.listdir(Path(__file__).parent / 'agents_to_compare_cont'):
    # will compare all agents in the agents_to_compare folder

    # skip DS_Store stuff and other hidden files
    if filepath[0] == '.':
        continue

    # load config file
    with open(Path(__file__).parent / 'agents_to_compare_cont' / filepath / 'config.txt', 'r') as f:
        configfile = json.load(f)
    # make object
    config.__dict__.update(configfile)

    config.RUNS = 20

    tqdm.write(f'\n Running {config.NAME} agent...')

    for i in range(1):

        env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')  # run on validation data
        agent = RLAgent(env=env, nn_type_assign=config.NN_TYPE_ASSIGN, nn_type_zone=config.NN_TYPE_ZONE, training=False)

        # load network(s) and mean+std
        agent.load(Path(__file__).parent / 'agents_to_compare_cont' / filepath, load_zone=config.LEARN_ZONING)

        name = config.NAME  # + '_' + str(i + 1)

        for run in tqdm(range(config.RUNS)):

            state, info = env.reset()
            terminated = False
            while not terminated:
                action = agent.act(state, None, info)
                state, reward, terminated, info = env.step(action)

            rewards[name].append(sum(env.episode_data.rewards))
            wait_times[name].append(np.mean(env.episode_data.passenger_total_times))
            consumptions[name].append(sum(env.episode_data.energy_consumption))
            num_responding[name].append(np.mean(env.episode_data.actions))

if compare_to_RuleBased:

    config.__dict__.update({'STATE_SIZE': 'large'})
    config.__dict__.update({'STATE_POSITION': 'distance'})
    config.__dict__.update({'STATE_ETD': 'STA'})
    env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')  # run on validation data

    # test zoning and no zoning variants
    agent1 = ETDAgent(env)
    name1 = 'ETDAgent_no_zoning'

    agent2 = ETDAgent(env, set_zoning=True)
    name2 = 'ETDAgent_zoning'

    for agent, name in zip([agent1, agent2], [name1, name2]):

        if name == 'ETDAgent_zoning':
            config.__dict__.update({'ZONING': True})
            env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')

        tqdm.write(f'\n Running {name} agent...')

        for run in tqdm(range(config.RUNS)):

            state, info = env.reset()
            terminated = False
            while not terminated:
                action = agent.act(state, env, info)
                state, reward, terminated, info = env.step(action)

            rewards[name].append(sum(env.episode_data.rewards))
            wait_times[name].append(np.mean(env.episode_data.passenger_total_times))
            consumptions[name].append(sum(env.episode_data.energy_consumption))
            num_responding[name].append(np.mean(env.episode_data.actions))


# plot data
def plotting(data, title, ylabel):

    # sort data on name
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}

    plt.figure()
    # prevent xlabels from being cut off
    plt.gcf().subplots_adjust(bottom=0.33, left=0.17)
    for i, name in enumerate(sorted_data.keys()):
        plt.boxplot(sorted_data[name], positions=[i + 1], widths=0.6)
    plt.xlabel('Agent')
    plt.ylabel(ylabel)
    plt.title(title)
    # rotate x labels
    plt.xticks(list(range(1, len(sorted_data.keys()) + 1)), list(sorted_data.keys()), rotation=45)
    plt.show()


sns.set(style='whitegrid')
for data, title, ylabel in zip([rewards, wait_times, consumptions, num_responding],
                               ['Total reward on validation environment',
                                'average passenger wait time on validation environment',
                                'Total energy consumption on validation environment',
                                'avg. amount of elevators responding per call'],
                               ['reward', 'wait time (s)', 'energy consumption (Arb. units)',
                                'amount of elevators']):
    plotting(data, title, ylabel)
