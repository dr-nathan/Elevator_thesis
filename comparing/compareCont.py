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
compare_to_RuleBased = True


def run(BM):

    rewards = defaultdict(list)
    wait_times = defaultdict(list)
    consumptions = defaultdict(list)
    num_responding = defaultdict(list)

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
        config.BUSYNESS_MULTIPLIER = float(BM)

        config.RUNS = 20

        tqdm.write(f'\n Running {config.NAME} agent...')

        for i in range(1):

            env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')  # run on validation data
            agent = RLAgent(env=env, nn_type_assign=config.NN_TYPE_ASSIGN, nn_type_zone=config.NN_TYPE_ZONE, training=False)

            # load network(s) and mean+std
            agent.load(Path(__file__).parent / 'agents_to_compare_cont' / filepath, load_zone=config.LEARN_ZONING)

            name = f'{config.NAME}' # {BM}x'

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
        name1 = 'ETDAgent'  # _no_zoning'

        # agent2 = ETDAgent(env, set_zoning=True)
        # name2 = 'ETDAgent_zoning'
        #
        # for agent, name in zip([agent1, agent2], [name1, name2]):

        # if name == 'ETDAgent_zoning':
        #     config.__dict__.update({'ZONING': True})
        #     env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')

        agent, name = agent1, name1

        name = f'{name}' # {BM}x'

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

    return rewards, wait_times, consumptions, num_responding


# plot data
def plotting(data, title, ylabel):

    # custom sort data
    # sort_list = ['ETDAgent 1x', 'Combinatorial 1x', 'ETDAgent 1.5x',
    #               'Combinatorial 1.5x', 'ETDAgent 2x', 'Combinatorial 2x']
    # sorted_data = {k: data[k] for k in sort_list}

    # sort alphabetically
    sorted_data = {k: data[k] for k in sorted(data.keys())}

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


rewards, wait_times, consumptions, num_responding = run('1')
# rewards15, wait_times15, consumptions15, num_responding15 = run('1.5')
# rewards2, wait_times2, consumptions2, num_responding2 = run('2')
# # merge dicts
# rewards.update(rewards15)
# rewards.update(rewards2)
# wait_times.update(wait_times15)
# wait_times.update(wait_times2)
# consumptions.update(consumptions15)
# consumptions.update(consumptions2)
# num_responding.update(num_responding15)
# num_responding.update(num_responding2)

sns.set(style='whitegrid')
for data, title, ylabel in zip([rewards, wait_times, consumptions, num_responding],
                               ['Total reward on validation environment',
                                'average passenger wait time on validation environment',
                                'Total energy consumption on validation environment',
                                'avg. amount of elevators responding per call'],
                               ['reward', 'wait time (s)', 'energy consumption (Arb. units)',
                                'amount of elevators']):
    plotting(data, title, ylabel)

# rename dict keys
# rewards['Branching'] = rewards.pop('Dueling Branching')
# rewards['Combinatorial'] = rewards.pop('Dueling Combinatorial')
# wait_times['Branching'] = wait_times.pop('Dueling Branching')
# wait_times['Combinatorial'] = wait_times.pop('Dueling Combinatorial')
# consumptions['Branching'] = consumptions.pop('Dueling Branching')
# consumptions['Combinatorial'] = consumptions.pop('Dueling Combinatorial')
# num_responding['Branching'] = num_responding.pop('Dueling Branching')
# num_responding['Combinatorial'] = num_responding.pop('Dueling Combinatorial')

# print means and stds
for name, data in rewards.items():
    print(f'{name} reward: {np.mean(data)} +- {np.std(data)}')
