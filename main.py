import time
from collections import defaultdict

import numpy as np
import pygame
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from environment.environment_discrete import DiscreteNocam
from environment.helper_functions import fetch_agent

n_elev = 2
n_floors = 5
env = DiscreteNocam(n_elev=n_elev, n_floors=n_floors)

if config.RUN_TYPE == 'single_run':

    agent_type = config.AGENT
    agent = fetch_agent(agent_type, env)

    state, info = env.reset()

    print(f'Running {agent_type} agent...')

    for _ in range(config.STEPS):
        if agent_type == 'random':
            action = env.sample_valid_action()
        else:
            action = agent.act(state, info)

        state, reward, done, info = env.step(action)

    env.episode_data.plot_wait_times()
    env.episode_data.plot_reward_lines()

    if config.RENDER:
        pygame.quit()

elif config.RUN_TYPE == 'multiple_runs':

    # to keep track of total reward per run
    tot_reward = defaultdict(list)
    tot_wait_time = defaultdict(list)

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
