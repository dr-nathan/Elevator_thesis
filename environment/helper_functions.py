import abc
import datetime
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.special import comb, factorial
from scipy.stats import rv_discrete

from agent.discrete_agents.RLAgents_disc import DQNAgent
from agent.discrete_agents.RLagents_multiagents import DDQNAgentMulti
from agent.discrete_agents.RuleBasedAgents import ConventionalAgent, SECTOR


class PolyaAeppli(rv_discrete):

    def _pmf(self, k, lambda_, theta_):
        """
        Probability mass function for Polya-Aeppli distribution. Extension of Poisson distribution for
        modeling group arrivals.

        :param k: internal parameter of rv_discrete
        :param lambda_: arrival rate param of Poisson dist: [0, inf). The higher the lambda, the more arrivals
        :param theta_: probability param of Geometric dist: [0, 1]. The LOWER the theta, the more arrivals
        :return: probability values for each k
        """

        lambda_ = np.atleast_1d(lambda_)
        theta_ = np.atleast_1d(theta_)
        k = np.atleast_1d(k)

        res = np.zeros(len(k))
        zero_indices = np.where(k == 0)[0]
        non_zero_indices = np.where(k != 0)[0]

        # if k = 0, then the probability is just exp(-lambda)
        res[zero_indices] = np.exp(-lambda_[0])

        if len(non_zero_indices) > 0:
            k_non_zero = k[non_zero_indices]
            k_list = np.arange(1, np.max(k_non_zero) + 1)

            powers = np.power(lambda_[0], k_list)
            factors = powers / factorial(k_list)
            subtract_values = np.subtract(k_non_zero[:, np.newaxis], k_list)
            mask = subtract_values >= 0

            res[non_zero_indices] = \
                np.exp(-lambda_[0]) * \
                np.sum(
                    factors * np.power(np.subtract(1, theta_[0]), subtract_values * mask) *
                    np.power(theta_[0], k_list) * comb((k_non_zero - 1).reshape(-1, 1), k_non_zero - 1),
                    axis=1)

        return res


@dataclass
class ElevatorMetrics:
    passenger_wait_times: list[int] = field(default_factory=list)
    passenger_travel_times: list[int] = field(default_factory=list)
    dates: list = field(default_factory=list)
    rewards: list[int] = field(default_factory=list)
    energy_consumption: list[int] = field(default_factory=list)
    zero_elevs_responding: int = 0
    actions: list[int] = field(default_factory=list)
    group_sizes: list[int] = field(default_factory=list)
    group_directions: list[int] = field(default_factory=list)
    group_arrival_times: list[datetime.datetime] = field(default_factory=list)
    rewards_wait_floor: list[int] = field(default_factory=list)
    rewards_wait_elev: list[int] = field(default_factory=list)
    rewards_moving: list[int] = field(default_factory=list)
    rewards_arrival: list[int] = field(default_factory=list)
    rewards_loading: list[int] = field(default_factory=list)
    rewards_full: list[int] = field(default_factory=list)

    passenger_total_times = property(lambda self: [self.passenger_wait_times[i] + self.passenger_travel_times[i]
                                                   for i in range(len(self.passenger_wait_times))])

    def __len__(self):
        return len(self.rewards)

    def add_passenger_data(self, wait_time, travel_time):
        self.passenger_wait_times.append(wait_time)
        self.passenger_travel_times.append(travel_time)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def plot_wait_time_distribution(self):
        plt.hist(self.passenger_wait_times, bins=100)
        plt.xlabel("Wait time (s)")
        plt.ylabel("Frequency")
        plt.title("Wait time distribution")
        plt.show()

    def plot_travel_time_distribution(self):
        plt.hist(self.passenger_travel_times, bins=100)
        plt.xlabel("Travel time (s)")
        plt.ylabel("Frequency")
        plt.title("Travel time distribution")
        plt.show()

    def plot_total_time_distribution(self):
        plt.hist(self.passenger_total_times, bins=100)
        plt.xlabel("Total time (s)")
        plt.ylabel("Frequency")
        plt.title("Total time distribution")
        plt.show()

    def plot_smoothed_wait_time_through_time(self, smoothing=20):
        smoothed = np.convolve(self.passenger_wait_times, np.ones(smoothing) / smoothing, mode='valid')
        plt.plot(smoothed)
        plt.xlabel("Time (s)")
        plt.ylabel("Wait time (s)")
        plt.title("Average wait time through time")
        plt.show()

    def plot_reward(self):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.plot(self.dates, self.rewards)
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=90)
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.title("Reward through time")
        plt.show()

    def plot_cumulative_reward(self):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.plot(self.dates, np.cumsum(self.rewards))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=90)
        plt.xlabel("Time")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative reward through time")
        plt.show()

    def plot_wait_times(self):
        seaborn.set_theme(style="whitegrid")
        self.plot_wait_time_distribution()
        self.plot_travel_time_distribution()
        self.plot_total_time_distribution()
        self.plot_smoothed_wait_time_through_time()

    def plot_reward_lines(self):
        # set figure size for all plots
        plt.rcParams["figure.figsize"] = (18, 6)
        # set theme for all plots
        seaborn.set_theme(style="whitegrid")
        self.plot_reward()
        self.plot_cumulative_reward()


def fetch_agent(agent_type: str, env):

    match agent_type:
        case 'conventional':
            agent = ConventionalAgent(env)
        case 'sector':
            agent = SECTOR(env)
        case 'DQNSingle':
            agent = DQNAgent(_env=env)
            filepath = Path(__file__).parent.parent / 'agent' / 'data' / f'single_online_network_{env.n_elev}' \
                                                                f'elev_{env.n_floors}floors.pt'
            try:
                agent.load(filepath)
            except FileNotFoundError:
                raise FileNotFoundError(f'No online network found for {env.n_elev} elevators '
                                        f'and {env.n_floors} floors. '
                                        'Please train a network first.')
        case 'DQNMulti':
            agent = DDQNAgentMulti(_env=env)
            filepath = Path(__file__).parent.parent / 'agent' / 'data' / (f'multi_online_network_{env.n_elev}elev_'
                                                                          f'{env.n_floors}floors')
            try:
                agent.load(filepath)
            except FileNotFoundError:
                raise FileNotFoundError(f'No online network found for {env.n_elev} elevators and'
                                        f' {env.n_floors} floors. '
                                        'Please train a network first.')
        case 'random':
            agent = None
            # doesnt need an agent, environment will sample random actions
        case _:
            raise NotImplementedError(f'Agent type {agent_type} not implemented')

    return agent


class KeyTransformDictionaryBase(dict, abc.ABC):

    @abc.abstractmethod
    def __key_transform__(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        return super().__contains__(self.__key_transform__(key))

    def __getitem__(self, key):
        return super().__getitem__(self.__key_transform__(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.__key_transform__(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self.__key_transform__(key))


class FloatKeyDictionary(KeyTransformDictionaryBase):

    def __init__(self, rounding_ndigits=1, data=None):
        super().__init__()
        self.rounding_ndigits = rounding_ndigits
        if data is not None:
            self.update(data)

    def __key_transform__(self, key):
        # key[0] is floor, key[1] is direction, key[2] is speed
        return str(key[0]) + key[1] + str(round(float(key[2]), self.rounding_ndigits))


if __name__ == '__main__':

    PA = PolyaAeppli(name='polya_aeppli')

    # # plot pmf
    x = np.arange(0, 10)
    plt.plot(x, PA.pmf(x, lambda_=0.5, theta_=1.0), label='a')
    plt.plot(x, PA.pmf(x, lambda_=0.5, theta_=0.5), label='b')
    plt.plot(x, PA.pmf(x, lambda_=1.0, theta_=1.0), label='c')
    plt.plot(x, PA.pmf(x, lambda_=1.0, theta_=0.5), label='d')
    plt.plot(x, PA.pmf(x, lambda_=2.5, theta_=1.0), label='e')
    plt.plot(x, PA.pmf(x, lambda_=2.5, theta_=0.5), label='f')
    plt.legend()
    plt.show()

    # # plot cdf
    x = np.arange(0, 10)
    plt.plot(x, PA.cdf(x, lambda_=0.5, theta_=1.0), label='a')
    plt.plot(x, PA.cdf(x, lambda_=0.5, theta_=0.5), label='b')
    plt.plot(x, PA.cdf(x, lambda_=1.0, theta_=1.0), label='c')
    plt.plot(x, PA.cdf(x, lambda_=1.0, theta_=0.5), label='d')
    plt.plot(x, PA.cdf(x, lambda_=2.5, theta_=1.0), label='e')
    plt.plot(x, PA.cdf(x, lambda_=2.5, theta_=0.5), label='f')
    plt.legend()
    plt.show()
