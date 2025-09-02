import datetime
import json
import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import seaborn as sns

import config as config
from environment.building import Building
from environment.helper_functions import ElevatorMetrics
from environment.passenger import Passenger
from environment.rendering import Rendering

logger = logging.getLogger(__name__)


class DiscreteEvent(gym.Env):

    def __repr__(self):
        state = self.get_state_assign()
        return f"DiscreteEvent(n_elev={self.n_elev}, n_floors={self.n_floors}):\n" \
               f"Time: {self.time}\n" \
               "Next call: \n" \
               f"floor: {state['floor']}\n" \
               f"direction: {state['direction']}"

    def __len__(self):
        return len(self.data)

    def __init__(self, n_elev, n_floors, data=None):

        super().__init__()
        self.learn_zoning = config.LEARN_ZONING
        self.discount_factor = config.DISCOUNT_FACTOR
        self.timedelta = config.TIMEDELTA
        self.n_elev = n_elev
        self.n_floors = n_floors
        self.time = datetime.datetime(2023, 6, 7, 6, 0, 0)
        self.data = self.import_data(data=data)
        self.p_params = self.make_groupsize_array()
        self.building = Building(self.n_floors, self.n_elev)
        self.episode_data = ElevatorMetrics()

        self.rewards = {}
        self.zoning = False
        self.pass_time_waited = None
        self.pass_destination = None
        self.pass_direction = None
        self.pass_floor = None
        self.next_pass_arrival_time = None
        self.terminated = False
        self.stranded_passenger = None
        self.stranded_passengers = []
        self.re_register = False

        if config.RENDER:
            self.rendering = Rendering(self, 'continuous')

    def reset(self, seed: int = None, options=None, start_time: datetime.datetime = None):
        """
        Reset the environment to a new episode
        :param seed: set seed for random number generator
        :param options: TBD
        :param start_time: start at a specific time
        :return: initial state
        """
        # self.time = datetime.datetime(2023, 6, 7, 6, 0, 0)
        self._set_state_iter(self.data)
        self._set_next_state()  # go to first event
        if start_time is not None:  # or go to specific start time
            self.time = start_time
            self.start_at_time(start_time)
        self.time = self.next_pass_arrival_time  # update time to first passenger arrival, the first event
        self.building.reset()
        self._update_env()  # register passenger at floor
        self.episode_data = ElevatorMetrics()
        self.zoning = False
        self.stranded_passenger = None
        self.stranded_passengers = []
        self.terminated = False
        self.re_register = False
        return self.get_state_assign(), {}

    def step(self, actions: np.ndarray, debugmode=False) -> tuple[dict, int, bool, dict]:
        """
        take a step in the environment. Process action, simulate until next decision point
        :param actions: list of actions, one for each elevator.
        :param debugmode:
        :return: state, reward, done, info
        """

        # 1. apply actions decided by agent(s).

        # if no elevators respond to call, give penalty
        if not self.zoning and sum(actions) == 0:
            action_reward = config.ZERO_ELEV_PENALTY
            # keep track of how many times this happens
            self.episode_data.zero_elevs_responding += 1
        else:
            action_reward = 0

        # special block for re-registering stranded passengers
        if self.re_register:
            passenger = self.stranded_passenger  # stranded passenger
            self.building.update_destinations(actions, passenger)
            self.re_register = False
            # add how many elevs responded to the call
            self.episode_data.actions.append(sum(actions))

        elif self.zoning:
            # if zoning, no passengers are generated. Set zones and move to next passenger arrival
            self.building.set_zones(actions)
            self.zoning = False  # toggle off

        else:
            passenger = self.building.floor_passengers[-1]  # fetch last added passenger (the relevant hall call)
            # update destinations of elevators according to actions
            self.building.update_destinations(actions, passenger)

            # add how many elevs responded to the call
            self.episode_data.actions.append(sum(actions))

            # fetch data for next passenger arrival. Updates self.next_pass_arrival_time, self.pass_floor,
            # self.pass_direction, self.pass_destination, self.pass_time_waited
            # This allows us to simulate until next passenger arrival, and then return the state
            self._set_next_state()
            # when group arrival consists of more than 1 destination, process all
            while (passenger.floor == self.pass_floor and passenger.direction == self.pass_direction and
                   passenger.arrival_time == self.next_pass_arrival_time):
                self._update_env()  # register passenger at floor. ! every call is a group
                self._set_next_state()  # fetch next passenger from data
                # if passenger is same floor and same direction, previous assignment decision counts for this pass.
                # as well. Dont return state (as it doesnt require an action from the agent)
                # but register passenger and continue until next passenger arrival.
                if self.terminated:
                    break

        # 2. simulate until next passenger arrival: automatic behaviour
        rewards = []
        while self.time <= self.next_pass_arrival_time:

            self.time += datetime.timedelta(seconds=self.timedelta)  # increment time

            # get all stranded passengers = passengers which' floor appears in no elevator's destination list
            self.stranded_passengers = self.building.get_stranded_passengers()

            # if there are any stranded passengers, get action from policy
            if len(self.stranded_passengers) > 0:
                logger.warning(f'{self.time}: stranded passenger! Re-registering at floor')
                self.stranded_passenger = self.stranded_passengers.pop(0)

                # get state
                self.building.update_buttons()
                # construct state for lost passenger
                state = self.get_state_assign()
                # override with stranded passenger info
                state['floor'] = self.stranded_passenger.floor
                state['direction'] = 1 if self.stranded_passenger.direction == 'up' else \
                    -1 if self.stranded_passenger.direction == 'down' else None

                step_length = len(rewards)
                discounted_reward = self.process_rewards(rewards) + action_reward
                self.episode_data.add_reward(discounted_reward)
                # register timestamp of when reward was received. For plotting
                self.episode_data.dates.append(self.time)
                info = {'step_length': step_length}

                self.re_register = True

                return state, discounted_reward, self.terminated, info

            # if it is a whole hour or half hour (e.g. 12:00:00), return state for zoning agent
            elif self.learn_zoning and (self.time.minute == 00 or self.time.minute == 30) and self.time.second == 0 \
                    and self.time.microsecond == 0:

                self.zoning = True
                self.building.update_buttons()
                # return special state for zoning agent
                state = self.get_state_zoning()
                discounted_reward = self.process_rewards(rewards) + action_reward
                self.episode_data.add_reward(discounted_reward)
                # self.episode_data.dates.append(self.time)
                info = {'zoning': True, 'step_length': len(rewards)}
                return state, discounted_reward, self.terminated, info

            reward = self.building.infra_step(self.episode_data, self.time)
            rewards.append(reward)

            # keep track of energy consumption (whenever an elevator moves) -> add to total energy consumption
            energy_consumed = sum([1 for x in self.building.elevators if not x.stopped])
            self.episode_data.energy_consumption.append(energy_consumed)

            if config.RENDER:
                self.rendering.render()

        # !Step length is the length from the relevant passenger until the next one, not before.
        step_length = len(rewards)
        discounted_reward = self.process_rewards(rewards) + action_reward
        self.episode_data.add_reward(discounted_reward)
        self.episode_data.dates.append(self.time)

        if not self.terminated:
            self._update_env()  # register next passenger(s) at floor, update buttons, etc.
        state = self.get_state_assign()

        info = {'step_length': step_length}  # needed for variable discounting scheme

        return state, discounted_reward, self.terminated, info

    @staticmethod
    def process_rewards(rewards):

        # -> set discount factor at infra-level, so rewards over one step are discounted within that step
        # then, at main level, rewards over steps are discounted depending on length of step

        if config.DISCOUNTING_SCHEME == 'variable':
            total_reward = 0
            discount = 0.9999  # self.discount_factor
            for reward in rewards:
                # every element in list is 1 infra-step
                total_reward += reward * discount
                # increment discount for next step
                discount *= discount

        elif config.DISCOUNTING_SCHEME == 'fixed':
            total_reward = sum(rewards)

        else:
            raise ValueError(f'Unknown discounting scheme: {config.DISCOUNTING_SCHEME}')

        return total_reward

    @staticmethod
    def import_data(data):
        data = 'passenger_data.json' if data is None else data
        # make import location robust
        cur_dir = os.path.dirname(__file__)
        with open(os.path.join(cur_dir, 'data', 'JSON', f'{data}')) as f:
            data = json.load(f)
        # change date string to datetime object. Format: '%Y-%m-%d %H:%M:%S'
        data_filtered = []
        for passenger in data:
            passenger['arrival_time'] = datetime.datetime.strptime(passenger['arrival_time'], '%Y-%m-%d %H:%M:%S')
            # remove passengers with no destination: extra hall calls.
            if passenger['destination'] is not None:
                data_filtered.append(passenger)
        return data_filtered

    def _update_env(self):
        """
        Register passengers at floor, update buttons
        """
        # register passengers at floor
        p_param = self.p_params[self.time.hour]
        group_n = np.random.geometric(p_param)
        # assign n passengers to floor, based on geometric distribution
        for _ in range(group_n):
            self.building.floor_passengers.append(
                Passenger(self.pass_floor, self.pass_destination, self.next_pass_arrival_time,
                          self.pass_direction))

        # save how many passengers, direction, and timestamp
        self.episode_data.group_sizes.append(group_n)
        self.episode_data.group_directions.append(self.pass_direction)
        self.episode_data.group_arrival_times.append(self.next_pass_arrival_time)
        # update buttons
        self.building.update_buttons()

    def get_state_assign(self):

        # convert time to fraction of day
        dt = datetime.timedelta(hours=self.time.hour, minutes=self.time.minute, seconds=self.time.second)
        current_time = dt.total_seconds() / (24 * 60 * 60)

        if config.STATE_POSITION == 'position':  # how to encode position of elevators
            position_enc = self.building.get_elevator_positions()  # position of elevators (floors)
        elif config.STATE_POSITION == 'distance':
            position_enc = self.building.get_elevator_distances(self.pass_floor)  # distance to target (meters)
        else:
            raise ValueError(f'Unknown state position encoding: {config.STATE_POSITION}')

        if config.STATE_ETD == 'STA':  # how to encode how busy the elevators are
            ETA = self.building.destination_queue_length()  # how many stops till available (STA)
        elif config.STATE_ETD == 'ETD':
            ETA = self.building.get_ETDs(self.pass_floor, self.pass_direction)  # estimated time to destination (ETD)
        else:
            raise ValueError(f'Unknown state ETD encoding: {config.STATE_ETD}')

        if config.STATE_SIZE == 'small':
            state = {'floor': self.pass_floor,
                     'direction': 1 if self.pass_direction == 'up' else -1 if self.pass_direction == 'down' else None,
                     'position_enc': position_enc,
                     'eta': ETA,
                     'time_of_day': current_time,  # as fraction of the day.
                     }

        elif config.STATE_SIZE == 'large':
            state = {'floor': self.pass_floor,
                     'direction': 1 if self.pass_direction == 'up' else -1 if self.pass_direction == 'down' else None,
                     'position_enc': position_enc,
                     'eta': ETA,
                     'time_of_day': current_time,  # as fraction of the day.
                     'day_of_week': self.time.weekday(),  # 0 is monday, 6 is sunday
                     'speed': self.building.get_elevator_speeds(),  # + is up, - is down
                     'weight': self.building.get_elevator_weights(),  # indirect measure of how busy the elev is
                     # TODO: also include whether its past threshold for stopping?
                     # TODO: also include next_stops for each elevator? ! variable length ! -> use elev buttons
                     }
        else:
            raise ValueError(f'Unknown state size: {config.STATE_SIZE}')

        return state

    def get_state_zoning(self):

        # convert time to fraction of day
        dt = datetime.timedelta(hours=self.time.hour, minutes=self.time.minute, seconds=self.time.second)
        current_time = dt.total_seconds() / (24 * 60 * 60)

        if config.NN_TYPE_ZONE == 'sequential':
            state = {'current_time': current_time,
                     'day_of_week': self.time.weekday(),  # 0 is monday, 6 is sunday
                     'other_elevs_zone': np.full(self.n_elev - 1, -1),  # placeholder, elevators will decide
                     'elev_index': 0}  # index of elevator that we currently want to zone (increments)
        elif config.NN_TYPE_ZONE == 'branching':
            state = {'current_time': current_time,
                     'day_of_week': self.time.weekday()}  # 0 is monday, 6 is sunday
        else:
            raise ValueError(f'Unknown NN type for zoning: {config.NN_TYPE_ZONE}')

        return state

    def _set_state_iter(self, passenger_data: list) -> None:
        self._state_iter = iter(passenger_data)

    def start_at_time(self, start_time: datetime.datetime) -> None:

        while self.next_pass_arrival_time < start_time:
            self.next_pass_arrival_time, self.pass_floor, self.pass_direction, self.pass_destination, \
                self.pass_time_waited = next(self._state_iter).values()

    def _set_next_state(self):
        try:
            self.next_pass_arrival_time, self.pass_floor, self.pass_direction, self.pass_destination, \
                self.pass_time_waited = next(self._state_iter).values()
        except StopIteration:
            self.terminated = True

    @staticmethod
    def make_groupsize_array():
        """for every hour, determine P for geometric distribution of group sizes"""

        try:
            busyness_multiplier = config.BUSYNESS_MULTIPLIER
        except AttributeError:
            busyness_multiplier = 1

        # if mean group size is X, p = 1/X
        group_sizes = np.array([1,  # 00.00
                                1,  # 1.00
                                1,  # 2.00
                                1,  # 3.00
                                1,  # 4.00
                                1,  # 5.00
                                1,  # 6.00
                                1.2,  # 7.00
                                1.5,  # 8.00
                                1.3,  # 9.00
                                1.4,  # 10.00
                                1.4,  # 11.00
                                1.35,  # 12.00
                                1.35,  # 13.00
                                1.3,  # 14.00
                                1.6,  # 15.00
                                1.4,  # 16.00
                                1.4,  # 17.00
                                1.4,  # 18.00
                                1.35,  # 19.00
                                1.3,  # 20.00
                                1.2,  # 21.00
                                1,  # 22.00
                                1])  # 23.00

        group_sizes *= busyness_multiplier

        # array of p-parameter for geometric distribution
        p_array = 1 / group_sizes

        return p_array


if __name__ == '__main__':

    # if file ran as main, run some tests and plot data from environment dynamics

    # from agent.continuous_agents.RuleBasedAgents import ETDAgent
    from agent.continuous_agents.RLAgents_cont import RLAgent

    sns.set_theme()

    # set up log file
    logging.basicConfig(filename=Path(__file__).parent / 'log.log', level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')

    agent_file = (Path(__file__).parent.parent / 'agent/continuous_agents/data/2024-04-18_12-32-53')

    # load config file
    with open(Path(agent_file) / 'config.txt', 'r') as f:
        configfile = json.load(f)
    config.__dict__.update(configfile)
    # config.RENDER = True

    env = DiscreteEvent(n_elev=6, n_floors=17)

    agent = RLAgent(env=env, nn_type_assign=config.NN_TYPE_ASSIGN, nn_type_zone=config.NN_TYPE_ZONE, training=False)
    agent.load(agent_file, load_zone=config.LEARN_ZONING)

    terminated = False
    step_lengths = []
    rewards = []
    state, info = env.reset(start_time=datetime.datetime(2023, 10, 30, 8, 55, 0))
    while not terminated:
        action = agent.act(state, env, info)
        state, reward, terminated, info = env.step(action)
        step_lengths.append(info['step_length'])
        rewards.append(reward)

    # get 99th percentile and cutoff (extremely long tail, prevents proper plotting)
    cutoff = np.percentile(step_lengths, 99)
    # set outliers to cutoff
    step_lengths = [cutoff if x > cutoff else x for x in step_lengths]
    # divide step length by 10 to get seconds (step is 0.1s)
    step_lengths = np.array(step_lengths) / 10

    import matplotlib.pyplot as plt

    # plot distribution of step lengths
    plt.hist(step_lengths, bins=100)
    plt.title('Distribution of step lengths in training data')
    plt.xlabel('Step length (s)')
    plt.ylabel('Frequency')
    plt.show()

    # plot rewards distribution
    plt.hist(rewards, bins=100)
    plt.title('Distribution of rewards in training data')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.show()

    # plot rewards as function of step length
    plt.rcParams['lines.markersize'] = 0.5  # make dots smaller
    plt.scatter(rewards, step_lengths)
    plt.title('Reward X Step length')
    plt.xlabel('Reward')
    plt.ylabel('Step length (s)')
    plt.savefig('reward_x_step_length.pdf', format='pdf')
    plt.show()

    # plot mean group size per hour
    hours = [x.hour for x in env.episode_data.group_arrival_times]
    group_sizes = env.episode_data.group_sizes
    mean_group_sizes = [np.mean([group_sizes[i] for i, x in enumerate(hours) if x == h]) for h in range(24)]
    plt.bar(range(24), mean_group_sizes)
    # add theoretical p_params for group size
    p_params = env.p_params
    plt.plot(range(24), 1 / p_params, color='r', label='Theoretical p parameter')
    plt.title('Mean group size per hour')
    plt.xlabel('Hour')
    plt.ylabel('Mean group size')
    plt.legend()
    plt.savefig('mean_group_size_per_hour.pdf', format='pdf')
    plt.show()

    # plot mean people per hour
    people = env.episode_data.group_sizes
    total_people = [np.sum([group_sizes[i] for i, x in enumerate(hours) if x == h]) for h in range(24)]
    plt.bar(range(24), total_people)
    plt.title('Total people per hour')
    plt.xlabel('Hour')
    plt.ylabel('Total people')
    plt.show()

    # plot directions per hour
    directions = env.episode_data.group_directions
    up = [sum([1 for i, x in enumerate(directions) if x == 'up' and hours[i] == h]) for h in range(24)]
    down = [sum([1 for i, x in enumerate(directions) if x == 'down' and hours[i] == h]) for h in range(24)]
    width = 0.35
    plt.bar(range(24), up, label='up', color='r', width=width)
    plt.bar([x + width for x in range(24)], down, label='down', color='b', width=width)
    plt.title('Direction of group arrivals per hour')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('directions_per_hour.pdf', format='pdf')
    plt.show()

    # analyse reward construction
    sums = [sum(x) for x in [env.episode_data.rewards_full,
                             env.episode_data.rewards_arrival,
                             env.episode_data.rewards_loading,
                             env.episode_data.rewards_moving,
                             env.episode_data.rewards_wait_elev,
                             env.episode_data.rewards_wait_floor]]
    # make pie chart
    p, tx, autotexts = plt.pie([abs(x) for x in sums],
                               labels=['full', 'arrival', 'loading', 'moving', 'wait elev', 'wait floor'],
                               autopct="", colors=sns.color_palette("Spectral"))
    # add value to each slice
    for i, a in enumerate(autotexts):
        a.set_text(f'{round(sums[i], 1)}')
    plt.title(f'Reward construction: RL Agent')
    plt.savefig('reward_construction.pdf', format='pdf')
    plt.show()

    # make barplot as well
    plt.bar(['full', 'arrival', 'loading', 'moving', 'wait elev', 'wait floor'], sums)
    plt.title(f'Reward construction: {agent}')
    plt.show()

    # make plot showing how many elevators are responding to calls per hour
    # hours
    timesteps = env.episode_data.dates
    hours = [x.hour for x in timesteps]
    # number of elevs responding per timestep
    responding = env.episode_data.actions
    # sum responding elevs per hour
    responding_per_hour = [np.mean([responding[i] for i, x in enumerate(hours) if x == h]) for h in range(24)]
    plt.bar(range(24), responding_per_hour)
    plt.title('Elevators responding to calls per hour')
    plt.xlabel('Hour')
    plt.xlim(0, 24)
    plt.ylim(0.7, 1.9)
    plt.ylabel('Elevators responding')
    plt.savefig('elevs_responding_per_hour.pdf', format='pdf')
    plt.show()
