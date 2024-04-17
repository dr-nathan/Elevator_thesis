import logging

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary

import config
import environment.var_mapping as vm
from environment.helper_functions import ElevatorMetrics
from environment.rendering import Rendering
from .elevator import Elevator
from .passenger import Passenger

logger = logging.getLogger(__name__)


class DiscreteNocam(gym.Env):
    """
    Discrete environment with no camera
    """

    floor_passengers = []
    exiting_passengers = []
    time_of_day = None
    floor_buttons = None
    elev_buttons = None
    floor_button_times = None
    elev_button_times = None
    elevators = None
    reward = None
    timecounter = None
    rendering = None

    def __init__(self, n_elev: int, n_floors: int):

        """"
        :param n_elev: number of elevators
        :param n_floors: number of floors
        """

        super().__init__()

        self.n_elev = n_elev
        self.n_floors = n_floors

        self.movement_penalty = config.MOVEMENT_PENALTY
        self.waiting_penalty = config.WAITING_PENALTY
        self.arrival_reward = config.ARRIVAL_REWARD
        self.loading_reward = config.LOADING_REWARD

        self.default_action = [None] * self.n_elev

        self.observation_space = Dict({
            # position of elevator 1 to n: n_floors x n_elev
            'elev_position': MultiDiscrete([self.n_floors] * self.n_elev),
            # elev direction: n_elev x (0,1,-1)
            'elev_direction': MultiDiscrete([3] * self.n_elev),
            # if button inside elevator is pressed: n_floors x (0,1) x n_elev
            'elev_button': MultiBinary([self.n_elev, self.n_floors]),
            # if up/down button is pressed inside elevator: 2x (0,1) x n_floors
            'floor_button': MultiBinary([self.n_floors, 2]),
            # time of day, 3 categories
            'time_of_day': Discrete(3)})

        self.action_space = MultiDiscrete([3] * self.n_elev)  # 3 actions per elevator: up, down, do nothing
        # NOTE: this implies a central controller for all elevators

        self.episode_data = ElevatorMetrics()

    def __repr__(self):
        state = self._get_state()
        return f"DiscreteNocam(n_elev={self.n_elev}, n_floors={self.n_floors}):\n" \
               f"State: elev_position: {state['elev_position']}\n" \
               f"elev_direction: {state['elev_direction']}\n" \
               f"elev_button: {state['elev_button']}\n" \
               f"floor_button: {state['floor_button']}\n" \
               f"time_of_day: {state['time_of_day']}"

    def reset(self, seed=None, options=None) -> tuple:
        """
        Reset environment to initial state
        :return: state of environment, info dict
        """

        super().reset()

        # reset episode data
        self.episode_data = ElevatorMetrics()

        self.elevators = [Elevator(self.n_floors) for _ in range(self.n_elev)]
        # elevator position is attribute of elevator object
        self.floor_passengers = []
        self.elev_buttons = np.zeros((self.n_elev, self.n_floors))
        self.floor_buttons = np.zeros((self.n_floors, 2))
        self.elev_button_times = np.zeros((self.n_elev, self.n_floors))
        self.floor_button_times = np.zeros((self.n_floors, 2))
        self.time_of_day = np.array([0])
        self.timecounter = 0
        self.reward = 0.0
        self.default_action = [None] * self.n_elev
        self.exiting_passengers = []

        if config.RENDER:
            self.rendering = Rendering(self)

        return self._get_state(), {'default_action': self.default_action, 'mask': self.get_valid_action_mask()}

    def step(self, action: np.ndarray | list, debugmode: bool = False) -> [dict, float, bool, dict]:
        """
        Take a step in the environment. Update state of environment and return new state, reward, done and info
        :param action: action to take. Note that action will be ignored if elevator is not in need of action
        :param debugmode: if True, no new passengers are generated
        :return: state, reward, done, info
        """

        self.reward = 0.0
        self.exiting_passengers = []

        self._take_elevator_step(action)

        self.timecounter += 1
        # every 20 steps, update time of day, loop through 3 times
        if self.timecounter % 50 == 0:
            self.time_of_day = (self.time_of_day + 1) % 3  # TODO: make real time of day

        # add rewards for waiting passengers
        self._add_wait_rewards()

        # add rewards to dataclass
        self.episode_data.add_reward(self.reward)

        # increment waiting time for all waiting passengers
        for passenger in self.floor_passengers:
            passenger.increment_time_waited()
        for elevator in self.elevators:
            for passenger in elevator.passengers:
                passenger.increment_time_travelled()

        # generate new passengers
        if not debugmode:
            self._generate_passengers()

        # update buttons
        self._update_buttons()
        self._update_button_times()

        # check if env needs action from agent for next step
        self.default_action = self._get_default_action()

        state = self._get_state()

        if config.RENDER:
            self.rendering.render()

        return state, self.reward, False, {'default_action': self.default_action, 'mask': self.get_valid_action_mask()}

    def _take_elevator_step(self, action):
        """
        Take one step for each elevator. Actually performs most of the environment logic per step.
        Updates position of elevator, loads and unloads passengers, adds penalty for movement to reward and
        loading+unloading of passengers.
        :param action: list of actions for each elevator
        :return: None
        """

        # randomize order of elevators
        order = np.random.permutation(self.n_elev)
        for i in order:

            # if going up
            if action[i] == vm.ACTION_UP:
                if self.elevators[i].position + 1 > self.n_floors:
                    raise ValueError(f'elevator {i} cannot go up above top floor. '
                                     'Action mask should not allow this action')
                elif self.elevators[i].direction == vm.DIR_DOWN:
                    raise ValueError(f'elevator {i} cannot go up before stopping'
                                     'Action mask should not allow this action')
                self.elevators[i].position += 1
                self.elevators[i].direction = vm.DIR_UP
                # add penalty for movement to reward
                self.reward += self.movement_penalty

            # if going down
            elif action[i] == vm.ACTION_DOWN:
                if self.elevators[i].position - 1 < 0:
                    raise ValueError(f'elevator {i} cannot go down below ground floor'
                                     'Action mask should not allow this action')
                elif self.elevators[i].direction == vm.DIR_UP:
                    raise ValueError(f'elevator {i} cannot go down before stopping'
                                     'Action mask should not allow this action')
                self.elevators[i].position -= 1
                self.elevators[i].direction = vm.DIR_DOWN
                # add penalty for movement
                self.reward += self.movement_penalty

            # if stay
            elif action[i] == vm.ACTION_STAY:
                self.elevators[i].direction = vm.DIR_STAY
                self.floor_passengers, exited_passengers, entered_passengers = self.elevators[i].load_unload(
                    self.floor_passengers, self.episode_data)
                self.reward += self.arrival_reward * exited_passengers
                self.reward += self.loading_reward * entered_passengers
                # keep track of where passengers exited and how many
                self.exiting_passengers.extend([self.elevators[i].position] * exited_passengers)
                # reset floor button time
                # no need to update elevator button time, is taken care of in _update_button_times()
                if self.elevators[i].indicator == 1:
                    self.floor_button_times[self.elevators[i].position, 0] = 0
                elif self.elevators[i].indicator == -1:
                    self.floor_button_times[self.elevators[i].position, 1] = 0

            else:
                raise ValueError('action must be 0, 1 or 2')

        assert [0 <= self.elevators[i].position <= self.n_floors for i in range(self.n_elev)]

    def _get_state(self) -> dict[str, int | np.ndarray]:
        """
        Return state of environment
        :return: state of environment composed of position of elevators, elevator buttons, floor buttons and time of day
        """

        state = {'elev_position': np.array([elevator.position for elevator in self.elevators]),
                 'elev_direction': np.array([elevator.direction for elevator in self.elevators]),
                 'elev_button': self.elev_buttons,
                 'floor_button': self.floor_buttons,
                 'time_of_day': np.array([self.time_of_day])}

        return state

    def _add_wait_rewards(self):
        """
        Add rewards for waiting passengers at floor and in elevator.
        """

        if config.REWARD_SCHEME == 'PASSENGER':
            # waiting at floor
            self.reward += self.waiting_penalty * len(self.floor_passengers)
            # waiting in elevator
            for elevator in self.elevators:
                self.reward += self.waiting_penalty * len(elevator.passengers)
        # alternative: sum of elevator buttons pressed
        elif config.REWARD_SCHEME == 'SUM':
            self.reward += self.waiting_penalty * sum(sum(self.floor_buttons))
            self.reward += self.waiting_penalty * sum(sum(self.elev_buttons))
        elif config.REWARD_SCHEME == 'TIME':
            self.reward += sum(sum(self.waiting_penalty * self.floor_button_times))
            self.reward += sum(sum(self.waiting_penalty * self.elev_button_times))
        elif config.REWARD_SCHEME == 'SQUARED':
            self.reward += sum(sum(self.waiting_penalty * self.floor_button_times ** 2))
            self.reward += sum(sum(self.waiting_penalty * self.elev_button_times ** 2))
        else:
            raise ValueError('Reward scheme not recognized')

    def _generate_passengers(self):
        """
        Generate new passengers for each floor according to arrival function.
        """

        for floor in range(self.n_floors):
            arrivals = self._arrivals_function(floor)  # determine amount of arrivals for that floor
            for i in range(arrivals):
                destination = np.random.randint(0, self.n_floors)  # assign destination
                # TODO: make probability matrix of destination, possibly dependent on time of day
                while destination == floor:
                    destination = np.random.randint(0, self.n_floors)  # cant have same floor as departure
                self.floor_passengers.append(Passenger(floor, destination))

    def _arrivals_function(self, floor: int) -> int:

        """
        Returns the number of arrivals for a given time bin and floor.
        Based on Poisson distribution with lambda depending on time of day and floor.
        :param floor: floor number
        """

        # time_of_day: 0 = morning, 1 = afternoon, 2 = evening (for now)
        # use time_of_day to determine lambda for Poisson distribution
        # morning = up-peak, evening = down-peak, afternoon = regular

        if self.time_of_day == 0:  # up-peak
            if floor == 0:
                lambda_ = 0.4  # high up-traffic
            else:
                lambda_ = 0.02  # low down-traffic

        elif self.time_of_day == 2:  # down-peak
            if floor > 0:
                lambda_ = ((floor / self.n_floors) + 1) / 10  # ranges from 1 to 2 = high down-traffic
            else:
                lambda_ = 0.02  # low up-traffic

        else:
            lambda_ = 0.1

        # TODO: sample from polya-aeppli distribution
        return np.random.poisson(lambda_)

    def _update_buttons(self):

        # update buttons on floors
        self.floor_buttons = np.zeros((self.n_floors, 2))  # 2: up, down
        for passenger in self.floor_passengers:
            if passenger.destination > passenger.floor:
                self.floor_buttons[passenger.floor, 0] = 1
            elif passenger.destination < passenger.floor:
                self.floor_buttons[passenger.floor, 1] = 1

        # update buttons in elevators
        self.elev_buttons = np.zeros((self.n_elev, self.n_floors))
        for i, elevator in enumerate(self.elevators):
            for passenger in elevator.passengers:
                self.elev_buttons[i, passenger.destination] = 1

    def _update_button_times(self):
        """
        For every button, update time since button was pressed.
        """
        for floor in range(self.n_floors):
            if any([passenger.floor == floor and passenger.destination > floor for passenger in self.floor_passengers]):
                self.floor_button_times[floor, 0] += 1
            else:
                # if no one is waiting to go up, reset time
                self.floor_button_times[floor, 0] = 0

            if any([passenger.floor == floor and passenger.destination < floor for passenger in self.floor_passengers]):
                self.floor_button_times[floor, 1] += 1
            else:
                # if no one is waiting to go down, reset time
                self.floor_button_times[floor, 1] = 0

            for i, elevator in enumerate(self.elevators):
                if any([passenger.destination == floor for passenger in elevator.passengers]):
                    self.elev_button_times[i, floor] += 1
                else:
                    # if no one is waiting to get off, reset time
                    self.elev_button_times[i, floor] = 0

    def _get_default_action(self) -> list:
        """
        Determine if agent needs to take action for each elevator in the next step.
        If the elevator had no choice of action, it does not need input from agent which it signals
        by returning False. If the elevator needs a new action, it returns True.
        :return: list of booleans indicating if agent needs to take action for each elevator
        """

        default_action = [None] * self.n_elev

        for i in range(self.n_elev):

            # if passengers want to go up and elevator is stopped, must go up
            if self.elevators[i].direction == vm.DIR_STAY and self.elevators[i].indicator == vm.DIR_UP:
                default_action[i] = vm.ACTION_UP

            # if passengers want to go down and elevator is stopped, must go down
            if self.elevators[i].direction == vm.DIR_STAY and self.elevators[i].indicator == vm.DIR_DOWN:
                default_action[i] = vm.ACTION_DOWN

            # if elevator is at floor where passengers want to go, must stop
            if any(passenger.destination == self.elevators[i].position for passenger in self.elevators[i].passengers):
                default_action[i] = vm.ACTION_STAY

        return default_action

    def get_valid_action_mask(self) -> np.ndarray:
        """"
        Returns a mask of valid actions for each elevator. Used for determining which actions to take but also
        for only considering valid actions in the Q-function. (max(Q_next))
        For more info check out https://boring-guy.sh/posts/masking-rl/
        :return: mask of valid actions
        """

        mask = np.ones(shape=[3, self.n_elev])

        # reminder of action space:
        # 0: go up
        # 1: go down
        # 2: do nothing

        for elevator in range(self.n_elev):
            # cannot go down if on bottom floor
            if self.elevators[elevator].position == 0:
                mask[1, elevator] = 0
            # cannot go up if on top floor
            if self.elevators[elevator].position == self.n_floors - 1:
                mask[0, elevator] = 0
            # cannot go down if going up
            if self.elevators[elevator].direction == vm.DIR_UP:
                mask[1, elevator] = 0
            # cannot go up if going down
            if self.elevators[elevator].direction == vm.DIR_DOWN:
                mask[0, elevator] = 0

        return mask

    def sample_valid_action(self) -> np.ndarray:
        """
        Randomly samples a valid action for each elevator.
        :return: array of actions, one for each elevator
        """
        mask = self.get_valid_action_mask()
        # sample a non-zero index from each column
        sampled = np.array([np.random.choice(np.nonzero(mask[:, i])[0]) for i in range(self.n_elev)])

        return sampled
    