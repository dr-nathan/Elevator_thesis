import json
import logging
import os
import random
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from agent.continuous_agents.neural_nets import CombinatorialQ, DuelingCombinatorialQ, BranchingQ, ZoningSequentialQ, \
    ZoningBranchingQ
from agent.continuous_agents.RuleBasedAgents import ETDAgent

logger = logging.getLogger(__name__)


class ExperienceReplay:

    def __init__(self, buffer_size: int, min_replay_size: int, agent_: 'ETDAgent' or 'RLAgent'):

        """
        Object that stores the experience replay buffer and samples random transitions from it.
        :param agent_: agent that needs to play in order to fill the replay memory
        :param buffer_size = max number of transitions that the experience replay buffer can store
        :param min_replay_size = min number of (random) transitions that the replay buffer needs
        to have when initialized
        """

        self.learn_zoning = config.LEARN_ZONING
        self.agent = agent_
        self.min_replay_size = min_replay_size
        self.replay_buffer_assign = deque(maxlen=buffer_size)
        if self.learn_zoning:
            self.replay_buffer_zone = deque(maxlen=buffer_size)

        self.pending_step_duration_assign = 0
        self.pending_step_duration_zone = 0
        self.acc_assign_reward = 0
        self.acc_zone_reward = 0
        self.pending_assign_transition = None
        self.pending_zone_transition = None

        self.fill_replay_memory()

    def fill_replay_memory(self):
        """
        Fills the replay memory with random transitions, according to self.min_replay_size
        """

        if not self.learn_zoning:  # only need to fill assign buffer
            print('Filling assigning agent replay memory...')

            state, info = self.agent.env.reset()
            state = self.agent.state_to_tensor(state)

            for _ in tqdm(range(self.min_replay_size)):
                # choose random action, no NN yet
                action = self.agent.choose_action(state, self.agent.env, policy="random")
                next_state, reward, terminated, info = self.agent.env.step(action)
                next_state = self.agent.state_to_tensor(next_state)
                action_duration = info['step_length']

                # skip zoning decisions
                while 'zoning' in info:
                    zone = self.agent.choose_zone(next_state, self.agent.env, policy="random")
                    next_state, rew, terminated, info = self.agent.env.step(zone)
                    next_state = self.agent.state_to_tensor(next_state)
                    reward += rew
                    action_duration += info['step_length']

                if action_duration == 0:
                    logger.info(f'{self.agent.env.time}: step duration is 0, setting to 1')
                    action_duration = 1

                transition = (state, action, reward, terminated, action_duration, next_state)
                self.replay_buffer_assign.append(transition)
                state = next_state

                if terminated:
                    state, info = self.agent.env.reset()
                    state = self.agent.state_to_tensor(state)

        # fill both buffers
        else:

            print('Filling assigning+zoning agent replay memory...')

            state, info = self.agent.env.reset()
            state = self.agent.state_to_tensor(state)

            for _ in tqdm(range(self.min_replay_size)):
                if 'zoning' in info:

                    # this state is the next state of the previous transition, append and add to buffer
                    if self.pending_zone_transition is not None:
                        # replace reward in pending transition with accumulated reward over 30 mins inter-zone time
                        self.pending_zone_transition[2] = self.acc_zone_reward
                        self.pending_zone_transition[4] = self.pending_step_duration_zone
                        # append current state as the next state of pending transition
                        self.pending_zone_transition.append(state)
                        transition = tuple(self.pending_zone_transition)
                        # and add to buffer
                        self.replay_buffer_zone.append(transition)

                    # start new transition
                    # take the step, get reward + new state
                    zone = self.agent.choose_zone(state, self.agent.env, "random")
                    next_state, reward, terminated, info = self.agent.env.step(zone)
                    next_state = self.agent.state_to_tensor(next_state)
                    # start new pending transition
                    self.acc_zone_reward = reward
                    self.pending_step_duration_zone = info['step_length']
                    self.pending_zone_transition = [state, zone, reward, terminated, info['step_length']]

                    # accumulate reward and duration for assign, as this is between 2 assign decisions
                    self.acc_assign_reward += reward
                    self.pending_step_duration_assign += info['step_length']

                else:

                    if self.pending_assign_transition is not None:
                        # replace reward and duration in pending transition
                        self.pending_assign_transition[2] = self.acc_assign_reward
                        self.pending_assign_transition[4] = self.pending_step_duration_assign
                        # append current state as the next state of pending transition
                        self.pending_assign_transition.append(state)
                        transition = tuple(self.pending_assign_transition)
                        # and add to buffer
                        self.replay_buffer_assign.append(transition)

                    # start new transition
                    action = self.agent.choose_action(state, self.agent.env, "random")
                    next_state, reward, terminated, info = self.agent.env.step(action)
                    next_state = self.agent.state_to_tensor(next_state)
                    # new pending transition
                    self.acc_assign_reward = reward
                    self.pending_step_duration_assign = info['step_length']
                    self.pending_assign_transition = [state, action, reward, terminated, info['step_length']]

                    # incremement zoning data
                    self.acc_zone_reward += reward
                    self.pending_step_duration_zone += info['step_length']

                state = next_state

                if terminated:
                    state, info = self.agent.env.reset()
                    state = self.agent.state_to_tensor(state)

                    # reset pending transitions
                    self.acc_zone_reward = 0
                    self.acc_assign_reward = 0
                    self.pending_step_duration_assign = 0
                    self.pending_step_duration_zone = 0
                    self.pending_assign_transition = None
                    self.pending_zone_transition = None

            # reset pending transitions (reused in training loop)
            self.acc_zone_reward = 0
            self.acc_assign_reward = 0
            self.pending_step_duration_assign = 0
            self.pending_step_duration_zone = 0
            self.pending_assign_transition = None
            self.pending_zone_transition = None

            tqdm.write(f"Length of replay buffer assign: {len(self.replay_buffer_assign)}")
            tqdm.write(f"Length of replay buffer zone: {len(self.replay_buffer_zone)}")

    def sample(self, batch_size: int, zoning: bool = False):

        # sample random transitions from the replay memory
        if zoning:
            transitions = random.sample(self.replay_buffer_zone, batch_size)
        else:
            transitions = random.sample(self.replay_buffer_assign, batch_size)

        # convert to array where needed, then to tensor (faster than directly to tensor)
        observations = np.asarray([t[0] for t in transitions], dtype=float)
        actions = np.asarray([t[1] for t in transitions], dtype=int)
        rewards = np.asarray([t[2] for t in transitions], dtype=float)
        dones = np.asarray([t[3] for t in transitions], dtype=int)
        durations = np.asarray([t[4] for t in transitions], dtype=float)
        new_observations = np.asarray([t[5] for t in transitions], dtype=float)

        # convert to tensors
        observations_t = torch.as_tensor(observations, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.int64)
        durations_t = torch.as_tensor(durations, dtype=torch.int64)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32)

        return observations_t, actions_t, rewards_t, dones_t, durations_t, new_observations_t

    def get_mean_and_std(self, zoning: bool = False):
        """
        Get mean and std of the states in the replay buffer. Called once after initially filling the buffer.
        Used to normalize the states. Save this mean and std for use during validation and test,
        as we have no access to that data then.
        """

        if zoning:
            buffer = self.replay_buffer_zone
        else:
            buffer = self.replay_buffer_assign

        states = []

        for exp in buffer:
            state, *_ = exp
            states.append(state)
        np.asarray(states)
        mean = torch.as_tensor(np.mean(states, axis=0), dtype=torch.float32)
        std = torch.as_tensor(np.std(states, axis=0), dtype=torch.float32)

        return mean, std


class RLAgent(object):

    def __init__(
            self,
            env: 'DiscreteEvent',
            train_env: 'DiscreteEvent | None' = None,
            val_env: 'DiscreteEvent | None' = None,
            training: bool = True,
            nn_type_assign: str = None,
            nn_type_zone: str = None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epsilon: float = 1.0,
            epsilon_decay: bool = True,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            n_episodes: int = 20,
            lr: float = 5e-4,
            buffer_size: int = 10000,
    ):

        super().__init__()

        # set/reset variables
        self.training = training
        self.debug_data = defaultdict(list)
        self.nn_type_assign = nn_type_assign
        self.nn_type_zone = nn_type_zone
        self.env = env
        self.train_env = train_env
        self.val_env = val_env
        self.device = device
        self.n_episodes = n_episodes
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.best_agent = None
        self.current_train_rew = None

        self.normalize_state = config.NORMALIZE_STATE
        self.normalize_reward = config.NORMALIZE_REWARD
        self.learn_zoning = config.LEARN_ZONING
        self.discount_rate = config.DISCOUNT_FACTOR

        # set epsilon decay for epsilon greedy policy
        if epsilon_decay:
            self.epsilon = epsilon_start
            # log decay
            self.epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / self.n_episodes
            )
        else:
            self.epsilon = epsilon
            self.epsilon_decay_step = 1.0

        # get state shape for assigning decisions and zoning decisions (separate NNs, separate state shapes)
        state_, _ = self.env.reset()
        self.inp_size_assign = self.state_to_tensor(state_).shape[0]
        if self.learn_zoning:
            state_, _ = self.move_to_first_zoning_decision()
            self.inp_size_zone = state_.shape[0]

        # set min_reply_size equal to buffer_size: fill up the buffer before training
        if self.training:
            agent_repl = ETDAgent(self.env) if config.FILL_WITH_BASELINE else self
            self.replay_memory = ExperienceReplay(
                buffer_size=self.buffer_size, min_replay_size=self.buffer_size, agent_=agent_repl)

            self.mean_assign, self.std_assign = self.replay_memory.get_mean_and_std(zoning=False)
            if self.learn_zoning:
                self.mean_zone, self.std_zone = self.replay_memory.get_mean_and_std(zoning=True)
            # if not training, fetch mean and std from file later in self.load()

            # copy pending transition variables from replay memory, rename for clarity
            # We dont actually use the data but just need the data structure
            self.acc_zone_reward = self.replay_memory.acc_zone_reward
            self.acc_assign_reward = self.replay_memory.acc_assign_reward
            self.pending_zone_transition = self.replay_memory.pending_zone_transition
            self.pending_assign_transition = self.replay_memory.pending_assign_transition
            self.pending_step_duration_assign = self.replay_memory.pending_step_duration_assign
            self.pending_step_duration_zone = self.replay_memory.pending_step_duration_zone

        # get q networks
        self.online_network_assign, self.target_network_assign, self.online_network_zone, self.target_network_zone = \
            self.get_q_networks()

    def __repr__(self):
        return f'RL {self.nn_type_assign} Agent. Zoning: {self.learn_zoning}'

    def get_q_networks(self):

        match self.nn_type_assign:
            case 'comb':
                online_network_assign = CombinatorialQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
                target_network_assign = CombinatorialQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
            case 'duel_comb':
                online_network_assign = DuelingCombinatorialQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
                target_network_assign = DuelingCombinatorialQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
            case 'branch':
                online_network_assign = BranchingQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
                target_network_assign = BranchingQ(
                    self.env, self.learning_rate, self.inp_size_assign, self.n_episodes).to(self.device)
            case _:
                raise ValueError(f'Unknown NN type {self.nn_type_assign}')

        # copy weights from online to target network
        target_network_assign.load_state_dict(online_network_assign.state_dict())

        # also load zoning networks if needed
        if self.learn_zoning:

            match self.nn_type_zone:
                case 'sequential':
                    online_network_zone = ZoningSequentialQ(
                        self.env, self.learning_rate, self.inp_size_zone, self.n_episodes).to(self.device)
                    target_network_zone = ZoningSequentialQ(
                        self.env, self.learning_rate, self.inp_size_zone, self.n_episodes).to(self.device)
                case 'branching':
                    online_network_zone = ZoningBranchingQ(
                        self.env, self.learning_rate, self.inp_size_zone, self.n_episodes).to(self.device)
                    target_network_zone = ZoningBranchingQ(
                        self.env, self.learning_rate, self.inp_size_zone, self.n_episodes).to(self.device)
                case _:
                    raise ValueError(f'Unknown NN type {self.nn_type_zone}')

            target_network_zone.load_state_dict(online_network_zone.state_dict())

            return online_network_assign, target_network_assign, online_network_zone, target_network_zone

        else:
            return online_network_assign, target_network_assign, None, None

    def move_to_first_zoning_decision(self):

        state, info = self.env.reset()
        state = self.state_to_tensor(state)

        # move to first zoning decision
        while 'zoning' not in info:
            action = self.choose_action(state, self.env, policy="random")
            state, reward, terminated, info = self.env.step(action)
            state = self.state_to_tensor(state)

        return state, info

    def act(self, state_, env_=None, info_=None, record_loss=False) -> np.ndarray:
        """
        Choose an action according to the greedy policy. Meant to be used outside of training
        :param state_: current state
        :param env_: environment (as param, cuz it can be different from self.env, when validating for example)
        :param info_: info dict from environment
        :param record_loss: whether to record the loss of the network
        :return: action
        """

        env_ = env_ if env_ is not None else self.env  # allows to use different envs for training and validation
        state_ = self.state_to_tensor(state_)

        if 'zoning' in info_:
            return self.choose_zone(state_, env_, "greedy")
        else:
            return self.choose_action(state_, env_, "greedy", record_loss=record_loss)

    def choose_action(self, state: torch.Tensor, _env: 'DiscreteEvent', policy: str, record_loss=False) -> np.ndarray:
        """
        Choose an action according to the given policy
        :param state: current state
        :param _env: environment. Used to sample a random action and to get the valid action mask
        :param policy: policy to use. Can be "random", "greedy" or "epsilon_greedy"
        :param record_loss: whether to record the loss of the network
        :return: action per elevator (np.ndarray of 1 or 0)
        """

        if policy == "random":
            # between 1 and 3 elevators responding
            repeats = random.randint(1, config.MAX_ELEVS_RESPONDING)
            # choose random elevators
            action = np.random.choice(_env.n_elev, repeats, replace=False)

            return np.bincount(action, minlength=_env.n_elev)

        elif policy == "greedy":

            # normalize state
            if self.normalize_state:
                state = (state - self.mean_assign) / (self.std_assign + 1e-9)

            with torch.no_grad():
                res = self.online_network_assign.forward(state)

                if record_loss:
                    target = self.target_network_assign.forward(state)
                    loss = torch.nn.functional.mse_loss(res, target) if config.LOSS_FN == 'MSE' else \
                        torch.nn.functional.smooth_l1_loss(res, target) if config.LOSS_FN == 'Huber' else None
                    self.debug_data['val_loss'][-1].append(loss)

            match self.nn_type_assign:
                case 'comb' | 'duel_comb':
                    # get action corresponding to max q value
                    action = self.online_network_assign.combinations[res.argmax().numpy()]
                case 'branch':
                    # choose max sub-action per action (assign/not assign)
                    action = res.argmax(dim=1).numpy()  # TODO: check dim
                case _:
                    raise ValueError(f'Unknown NN type {self.nn_type_assign}')

            return action

        elif policy == "epsilon_greedy":

            if random.random() < self.epsilon:
                return self.choose_action(state, _env, "random")
            else:
                return self.choose_action(state, _env, "greedy")

        raise ValueError("Unknown policy")

    def choose_zone(self, state: torch.Tensor, _env: 'DiscreteEvent', policy: str) -> np.ndarray:
        """
        Choose a zone for each elevator according to the given policy
        :param state: current state
        :param _env: environment. Used to sample a random action and to get the valid action mask
        :param policy: policy to use. Can be "random", "greedy" or "epsilon_greedy"
        :return: zone per elevator
        """

        if policy == "random":

            # choose random floor for every elevator. Can also choose None, which means no zoning decision
            actions = np.random.choice(_env.n_floors + 1, _env.n_elev, replace=True)

            return actions

        elif policy == "greedy":

            if self.nn_type_zone == 'sequential':
                # don't normalize state, as it is variable trough elevators. Wouldn't work
                actions = []
                for elevator in range(_env.n_elev):
                    with torch.no_grad():
                        # set elev indicator to elevator index
                        state[-1] = elevator + 1  # 1 to 6 (indicator variable)
                        res = self.online_network_zone.forward(state)
                        action = res.argmax().detach()
                        actions.append(action)
                        # input decision in state (not for last elevator)
                        if elevator < _env.n_elev - 1:
                            state[-6 + elevator] = action

                return np.asarray(actions)

            elif self.nn_type_zone == 'branching':
                # normalize state
                if self.normalize_state:
                    state = (state - self.mean_zone) / (self.std_zone + 1e-9)
                with torch.no_grad():
                    res = self.online_network_zone.forward(state)
                    # choose max sub-action per action (floor/zone)
                    actions = res.argmax(dim=1).numpy()

                return actions

        elif policy == "epsilon_greedy":

            if random.random() < self.epsilon:
                return self.choose_zone(state, _env, "random")
            else:
                return self.choose_zone(state, _env, "greedy")

        raise ValueError("Unknown policy")

    def save_better_model(self, save_path):

        # if learning zoning, save both networks
        if self.learn_zoning:

            torch.save(self.online_network_zone.state_dict(), str(save_path) + '/online_network_zone.pt')
            torch.save(self.online_network_assign.state_dict(), str(save_path) + '/online_network_assign.pt')

        else:

            torch.save(self.online_network_assign.state_dict(), str(save_path) + '/online_network_assign.pt')

    def training_loop(self, batch_size: int, save_path: Path):
        """
        Main training loop
        :param batch_size: number of transitions that will be sampled from the replay memory
        :param save_path: path where the best model will be saved
        :return: train and validation rewards
        """
        state, info = self.env.reset()
        state = self.state_to_tensor(state)

        _train_rewards = []
        _val_rewards = []
        _wait_times_train = []
        _wait_times_val = []
        _energy_cons_train = []
        _energy_cons_val = []

        tqdm.write(f"Training {self.nn_type_assign}({config.NN_ASSIGN_AGG})"
                   f"{'/' + self.nn_type_zone if self.learn_zoning else ''} "
                   f"agent for {self.n_episodes} episodes...")

        for iteration in tqdm(range(self.n_episodes), colour='green'):

            # play one move, add the transition to the replay memory
            state, info, *_ = self.take_step_training(state, info)

            # decay epsilon
            self.epsilon *= self.epsilon_decay_step
            self.debug_data['epsilon'].append(self.epsilon)

            if (iteration + 1) % config.LEARN_INTERVAL == 0:
                # sample a batch of transitions from the replay memory, forward-backward pass on online network
                self.learn_wrapper(batch_size=batch_size)

            # every 300 iterations, update the target network
            if (iteration + 1) % config.TARGET_NETWORK_UPDATE_INTERVAL == 0:
                self.target_network_assign.load_state_dict(self.online_network_assign.state_dict())

            # get some statistics, 30 times during training
            if (iteration + 1) % (self.n_episodes // 30) == 0:
                reward_train, wait_time_train, energy_cons_train = self.validate(val=False)
                reward_val, wait_time_val, energy_cons_val = self.validate(val=True)
                if args.from_terminal:
                    tqdm.write(f"Train reward: {reward_train:.2f}, val reward: {reward_val:.2f}")
                    tqdm.write(f"Train wait time: {wait_time_train:.2f}, val wait time: {wait_time_val:.2f}")

                # Save the model if the validation reward is the best so far
                if _val_rewards == [] or reward_val > max(_val_rewards):
                    self.save_better_model(save_path)
                    self.best_agent = iteration

                # save the rewards, wait times and energy consumption
                _train_rewards.append(reward_train)
                _val_rewards.append(reward_val)
                _wait_times_train.append(wait_time_train)
                _wait_times_val.append(wait_time_val)
                _energy_cons_train.append(energy_cons_train)
                _energy_cons_val.append(energy_cons_val)

        # filter out private variables for readability
        configfile = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        # save config file with json module
        with open(str(save_path) + '/config.txt', 'w') as f:
            json.dump(configfile, f, indent=4)

        # save mean and std to agent map for validation and test
        torch.save(self.mean_assign, Path(save_path) / 'mean_assign.pt')
        torch.save(self.std_assign, Path(save_path) / 'std_assign.pt')
        if self.learn_zoning:
            torch.save(self.mean_zone, Path(save_path) / 'mean_zone.pt')
            torch.save(self.std_zone, Path(save_path) / 'std_zone.pt')

        print(f"Best agent at iteration {self.best_agent} of {self.n_episodes}\n"
              f"saved at {save_path}")

        self.training_plots(_train_rewards, _val_rewards, _wait_times_train, _wait_times_val,
                            _energy_cons_train, _energy_cons_val, save_path)

        return _train_rewards, _val_rewards, _wait_times_train, _wait_times_val, _energy_cons_train, _energy_cons_val

    def take_step_training(self, state: torch.Tensor, info: dict) -> (torch.Tensor, dict, int, float):

        """
        Take one step in the environment, and add the transition to the replay memory. Always choose epsilon-greedy
        :param state: current state
        :param info: info dict from environment
        :return: next state, action, reward
        """

        # NOTE: 'zoning' never shows up in info if config.learn_zoning is False
        if 'zoning' in info:

            if self.pending_zone_transition is not None:
                # replace reward in pending transition with accumulated reward over 30 mins inter-zone time
                self.pending_zone_transition[2] = self.acc_zone_reward
                self.pending_zone_transition[4] = self.pending_step_duration_zone
                # append current state as the next state of pending transition
                self.pending_zone_transition.append(state)
                transition = tuple(self.pending_zone_transition)
                # and add to buffer
                self.replay_memory.replay_buffer_zone.append(transition)

            # start new transition
            action = self.choose_zone(state, self.env, "epsilon_greedy")
            next_state, reward, terminated, info = self.env.step(action)
            next_state = self.state_to_tensor(next_state)
            self.acc_zone_reward = reward
            self.pending_step_duration_zone = info['step_length']
            self.pending_zone_transition = [state, action, reward, terminated, info['step_length']]

            # increment assign data
            self.acc_assign_reward += reward
            self.pending_step_duration_assign += info['step_length']

        else:

            if self.pending_assign_transition is not None:
                # replace reward and duration in pending transition
                self.pending_assign_transition[2] = self.acc_assign_reward
                if self.pending_step_duration_assign == 0:
                    logger.warning(f'{self.env.time}: Step duration is 0, setting to 1')
                    self.pending_step_duration_assign = 1
                self.pending_assign_transition[4] = self.pending_step_duration_assign
                # append current state as the next state of pending transition
                self.pending_assign_transition.append(state)
                transition = tuple(self.pending_assign_transition)
                # and add to buffer
                self.replay_memory.replay_buffer_assign.append(transition)

            # start new transition
            action = self.choose_action(state, self.env, "epsilon_greedy")
            # record how many elevs were assigned
            self.debug_data['n_elevs_assigned'].append(action.sum())
            next_state, reward, terminated, info = self.env.step(action)
            next_state = self.state_to_tensor(next_state)
            self.acc_assign_reward = reward
            self.pending_step_duration_assign = info['step_length']
            self.pending_assign_transition = [state, action, reward, terminated, info['step_length']]

            # incremement zoning data
            self.acc_zone_reward += reward
            self.pending_step_duration_zone += info['step_length']

        state = next_state

        if terminated:
            state, info = self.env.reset()
            state = self.state_to_tensor(state)

            # reset pending transitions
            self.acc_zone_reward = 0
            self.acc_assign_reward = 0
            self.pending_step_duration_assign = 0
            self.pending_step_duration_zone = 0
            self.pending_assign_transition = None
            self.pending_zone_transition = None

        return state, info, action, reward

    def learn_wrapper(self, batch_size: int):

        """ Wrapper allows to learn twice, once for assigning network, and optionally for zoning network """

        # sampler also takes care of converting to tensors
        (observations_t,
         actions_t,
         rewards_t,
         dones_t,
         durations_t,
         new_observations_t) = self.replay_memory.sample(batch_size, zoning=False)
        online_net = self.online_network_assign
        target_net = self.target_network_assign
        mean = self.mean_assign
        std = self.std_assign

        self.learn(observations_t,
                   actions_t,
                   rewards_t,
                   dones_t,
                   durations_t,
                   new_observations_t,
                   online_net,
                   target_net,
                   mean,
                   std,
                   zoning=False)

        if self.learn_zoning:
            (observations_t,
             actions_t,
             rewards_t,
             dones_t,
             durations_t,
             new_observations_t) = self.replay_memory.sample(batch_size, zoning=True)
            online_net = self.online_network_zone
            target_net = self.target_network_zone
            mean = self.mean_zone
            std = self.std_zone

            self.learn(observations_t,
                       actions_t,
                       rewards_t,
                       dones_t,
                       durations_t,
                       new_observations_t,
                       online_net,
                       target_net,
                       mean,
                       std,
                       zoning=True)

    def learn(self,
              observations_t,
              actions_t,
              rewards_t,
              dones_t,
              durations_t,
              new_observations_t,
              online_net,
              target_net,
              mean,
              std,
              zoning: bool = False):

        """
        Sample a batch of transitions from the replay memory, and update the online network
        """

        # set networks to appropriate mode
        target_net.eval()
        online_net.train()

        # sequential cant handle normalized state data (state is dynamic through elevators)
        if not (zoning and self.nn_type_zone == 'sequential') and self.normalize_state:
            # use mean and std from replay memory
            observations_t = (observations_t - mean) / (std + 1e-9)
            new_observations_t = (new_observations_t - mean) / (std + 1e-9)

        if self.normalize_reward:
            # batch-normalize rewards
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-9)

        # compute targets and q values for zoning Q network, if relevant #
        if zoning:

            # Zoning decisions can be understood as a multi-armed bandit problem, as we must choose the best
            # zoning strategy for the next 30 mins, but the next state does not really matter.
            # So, we decide that the targets are only the rewards over the next 30 mins,
            # aka. we set the discount rate to 0. The difference with a multi-armed bandit problem is that
            # we need to take into account the inter-dependence between actions (action of an elevator
            # affects the others). So we use a RL framework, but with a discount rate of 0.

            # forward on observations
            if self.nn_type_zone == 'sequential':

                # make sequential states
                # train sequentially, from elev 1 to 6
                states_t = torch.zeros((6, *observations_t.shape))
                # add actions taken to observations but not the last elev
                for elev in range(6):
                    # observations_t.shape = torch.Size([batch, state_size])
                    observations_t[:, -7:-7 + elev] = actions_t[:, :-6 + elev]
                    # add elevator index to state (last one)
                    observations_t[:, -1] = elev
                    # must save states for each elevator, as they are different. Needed when doing backprop
                    states_t[elev] = observations_t.detach()

                # with now built successive states, can process all elevators at once
                zone_q_values_all = online_net.forward(states_t)

                # get chosen zone for each elevator
                chosen_zones = torch.gather(input=zone_q_values_all, dim=-1,
                                            index=actions_t.transpose(0, 1).unsqueeze(-1)).squeeze(-1)

                # aggregate over elevators (combine q value of every elevator into one q value)
                if config.NN_ZONING_AGG == 'sum':
                    action_q_values = chosen_zones.sum(dim=0)
                elif config.NN_ZONING_AGG == 'mean':
                    action_q_values = chosen_zones.mean(dim=0)
                # config.NN_ZONING_AGG == 'none' makes no sense here as the targets are simply the rewards.
                # We are therefore trying to predict the rewards, which can most simply be considered to be either
                # the mean or sum of the q values per elevator. Selecting none would mean every branch tried to
                # predict the same reward independently
                else:
                    raise ValueError(f'Unknown zoning aggregation method {config.NN_ZONING_AGG}')

                # compute targets (no discount, solely reward over 30 mins inter-zone time, next state does not matter)
                targets = rewards_t

            elif self.nn_type_zone == 'branching':

                # compute q values #
                # forward on observations
                q_values = online_net.forward(observations_t)
                # collect q values of actions taken
                action_q_values = torch.gather(input=q_values, dim=-1,
                                               index=actions_t.unsqueeze(-1)).squeeze(-1)

                # sum over separate action branches -> unique Q value for each state
                if config.NN_ZONING_AGG == 'sum':
                    action_q_values = action_q_values.sum(dim=1)
                elif config.NN_ZONING_AGG == 'mean':
                    action_q_values = action_q_values.mean(dim=1)
                else:
                    raise ValueError(f'Unknown zoning aggregation method {config.NN_ZONING_AGG}')

                # discount is 0 -> targets are rewards
                targets = rewards_t

            else:
                raise ValueError(f'Unknown NN type {self.nn_type_zone}')

        elif self.nn_type_assign == 'comb' or self.nn_type_assign == 'duel_comb':

            # compute targets #

            with torch.no_grad():
                target_q_values = target_net.forward(new_observations_t)

            # for comb, simply get max q value of next states
            max_target_q_values = target_q_values.max(dim=1)[0]  # TODO: check dim
            # Bellman equation
            if config.DISCOUNTING_SCHEME == 'fixed':
                targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values
            elif config.DISCOUNTING_SCHEME == 'variable':
                targets = rewards_t + (self.discount_rate ** durations_t) * (1 - dones_t) * max_target_q_values
            else:
                raise ValueError(f'Unknown discounting scheme {config.DISCOUNTING_SCHEME}') \

            # compute q values #

            # get index in output of actions taken
            actions_t_ix = online_net.action_to_output_ix(actions_t.tolist())
            # forward on observations
            q_values = online_net.forward(observations_t)
            # collect q values of actions taken
            action_q_values = torch.gather(input=q_values, dim=1,
                                           index=torch.as_tensor(actions_t_ix).unsqueeze(-1)).squeeze(-1)

        elif self.nn_type_assign == 'branch':

            # target #
            with torch.no_grad():
                target_q_values = target_net.forward(new_observations_t)

            # per branch, select sub-action with max q value as target
            max_target_q_values = target_q_values.max(dim=-1)[0]

            # aggregate over branches (or don't)
            if config.NN_ASSIGN_AGG == 'sum':
                # value of joint action is the sum of the max q values of each action branch
                max_target_q_values = max_target_q_values.sum(dim=1)
                # -> dimension is [batch]
            elif config.NN_ASSIGN_AGG == 'mean':
                max_target_q_values = max_target_q_values.mean(dim=1)
                # -> dimension is [batch]
            elif config.NN_ASSIGN_AGG == 'none':
                pass
                # -> dimension is [batch, n_branches]
            else:
                raise ValueError(f'Unknown assignment aggregation method {config.NN_ASSIGN_AGG}')

            # Bellman equation
            if config.DISCOUNTING_SCHEME == 'fixed':
                if config.NN_ASSIGN_AGG == 'none':
                    # different dimensionality if not aggregated
                    targets = rewards_t.unsqueeze(-1) + (
                            self.discount_rate * (1 - dones_t)).unsqueeze(-1) * max_target_q_values
                else:
                    targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values
            elif config.DISCOUNTING_SCHEME == 'variable':
                if config.NN_ASSIGN_AGG == 'none':
                    targets = rewards_t.unsqueeze(-1) + (
                            (self.discount_rate ** durations_t) * (1 - dones_t)).unsqueeze(-1) * max_target_q_values
                else:
                    targets = rewards_t + (self.discount_rate ** durations_t) * (1 - dones_t) * max_target_q_values
            else:
                raise ValueError(f'Unknown discounting scheme {config.DISCOUNTING_SCHEME}')

            # q values #
            # forward on observations
            q_values = online_net.forward(observations_t)
            # collect q values of actions taken
            action_q_values = torch.gather(input=q_values, dim=-1,
                                           index=actions_t.unsqueeze(-1)).squeeze(-1)

            if config.NN_ASSIGN_AGG == 'sum':
                # sum over separate action branches -> unique Q value for each state
                action_q_values = action_q_values.sum(dim=1)
            elif config.NN_ASSIGN_AGG == 'mean':
                action_q_values = action_q_values.mean(dim=1)
            elif config.NN_ASSIGN_AGG == 'none':
                pass
            else:
                raise ValueError(f'Unknown assignment aggregation method {config.NN_ASSIGN_AGG}')

        else:
            raise ValueError(f'Unknown NN type {self.nn_type_assign}')

        # compute loss and backprop. In case the agg method above is 'none', the dimensions are different but
        # the loss function automatically applies mean reduction, which is equivalent to the Action Branching paper
        # TODO: agg=none is equivalent to mean because pytorch automatically applies mean reduction on all dimensions
        if config.LOSS_FN == 'Huber':
            # Huber loss
            loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)
        elif config.LOSS_FN == 'MSE':
            # L2 loss
            loss = torch.nn.functional.mse_loss(action_q_values, targets)
        else:
            raise ValueError(f'Unknown loss function {config.LOSS_FN}')

        # Gradient descent to update the weights of the neural network
        online_net.optimizer.zero_grad()
        loss.backward()

        if config.CLIP_GRADIENTS:
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1)

        online_net.optimizer.step()
        online_net.scheduler.step()
        # online_net.warmup_scheduler.dampening()

        # save loss
        if zoning:
            self.debug_data['loss_zone'].append(loss.item())
            # save gradients
            grads = []
            for param in online_net.parameters():
                grads.append(param.grad)
            mean_grads = np.mean([torch.mean(torch.abs(g)) for g in grads])
            self.debug_data['grads_zone'].append(mean_grads)
            # save Learning rate
            self.debug_data['lr_zone'].append(online_net.scheduler.get_last_lr()[0])

        else:
            self.debug_data['loss_assign'].append(loss.item())
            # save gradients
            grads = []
            for param in online_net.parameters():
                grads.append(param.grad)
            mean_grads = np.mean([torch.mean(torch.abs(g)) for g in grads])
            self.debug_data['grads_assign'].append(mean_grads)
            # save Learning rate
            self.debug_data['lr_assign'].append(online_net.scheduler.get_last_lr()[0])

    def validate(self, val: bool) -> (float, float):
        """
        Run the agent on the full environment, without updating the weights, fully greedy policy
        :return: total reward and episode data
        """

        # reset the environment
        if val:
            env_ = self.val_env
        else:
            env_ = self.train_env

        state_, info = env_.reset()
        total_reward = 0

        # make nested lists for loss data: one list for every val run
        self.debug_data['val_loss'].append([])

        # 1000 steps -> equal length for train and val, for comparison (otherwise, train is longer)
        for _ in range(1000):
            action_ = self.act(state_, env_, info, record_loss=True)
            next_state, reward_, terminated, info = env_.step(action_)
            total_reward += reward_
            state_ = next_state

        average_wait_time = np.mean(env_.episode_data.passenger_total_times)
        energy_consumption = np.sum(env_.episode_data.energy_consumption)

        return total_reward, average_wait_time, energy_consumption

    def load(self, load_path: Path | str, load_zone: bool = False):

        # load networks
        self.online_network_assign.load_state_dict(torch.load(load_path / 'online_network_assign.pt',
                                                              map_location=self.device))
        if load_zone:
            self.online_network_zone.load_state_dict(torch.load(load_path / 'online_network_zone.pt',
                                                                map_location=self.device))

        # load mean and std
        self.mean_assign, self.std_assign = torch.load(load_path / 'mean_assign.pt'), \
            torch.load(load_path / 'std_assign.pt')
        if load_zone:
            self.mean_zone, self.std_zone = torch.load(load_path / 'mean_zone.pt'), \
                torch.load(load_path / 'std_zone.pt')

    @staticmethod
    def state_to_tensor(state_):

        state_ = [state_[key] for key in state_.keys()]
        state_ = np.hstack(state_)
        state_ = torch.as_tensor(state_, dtype=torch.float32)
        return state_

    def get_zoning_matrix(self):
        # get matrix of zones per hour
        actions_m = np.zeros((24, self.env.n_elev))
        if self.learn_zoning:
            for hour in range(24):
                _ = self.env.reset()
                state, info = self.move_to_first_zoning_decision()
                state[0] = hour / 24
                state[1] = 2  # wednesday
                actions_m[hour] = self.choose_zone(state, self.env, policy="greedy")

        return actions_m

    def training_plots(self, train_rewards, val_rewards, wait_times_train, wait_times_val,
                          energy_cons_train, energy_cons_val, save_path):

        # create subplots to plot all data
        fig, axs = plt.subplots(3, 3)

        # change figure size, for this plot only
        fig.set_figheight(20)
        fig.set_figwidth(20)

        # plot amount of elevs respondign to call (avg every 100 steps)
        # n_elevs_assigned is not necessarily divisible by 100, so reshape and take mean
        # Calculate the number of elements to discard
        num_elements_to_discard = len(self.debug_data['n_elevs_assigned']) % 100
        # Remove the first elements
        trimmed_array = self.debug_data['n_elevs_assigned'][num_elements_to_discard:]
        axs[0, 0].plot(np.mean(np.array(trimmed_array).reshape(-1, 100), axis=1))
        axs[0, 0].set_title('Number of elevators assigned')

        # plot loss
        axs[0, 1].plot(np.mean(np.array(self.debug_data['loss_assign']).reshape(-1, 100), axis=1))
        axs[0, 1].set_title('Loss assign')

        # plot grads
        axs[0, 2].plot(np.mean(np.array(self.debug_data['grads_assign']).reshape(-1, 100), axis=1))
        axs[0, 2].set_title('Gradients assign')

        if self.learn_zoning:
            axs[1, 0].plot(self.debug_data['lr_zone'])
            axs[1, 0].set_title('Learning rate Zone')

            avg_loss = np.mean(np.array(self.debug_data['loss_zone']).reshape(-1, 20), axis=1)
            axs[1, 1].plot(avg_loss)
            axs[1, 1].set_title('Loss Zone')

            avg_grads = np.mean(np.array(self.debug_data['grads_zone']).reshape(-1, 20), axis=1)
            axs[1, 2].plot(avg_grads)
            axs[1, 2].set_title('Gradients Zone')

        # plot rewards
        axs[2, 0].plot(train_rewards, label='train')
        axs[2, 0].plot(val_rewards, label='val')
        axs[2, 0].set_title('Rewards')
        axs[2, 0].legend()
        axs[2, 0].set_xlabel('Episode')

        # plot average waiting time per episode
        axs[2, 1].plot(wait_times_train, label='train')
        axs[2, 1].plot(wait_times_val, label='val')
        axs[2, 1].set_title('Average waiting time per episode')
        axs[2, 1].legend()
        axs[2, 1].set_xlabel('Episode')

        # plot energy consumption
        axs[2, 2].plot(energy_cons_train, label='train')
        axs[2, 2].plot(energy_cons_val, label='val')
        axs[2, 2].set_title('Total energy consumption')
        axs[2, 2].legend()
        axs[2, 2].set_xlabel('Episode')

        # make 2 extra plots with Loss_val vs loss_train, and loss_zone

        # # plot loss in validation env, compare train env to val
        # fig3 = plt.figure()
        # even = np.array(self.debug_data['val_loss'])[::2]
        # odd = np.array(self.debug_data['val_loss'])[1::2]
        # even = even.mean(axis=1)
        # odd = odd.mean(axis=1)
        # plt.plot(even, label='train')
        # plt.plot(odd, label='val')
        # plt.title('Loss: train vs val')
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.legend()
        # fig3.savefig(str(save_path) + '/loss_train_val.png')

        # show all plots
        plt.show()

        # save plots
        fig.savefig(str(save_path) + '/training_plots.png')

        if self.learn_zoning:
            # separate plot for zoning matrix
            fig2 = plt.figure()
            # change figure size, for this plot only
            fig2.set_figheight(10)
            fig2.set_figwidth(10)
            # get zoning matrix
            zones = self.get_zoning_matrix().T
            plt.imshow(zones, cmap='viridis')
            # show number in each cell
            for i in range(zones.shape[0]):
                for j in range(zones.shape[1]):
                    floor = str(int(zones[i, j])) if zones[i, j] != 17 else 'X'
                    # change color of cell if no zoning decision
                    if floor == 'X':
                        plt.text(j, i, floor, ha="center", va="center", color="red")
                    else:
                        plt.text(j, i, floor, ha="center", va="center", color="w")
            # print ever 2 hours
            plt.xticks(np.arange(0, 24, 2))
            plt.title('Zoning matrix')
            plt.ylabel('Elevator')
            plt.xlabel('Hour')

            fig2.savefig(str(save_path) + '/zoning_matrix.png')


if __name__ == '__main__':

    from environment.environment_continuous import DiscreteEvent
    import argparse

    parser = argparse.ArgumentParser(description='Check if ran from DAS-5')
    parser.add_argument('--on_cluster', action='store_true')
    parser.add_argument('--from_terminal', action='store_true')
    args = parser.parse_args()

    # create save map for agents
    path = Path(__file__).parent / 'data' / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(path):
        os.makedirs(path)

    file_loc = path / 'log.log'
    logging.basicConfig(filename=file_loc, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    n_elev = 6
    n_floors = 17

    # used for training
    env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors)
    # same env but used during val, so as not to interfere with training. (deepcopy doesnt work for some reason)
    train_env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors)
    # validation data, different dataset
    val_env = DiscreteEvent(n_elev=n_elev, n_floors=n_floors, data='woensdag_donderdag.json')

    # ! Most params are set in config.py, except the ones that never change and thus are "hard-coded"
    agent = RLAgent(env=env,
                    train_env=train_env,
                    val_env=val_env,
                    training=True,
                    nn_type_assign=config.NN_TYPE_ASSIGN,
                    nn_type_zone=config.NN_TYPE_ZONE,
                    device=torch.device('cpu'),
                    n_episodes=config.N_EPISODES,
                    buffer_size=10000,
                    lr=config.LR,
                    epsilon_decay=True,
                    epsilon_start=1,
                    epsilon_end=0.1)

    train_rewards, val_rewards, wait_times_train, wait_times_val, energy_cons_train, energy_cons_val = \
        agent.training_loop(batch_size=config.BATCH_SIZE, save_path=path)

    if args.on_cluster:
        print('done training')

    # do one val run to extract results
    state, info = val_env.reset()
    terminated = False
    while not terminated:
        action = agent.act(state, info_=info)
        state, reward, terminated, info = val_env.step(action)

    print(f'Total reward: {sum(val_env.episode_data.rewards)}')
    print(f'Average reward: {np.mean(val_env.episode_data.rewards)}')
    print('Average wait time: ', np.mean(val_env.episode_data.passenger_wait_times))

    val_env.episode_data.plot_wait_times()
    val_env.episode_data.plot_reward_lines()
