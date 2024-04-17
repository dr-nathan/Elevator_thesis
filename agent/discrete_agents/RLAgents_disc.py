import copy
import random
import typing
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import pytorch_warmup as warmup
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

if typing.TYPE_CHECKING:
    import gymnasium as gym
    from environment.environment_discrete import DiscreteNocam

from agent.discrete_agents.RuleBasedAgents import BaseAgent


class DQN(nn.Module):
    """
    Deep Q Network.
    :param _env: gym environment
    :param learning_rate: learning rate for the optimizer
    :param inp_size: input size of the neural network.
    """

    def __init__(self, _env, learning_rate, inp_size, n_episodes):
        super().__init__()
        # convert Dict to array with wrapper
        input_features = inp_size
        action_space = len(_env.action_space.nvec)

        self.dense1 = nn.Linear(in_features=input_features, out_features=512)
        # self.dense1_1 = nn.Linear(in_features=512, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=264)
        self.dense3 = nn.Linear(in_features=264, out_features=128)
        self.dense4 = nn.Linear(in_features=128, out_features=64)
        self.dense5 = nn.Linear(in_features=64, out_features=32)
        self.dense5ActionA = nn.Linear(in_features=32, out_features=action_space)
        self.dense5ActionB = nn.Linear(in_features=32, out_features=action_space)
        self.dense5ActionC = nn.Linear(in_features=32, out_features=action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes/10)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.init_weights()

    def forward(self, x):

        x = torch.relu(self.dense1(x))
        # x = torch.relu(self.dense1_1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x1 = self.dense5ActionA(x)
        x2 = self.dense5ActionB(x)
        x3 = self.dense5ActionC(x)
        x = torch.stack([x1, x2, x3], dim=-2)
        return x

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)


class ExperienceReplay:
    def __init__(self, buffer_size: int, min_replay_size: int, _agent: 'DQNAgent'):

        """
        Object that stores the experience replay buffer and samples random transitions from it.
        :param _agent: agent that needs to play
        :param buffer_size = max number of transitions that the experience replay buffer can store
        :param min_replay_size = min number of (random) transitions that the replay buffer needs
        to have when initialized
        """

        self.agent = _agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)

        self.fill_replay_memory()

    def fill_replay_memory(self):
        """
        Fills the replay memory with random transitions, according to self.min_replay_size
        """

        state, info = self.agent.env.reset()
        state = self.agent.convert_state_dict(state)

        for i in range(self.min_replay_size):

            # choose random action, no NN yet
            action = self.agent.choose_action(state, self.agent.env, info, policy="random")
            next_state, reward, terminated, info = self.agent.env.step(action)
            next_state = self.agent.convert_state_dict(next_state)
            mask = info['mask']
            default_action = info['default_action']
            transition = (state, action, reward, terminated, mask, default_action, next_state)
            self.replay_buffer.append(transition)
            state = next_state

            if terminated:
                state, info = self.agent.env.reset()
                state = self.agent.convert_state_dict(state)

    def sample(self, batch_size: int):

        # sample random transitions from the replay memory
        transitions = random.sample(self.replay_buffer, batch_size)

        # convert to array where needed, then to tensor (faster than directly to tensor)
        observations = np.asarray([t[0] for t in transitions], dtype=float)
        actions = np.asarray([t[1] for t in transitions], dtype=int)
        rewards = np.asarray([t[2] for t in transitions], dtype=float)
        dones = np.asarray([t[3] for t in transitions], dtype=int)
        masks = np.asarray([t[4] for t in transitions], dtype=int)
        default_action = np.asarray([t[5] for t in transitions], dtype=float)
        new_observations = np.asarray([t[6] for t in transitions], dtype=float)

        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        )
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(
            dones, dtype=torch.int64, device=self.device
        )
        masks_t = torch.as_tensor(
            masks, dtype=torch.int64, device=self.device
        )
        default_action_t = torch.as_tensor(
            default_action, dtype=torch.float32, device=self.device
        )
        new_observations_t = torch.as_tensor(
            new_observations, dtype=torch.float32, device=self.device
        )

        return observations_t, actions_t, rewards_t, dones_t, masks_t, default_action_t, new_observations_t

    def get_mean_and_std(self):

        states = []

        for exp in self.replay_buffer:
            state, *_ = exp
            states.append(state)
        np.asarray(states)
        mean = np.mean(states, axis=0)
        std = np.std(states, axis=0)
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device)

        return mean, std


class DQNAgent(BaseAgent):
    """
    DQN agent that uses a neural network to approximate the Q-function.
    :param _env: environment that the agent needs to play in
    :param val_env: environment that the agent uses to evaluate its performance
    :param device: device on which neural network is stored
    :param epsilon: exploration rate
    :param epsilon_decay: whether to decay the exploration rate
    :param epsilon_start: starting exploration rate, only used if epsilon_decay = True
    :param epsilon_end: final exploration rate, only used if epsilon_decay = True
    :param n_episodes: number of episodes to train the agent
    :param discount_rate: discount rate for future rewards
    :param lr: learning rate for the neural network
    :param buffer_size: size of the experience replay buffer
    """

    def __init__(
            self,
            _env: 'DiscreteNocam',
            val_env: 'DiscreteNocam | None' = None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epsilon: float = 1.0,
            epsilon_decay: bool = True,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            n_episodes: int = 20,
            discount_rate: float = 0.98,
            lr: float = 5e-4,
            buffer_size: int = 10000,
    ):

        super().__init__(_env)

        self.debug_data = defaultdict(list)
        self.env = _env
        self.val_env = val_env
        self.device = device
        self.n_episodes = n_episodes
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size

        self.best_agent = None

        if epsilon_decay:
            self.epsilon = epsilon_start
            self.epsilon_decay_step = np.exp(
                np.log(epsilon_end / epsilon_start) / self.n_episodes
            )
        else:
            self.epsilon = epsilon
            self.epsilon_decay_step = 1.0

        # workaround since FlattenObservation does not work
        self.inp_size = len(self.convert_state_dict(self.env.observation_space.sample()))

        # set min_reply_size equal to buffer_size: fill up the buffer before training
        self.replay_memory = ExperienceReplay(
            buffer_size=self.buffer_size, min_replay_size=self.buffer_size, _agent=self)
        self.mean, self.std = self.replay_memory.get_mean_and_std()
        self.online_network = DQN(self.env, self.learning_rate, self.inp_size, self.n_episodes).to(self.device)
        self.target_network = DQN(self.env, self.learning_rate, self.inp_size, self.n_episodes).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def training_loop(self, batch_size: int, save_path: Path):

        # reset the environment
        state, info = self.env.reset()
        state = self.convert_state_dict(state)

        _train_rewards = []
        _val_rewards = []

        self.best_agent = None

        for iteration in tqdm(range(self.n_episodes)):

            # play one move, add the transition to the replay memory
            state, info, *_ = self.take_step_training(state, info)

            # decay epsilon
            self.epsilon *= self.epsilon_decay_step
            self.debug_data['epsilon'].append(self.epsilon)

            if (iteration + 1) % 10 == 0:
                # sample a batch of transitions from the replay memory, and update online network
                self.learn(batch_size=batch_size)

            # every 500 iterations, update the target network
            if (iteration + 1) % 400 == 0:
                self.update_target_network()

            # get some statistics, 100 times during training
            if (iteration + 1) % (self.n_episodes // 100) == 0:
                data_train, _ = self.validate(copy.deepcopy(self.env))
                data_val, _ = self.validate(copy.deepcopy(self.val_env))

                # Save the model if the validation reward is the best so far
                self.save_better_model(save_path, data_val, _val_rewards, iteration)

                # save the rewards
                _train_rewards.append(data_train)
                _val_rewards.append(data_val)

                # reset env as well
                state, info = self.env.reset()
                state = self.convert_state_dict(state)

        print(f"Best agent at iteration {self.best_agent} of {self.n_episodes}\n"
              f"saved at {save_path}")

        return _train_rewards, _val_rewards

    def update_target_network(self):
        """
        Update the target network with the parameters of the online network
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_better_model(self, save_path, data_val, _val_rewards, iteration):
        if _val_rewards == [] or data_val > max(_val_rewards):
            torch.save(self.online_network.state_dict(), save_path)
            self.best_agent = iteration

    def take_step_training(self, state: np.ndarray, info: dict):
        """
        Take one step in the environment, and add the transition to the replay memory. Always choose epsilon-greedy
        :param state: current state
        :param info: info dict from the environment
        :return: next state, action, reward
        """

        action = self.choose_action(state, self.env, info, "epsilon_greedy")
        next_state, reward, terminated, info = self.env.step(action)
        next_state = self.convert_state_dict(next_state)
        mask = info['mask']
        default_action = info['default_action']
        self.replay_memory.replay_buffer.append(
            (state, action, reward, terminated, mask, default_action, next_state)
        )

        state = next_state

        if terminated:
            state, info = self.env.reset()
            state = self.convert_state_dict(state)

        return state, info, action, reward

    def choose_action(self, state: np.ndarray, _env: 'DiscreteNocam',
                      info: dict, policy: str):
        """
        Choose an action according to the given policy
        :param state: current state
        :param _env: environment. Used to sample a random action and to get the valid action mask
        :param info: info dict from the environment
        :param policy: policy to use. Can be "random", "greedy" or "epsilon_greedy"
        :return: action
        """

        if policy == "random":
            action = _env.sample_valid_action()

            # replace with default action if necessary
            action = \
                np.where(
                    np.isnan(
                        np.asarray(
                            info['default_action'], dtype=float)), action, info['default_action'])

            return action

        elif policy == "greedy":

            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            # normalize state
            state = (state - self.mean) / (self.std + 1e-9)

            res = self.online_network.forward(state)
            # mask invalid actions
            mask = torch.as_tensor(info['mask'], dtype=torch.float32, device=self.device)
            ix = np.where(mask == 0)
            res[ix] = -np.inf  # to make sure to not choose invalid actions (set 0 to -inf)
            action = res.argmax(dim=0).tolist()
            # replace default actions
            action = \
                np.where(
                    np.isnan(
                        np.asarray(
                            info['default_action'], dtype=float)), action, info['default_action'])
            return action

        elif policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return self.choose_action(state, _env, info, "random")
            else:
                return self.choose_action(state, _env, info, "greedy")

        raise ValueError("Unknown policy")

    def act(self, state: 'gym.spaces.Dict', info: dict, _env: 'DiscreteNocam | None' = None) -> np.ndarray:
        """
        Choose an action according to the greedy policy. Meant to be used outside of training
        :param state: current state
        :param _env: environment (as param, cuz it can be different from self.env, when validating for example)
        :param info: info dict from the environment
        :return: action
        """

        state = self.convert_state_dict(state)
        _env = _env if _env is not None else self.env
        return self.choose_action(state, _env, info, "greedy")

    def learn(self, batch_size: int):

        """
        Sample a batch of transitions from the replay memory, and update the online network
        :param batch_size: number of transitions that will be sampled
        """

        # sampler also takes care of converting to tensors
        (
            observations_t,
            actions_t,
            rewards_t,
            dones_t,
            masks_t,
            default_action_t,
            new_observations_t,
        ) = self.replay_memory.sample(batch_size)

        # Compute targets #
        self.target_network.eval()
        with torch.no_grad():
            # normalize observations
            new_observations_t = (new_observations_t - self.mean) / (self.std + 1e-9)
            target_q_values = self.target_network.forward(new_observations_t)
            # mask invalid actions
            target_q_values[masks_t == 0] = -np.inf  # to make sure to not choose invalid actions (set 0 to -inf)

            # replace with default actions where needed
            long_tensor = torch.as_tensor(default_action_t, dtype=torch.long)
            default_action_values = target_q_values.gather(1, long_tensor.unsqueeze(1)).squeeze(1)

            # get max q values of allowed actions (so when default action is set,
            # take the q value of the default action,
            # otherwise take the max q value of legal actions)
            max_target_q_values = target_q_values.max(dim=1)[0]
            max_target_q_values = torch.where(default_action_t.isnan(),
                                              max_target_q_values,
                                              default_action_values)

            # normalize rewards
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-9)
            targets = rewards_t[:, None] + self.discount_rate * (1 - dones_t[:, None]) * max_target_q_values

        # Compute loss #
        self.online_network.train()
        # normalize observations
        observations_t = (observations_t - self.mean) / (self.std + 1e-9)
        q_values = self.online_network.forward(observations_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t.unsqueeze(-2)).squeeze(-2)
        # Huber loss
        # loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)
        # L2 loss
        loss = torch.nn.functional.mse_loss(action_q_values, targets)

        # Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1)
        self.online_network.optimizer.step()
        with self.online_network.warmup_scheduler.dampening():
            self.online_network.scheduler.step()

        # save loss
        self.debug_data['loss'].append(loss.item())
        # save gradients
        grads = []
        for param in self.online_network.parameters():
            grads.append(param.grad)
        mean_grads = np.mean([torch.mean(torch.abs(g)) for g in grads])
        self.debug_data['grads'].append(mean_grads)
        # save Learning rate
        self.debug_data['lr'].append(self.online_network.warmup_scheduler.lrs[0])

    def validate(self, _env: 'DiscreteNocam') -> tuple[float, dict]:
        """
        Run the agent on the environment for 1000 steps, without updating the weights, fully greedy policy
        :param _env: environment to validate on
        :return: total reward and episode data
        """

        # reset the environment
        state, info = _env.reset()
        total_reward = 0

        for _ in range(1000):
            action = self.act(state, info, _env)
            next_state, reward, terminated, info = _env.step(action)
            total_reward += reward
            state = next_state

        return total_reward, _env.episode_data

    def load(self, load_path: Path):
        self.online_network.load_state_dict(torch.load(load_path, map_location=self.device))

    @staticmethod
    def convert_state_dict(state):

        state = [state[key] for key in state.keys()]
        list1 = []
        [list1.extend(part.flatten()) for part in state]

        return np.array(list1)


if __name__ == "__main__":
    from environment.environment_discrete import DiscreteNocam

    n_elev = 2
    n_floors = 8

    env = DiscreteNocam(n_elev=n_elev, n_floors=n_floors)

    agent = DQNAgent(_env=env,
                     val_env=copy.deepcopy(env),
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     n_episodes=2000000,
                     buffer_size=10000,
                     discount_rate=0.99,
                     lr=5e-4,
                     epsilon_decay=True,
                     epsilon_start=1,
                     epsilon_end=0.1)

    path = Path(__file__).parent / 'data' / 'discrete' / f'single_online_network_{n_elev}elev_{n_floors}floors.pt'

    train_rewards, val_rewards = agent.training_loop(batch_size=32, save_path=path)

    plt.plot(agent.debug_data['lr'])
    plt.title('Learning rate')
    plt.show()

    plt.plot(agent.debug_data['epsilon'])
    plt.title('Epsilon')
    plt.show()

    # average every 20 steps
    avg_loss = np.mean(np.array(agent.debug_data['loss']).reshape(-1, 20), axis=1)
    plt.plot(avg_loss)
    plt.title('Loss')
    plt.show()

    # plot grads
    avg_grads = np.mean(np.array(agent.debug_data['grads']).reshape(-1, 20), axis=1)
    plt.plot(avg_grads)
    plt.title('Gradients')
    plt.show()

    plt.plot(train_rewards, label='train')
    plt.plot(val_rewards, label='val')
    plt.title('Rewards over time')
    plt.legend()
    plt.show()
