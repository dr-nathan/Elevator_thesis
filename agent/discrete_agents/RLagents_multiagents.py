import copy
import os
import random
from pathlib import Path

import numpy as np
import pytorch_warmup as warmup
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

import config
from agent.discrete_agents.RLAgents_disc import DQNAgent


class DQNMulti(nn.Module):
    """
    Deep Q Network with 5 dense layers and a final layer with 3 outputs (one for each action)
    In this case, each elevator is an independent agent with its own NN. The NN therefore outputs
    3 actions for each elevator.
    :param _env: environment. Used to get the action space
    :param learning_rate: learning rate for the optimizer
    :param inp_size: input size for the NN
    :param n_episodes: number of episodes. Used for the warmup scheduler
    """

    def __init__(self, _env, learning_rate, inp_size, n_episodes):
        super().__init__()

        action_space = _env.action_space.nvec[0]  # should be 3: up, down, stay

        self.dense1 = nn.Linear(in_features=inp_size, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=264)
        self.dense3 = nn.Linear(in_features=264, out_features=128)
        self.dense4 = nn.Linear(in_features=128, out_features=64)
        self.dense5 = nn.Linear(in_features=64, out_features=32)
        self.final_layer = nn.Linear(self.dense5.out_features, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / 10)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x = self.final_layer(x)

        return x

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)


class DDQNAgentMulti(DQNAgent):
    """ Multi-Agent version of the DQN agent. Each elevator is an independent agent with its own NN.
    The NN therefore outputs 3 actions for each elevator. Neural nets are kept in a dictionary.
    See DQNAgent for params and details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dictionary of neural nets, one per elevator
        self.target_nets = {}
        self.online_nets = {}
        for agent_ in range(self.env.n_elev):
            self.target_nets[agent_] = DQNMulti(
                _env=self.env, learning_rate=self.learning_rate, inp_size=self.inp_size, n_episodes=self.n_episodes)
            self.online_nets[agent_] = DQNMulti(
                _env=self.env, learning_rate=self.learning_rate, inp_size=self.inp_size, n_episodes=self.n_episodes)

    def __repr__(self):
        return f"DQNMulti, {env.n_elev} agents, {env.n_floors} floors. \n" \
               f" {self.inp_size} input features, {self.env.action_space.nvec[0]} output features \n" \
               f"reward function: {config.REWARD_SCHEME} \n" \


    def update_target_network(self):
        """ update every target network with the online network """
        for agent_ in range(self.env.n_elev):
            self.target_nets[agent_].load_state_dict(self.online_nets[agent_].state_dict())

    def save_better_model(self, save_path: Path, data_val, _val_rewards: list, iteration: int):
        """ save all NN's in map if better performance on validation set
        :param save_path: path to map that contains the NN's
        :param data_val: reward on most recent validation set
        :param _val_rewards: all validation rewards
        :param iteration: current iteration
        """
        if _val_rewards == [] or data_val > max(_val_rewards):
            for agent_ in range(self.env.n_elev):
                torch.save(self.online_nets[agent_].state_dict(), str(save_path) + f'/agent_{agent_}.pt')
            self.best_agent = iteration

    def load(self, load_path: Path):
        for i in range(self.env.n_elev):
            self.online_nets[i].load_state_dict(torch.load(str(load_path) + f'/agent_{i}.pt'))

    def choose_action(self, state: np.ndarray, _env: 'DiscreteNocam',
                      info: dict, policy: str):
        """
        Choose an action according to the given policy
        :param state: current state
        :param _env: environment. Used to sample a random action and to get the valid action mask
        :param info: info dict from the environment
        :param policy: policy to use. Can be "random", "greedy" or "epsilon_greedy"
        :returns: action
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
            state = (state - self.mean) / (self.std + 1e-9)  # TODO: mean and std based on transitions in buffer. OK?
            # initial buffer-> fully random. Accurate?

            with torch.no_grad():
                # forward pass on state to get Q-values
                res = torch.stack([net.forward(state) for net in self.online_nets.values()], dim=1)
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
                # recurse
                return self.choose_action(state, _env, info, "random")
            else:
                return self.choose_action(state, _env, info, "greedy")

        raise ValueError("Unknown policy")

    def learn(self, batch_size: int):

        """
        Sample a batch of transitions from the replay memory, and update the online network
        :param batch_size: number of transitions that will be sampled
        """

        # sampler also takes care of converting to tensors, hence _t suffix
        (
            observations_t,
            actions_t,
            rewards_t,
            dones_t,
            masks_t,
            default_action_t,
            new_observations_t,
        ) = self.replay_memory.sample(batch_size)

        # normalize observations. Again, based on initial buffer only. Might not be accurate
        new_observations_t = (new_observations_t - self.mean) / (self.std + 1e-9)
        observations_t = (observations_t - self.mean) / (self.std + 1e-9)
        # normalize rewards. This is batch normalized.
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-9)

        # aggregate losses, learning rates and gradients for each agent. Add to self.episode_data later
        local_losses = []
        local_grads = []
        local_lr = []

        for agent_ in range(self.env.n_elev):
            self.online_nets[agent_].train()
            self.target_nets[agent_].eval()

            # Compute targets #
            with torch.no_grad():
                target_q_values = self.target_nets[agent_](new_observations_t)
                # mask invalid actions
                target_q_values[masks_t[:, :, agent_] == 0] = -np.inf

                # replace with default actions where needed
                long_tensor = torch.as_tensor(default_action_t[:, agent_], dtype=torch.long)
                default_action_values = target_q_values.gather(1, long_tensor.unsqueeze(1)).squeeze(1)

                # get max q values of allowed actions (so when default action is set,
                # take the q value of the default action,
                # otherwise take the max q value of legal actions)
                max_target_q_values = target_q_values.max(dim=1)[0]
                max_target_q_values = torch.where(default_action_t[:, agent_].isnan(),
                                                  max_target_q_values,
                                                  default_action_values)

                targets = rewards_t + self.discount_rate * (1 - dones_t) * max_target_q_values

            # Compute loss #
            q_values = self.online_nets[agent_](observations_t)
            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t[:, agent_].unsqueeze(1)).squeeze(1)
            # Huber loss
            # loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)
            # MSE loss
            loss_ = torch.nn.functional.mse_loss(action_q_values, targets)

            # Gradient descent to update the weights of the neural network
            self.online_nets[agent_].optimizer.zero_grad()
            loss_.backward()
            # torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1)
            self.online_nets[agent_].optimizer.step()
            with self.online_nets[agent_].warmup_scheduler.dampening():
                self.online_nets[agent_].scheduler.step()

            # save loss
            local_losses.append(loss_.item())
            # save gradients
            grads = []
            for param in self.online_nets[agent_].parameters():
                grads.append(param.grad)
            mean_grads = np.mean([torch.mean(torch.abs(g)) for g in grads])
            local_grads.append(mean_grads)
            # save Learning rate
            local_lr.append(self.online_nets[agent_].warmup_scheduler.lrs[0])

        self.debug_data['loss'].append(local_losses)  # list of lists
        self.debug_data['grads'].append(local_grads)
        self.debug_data['lr'].append(local_lr)


if __name__ == "__main__":
    from environment.environment_discrete import DiscreteNocam

    n_elev = 2
    n_floors = 5

    env = DiscreteNocam(n_elev=n_elev, n_floors=n_floors)

    agent = DDQNAgentMulti(_env=env,
                           val_env=copy.deepcopy(env),
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           n_episodes=2000000,
                           buffer_size=10000,
                           discount_rate=0.99,
                           lr=5e-4,
                           epsilon_decay=True,
                           epsilon_start=1,
                           epsilon_end=0.1)

    # create map for agents
    path = Path(__file__).parent / 'data' / 'discrete' / f'multi_online_network_{n_elev}elev_{n_floors}floors'
    if not os.path.exists(path):
        os.makedirs(path)

    train_rewards, val_rewards = agent.training_loop(batch_size=32, save_path=path)

    plt.plot(agent.debug_data['lr'])
    plt.title('Learning rate')
    plt.show()

    plt.plot(agent.debug_data['epsilon'])
    plt.title('Epsilon')
    plt.show()

    # average every 20 steps
    loss = np.array(agent.debug_data['loss'])
    # loss shape = (n_episodes, n_agents) reshape by adding 3rd dimension
    avg_loss = np.mean(loss.reshape((-1, 20, n_elev)), axis=1)
    plt.plot(avg_loss)
    plt.title('Loss')
    plt.ylim(0, 1)
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
