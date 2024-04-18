import itertools

import numpy as np
import torch
from torch import nn as nn, optim as optim

import config
from environment.environment_continuous import DiscreteEvent


# Assigning agents first

class CombinatorialQ(nn.Module):
    """
    Deep Q Network. Takes as input the state, output layer is all the possible combinations of elevators responding.
    It exludes the possibility of no elevator responding.
    :param env_: gym environment
    :param learning_rate: learning rate for the optimizer
    :param inp_size: input size of the neural network.
    """

    def __init__(self, env_: 'DiscreteEvent', learning_rate, inp_size, n_episodes):
        super().__init__()

        # all combinations of 3, 2 or 1 elevators assigned to call (explicitly remove 0 -> always 1 elevator assigned)
        self.out_features, self.combinations, self.combinations_list = (
            self.get_output_size_and_combinations(config.MAX_ELEVS_RESPONDING, env_))

        # Current input size: 28  (big state size)
        if config.NN_SIZE == 'small':
            self.main_body = nn.Sequential(nn.Linear(in_features=inp_size, out_features=64),
                                           nn.ReLU(),
                                           nn.Linear(in_features=64, out_features=256),
                                           nn.ReLU())
        elif config.NN_SIZE == 'large':
            self.main_body = nn.Sequential(nn.Linear(in_features=inp_size, out_features=128),
                                           nn.ReLU(),
                                           nn.Linear(in_features=128, out_features=512),
                                           nn.ReLU(),
                                           nn.Linear(in_features=512, out_features=256),
                                           nn.ReLU())
        self.out = nn.Linear(in_features=256, out_features=self.out_features)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / config.LEARN_INTERVAL)
        # self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=n_episodes / 100)
        # cant get it to work
        self.init_weights()

    def forward(self, x):

        x = self.main_body(x)
        x = self.out(x)
        return x

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)

    @staticmethod
    def get_output_size_and_combinations(n_elevators: int, env_: 'DiscreteEvent'):
        """ Build all possible combinations of elevators responding to a call.
        :param n_elevators: number of elevators responding to a call
        :param env_: gym environment
        :return: number of combinations, all combinations, all combinations as list"""

        # make all combinations of 1 of 6 elevators responding 1
        comb1 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 1)]
        # make all combinations of 2 of 6 elevators responding 1
        comb2 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 2)]
        # make all combinations of 3 of 6 elevators responding 1
        comb3 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 3)]

        if n_elevators == 1:
            combinations = comb1  # 6 combinations
        elif n_elevators == 2:
            combinations = comb1 + comb2  # 21 combinations
        elif n_elevators == 3:
            combinations = comb1 + comb2 + comb3  # 41 combinations
        else:
            raise ValueError('n_elevators must be 1, 2 or 3')

        return len(combinations), combinations, [list(combi) for combi in combinations]

    def action_to_output_ix(self, actions: list):
        """ used to encode action to corresponding output"""
        return [self.combinations_list.index(action_) for action_ in actions]


class DuelingCombinatorialQ(nn.Module):

    """
    Dueling Deep Q Network. Takes as input the state,
    output layer is all the possible combinations of elevators responding.
    Very similar to CombinatorialQ, but with separate value and advantage heads.
    :param env_: environment
    :param learning_rate: learning rate for the optimizer
    :param inp_size: input size of the neural network.
    """

    def __init__(self, env_: 'DiscreteEvent', learning_rate, inp_size, n_episodes):
        super().__init__()

        # all combinations of 3, 2 or 1 elevators assigned to call (explicitly remove 0 -> always 1 elevator assigned)
        self.out_features, self.combinations, self.combinations_list = (
            self.get_output_size_and_combinations(config.MAX_ELEVS_RESPONDING, env_))

        # Current input size: 28  (big state size)
        if config.NN_SIZE == 'small':
            self.main_body = nn.Sequential(nn.Linear(in_features=inp_size, out_features=64),
                                           nn.ReLU(),
                                           nn.Linear(in_features=64, out_features=256),
                                           nn.ReLU())
        elif config.NN_SIZE == 'large':
            self.main_body = nn.Sequential(nn.Linear(in_features=inp_size, out_features=128),
                                           nn.ReLU(),
                                           nn.Linear(in_features=128, out_features=512),
                                           nn.ReLU(),
                                           nn.Linear(in_features=512, out_features=256),
                                           nn.ReLU())

        # add separate heads for value and advantage
        self.value_head = nn.Linear(in_features=256, out_features=1)
        self.adv_head = nn.Linear(in_features=256, out_features=self.out_features)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / config.LEARN_INTERVAL)
        # self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=n_episodes / 100)
        # cant get it to work
        self.init_weights()

    def forward(self, x):
        x = self.main_body(x)
        value = self.value_head(x)
        adv = self.adv_head(x)

        return value + (adv - adv.mean(dim=-1, keepdim=True))

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)

    @staticmethod
    def get_output_size_and_combinations(n_elevators: int, env_: 'DiscreteEvent'):
        """ Build all possible combinations of elevators responding to a call.
        :param n_elevators: number of elevators responding to a call
        :param env_: gym environment
        :return: number of combinations, all combinations, all combinations as list"""

        # make all combinations of 1 of 6 elevators responding 1
        comb1 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 1)]
        # make all combinations of 2 of 6 elevators responding 1
        comb2 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 2)]
        # make all combinations of 3 of 6 elevators responding 1
        comb3 = [np.bincount(xs, minlength=env_.n_elev) for xs in itertools.combinations(range(env_.n_elev), 3)]

        if n_elevators == 1:
            combinations = comb1  # 6 combinations
        elif n_elevators == 2:
            combinations = comb1 + comb2  # 21 combinations
        elif n_elevators == 3:
            combinations = comb1 + comb2 + comb3  # 41 combinations
        else:
            raise ValueError('n_elevators must be 1, 2 or 3')

        return len(combinations), combinations, [list(combi) for combi in combinations]

    def action_to_output_ix(self, actions: list):
        """ used to encode action to corresponding output"""
        return [self.combinations_list.index(action_) for action_ in actions]


class BranchingQ(nn.Module):
    """
        Deep Q Network.
        :param _env: gym environment
        :param learning_rate: learning rate for the optimizer
        :param inp_size: input size of the neural network.
        """

    def __init__(self, _env: 'DiscreteEvent', learning_rate, inp_size, n_episodes):
        super().__init__()

        # common layers
        self.dense1 = nn.Linear(in_features=inp_size, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=256)

        # value layers
        if config.NN_ASSIGN_USE_ADV:
            self.value_hidden = nn.Linear(256, 128)
            self.value_head = nn.Linear(128, 1)

        # advantage layers
        self.adv_hiddens = nn.ModuleList([nn.Linear(256, 128) for _ in range(_env.n_elev)])
        self.adv_heads = nn.ModuleList([nn.Linear(128, 2) for _ in range(_env.n_elev)])  # 2 = assign or not

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / config.LEARN_INTERVAL)
        # self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.init_weights()

    def forward(self, x):

        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))

        advs_h = torch.stack([adv_hidden(x) for adv_hidden in self.adv_hiddens])
        advs = torch.stack([adv_head(adv_h) for adv_head, adv_h in zip(self.adv_heads, advs_h)], dim=-2)

        if config.NN_ASSIGN_USE_ADV:
            value_h = self.value_hidden(x)
            value = self.value_head(value_h)

            q_val = value.unsqueeze(-1) + advs - advs.mean(dim=-1, keepdim=True)

        else:
            q_val = advs

        return q_val

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)


class ZoningSequentialQ(nn.Module):
    """
    Q network for learning where to zone each elevator at every hour. processes every elevator separately.
    """

    def __init__(self, _env: 'DiscreteEvent', learning_rate, inp_size, n_episodes):
        super().__init__()

        self.dense1 = nn.Linear(in_features=inp_size, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=_env.n_floors+1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / config.LEARN_INTERVAL)
        # self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = self.out(x)

        return x

    def init_weights(self):
        """ initialize the weights of the NN """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0.1)


class ZoningBranchingQ(BranchingQ):
    """
    Q network for learning where to zone each elevator at every hour.
    Similar to Branching architecture, but without separate value and advantage heads.
    Just one main body and separate heads for each elevator.
    Takes as input the state, must output a decision for each elevator.
    Uses same multi-head architecture as BranchingQ, but outputs one floor out of 17 possibilities for each elevator.
    -> 4 branches, each with 17 sub-actions
    """

    def __init__(self, _env: 'DiscreteEvent', learning_rate, inp_size, n_episodes):
        super().__init__(_env, learning_rate, inp_size, n_episodes)

        # delete value and advantage heads
        if config.NN_ASSIGN_USE_ADV:
            # dont exist if not using advantage for assign agent
            del self.value_hidden
            del self.value_head
        del self.adv_hiddens
        del self.adv_heads

        # add separate heads for each elevator
        self.elev_hiddens = nn.ModuleList([nn.Linear(256, 128) for _ in range(_env.n_elev)])
        self.elev_heads = nn.ModuleList([nn.Linear(128, _env.n_floors + 1) for _ in range(_env.n_elev)])

        if config.NN_ZONING_USE_ADV:
            # dont exist if not using advantage for zoning agent
            self.value_hidden = nn.Linear(256, 128)
            self.value_head = nn.Linear(128, 1)

        # reinitialize optimizer and scheduler now that parameters and modules have changed
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_episodes / config.LEARN_INTERVAL)
        # self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.init_weights()

    def forward(self, x):

        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))

        elevs_h = torch.stack([elev_hidden(x) for elev_hidden in self.elev_hiddens])
        elevs = torch.stack([elev_head(elev) for elev_head, elev in zip(self.elev_heads, elevs_h)], dim=-2)

        if config.NN_ZONING_USE_ADV:
            value_h = self.value_hidden(x)
            value = self.value_head(value_h)

            q_val = value.unsqueeze(-1) + elevs - elevs.mean(dim=-1, keepdim=True)
        else:
            q_val = elevs

        return q_val
