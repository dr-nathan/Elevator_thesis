import numpy as np


class FirstElevAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def __repr__(self):
        return 'FirstElevAgent'

    def act(self, state, info):

        action = np.zeros(self.env.n_elev)

        action[0] = 1

        return action


class RandomAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def __repr__(self):
        return 'RandomAgent'

    def act(self, state, env, info):

        action = np.zeros(self.env.n_elev)

        action[np.random.randint(0, self.env.n_elev)] = 1

        return action


class ClosestAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def __repr__(self):
        return 'ClosestAgent'

    def act(self, state, env, info):

        action = np.zeros(self.env.n_elev)

        # use actual env, not self.env (self.env is used to determine parameters but can remain static)
        closest_elevator = np.argmin(np.abs(env.building.get_elevator_distances(state['floor'])))

        action[closest_elevator] = 1

        return action


class SectorAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.sectors = self.assign_sectors()

    def __repr__(self):
        return 'SectorAgent'

    def assign_sectors(self):

        floors = np.arange(self.env.n_floors)
        return np.array_split(floors, self.env.n_elev)

    def act(self, state, env, info):

        action = np.zeros(self.env.n_elev)

        floor = state['floor']
        sector = np.argwhere([floor in s for s in self.sectors])[0][0]
        action[sector] = 1

        return action


class LeastBusyAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def __repr__(self):
        return 'LeastBusyAgent'

    def act(self, state, env, info):

        # zoning decisions, skip
        if 'zoning' in info:

            action = [None] * self.env.n_elev

            return action

        action = np.zeros(self.env.n_elev)

        # get all elevs with least stops
        stops_in_queue = [len(env.building.elevators[ix].destination_queue) for ix in range(self.env.n_elev)]
        least_busy_elevators = np.argwhere(stops_in_queue == np.min(stops_in_queue)).flatten()

        # if any ties, choose closest
        if len(least_busy_elevators) > 1:
            # get closest elevator within array of least busy elevators
            least_busy_elevators_distances = np.abs([env.building.get_elevator_distances(state['floor'])[ix]
                                                for ix in least_busy_elevators])

            closest_elevator = least_busy_elevators[np.argmin(least_busy_elevators_distances)]
            action[closest_elevator] = 1
        else:
            action[least_busy_elevators[0]] = 1

        return action


class ETDAgent(object):

    def __init__(self, env, set_zoning=False):
        super().__init__()
        self.env = env
        self.set_zoning = set_zoning
        self.zone_matrix = self.make_zoning_schedule()

    def __repr__(self):
        return 'ETDAgent'

    @staticmethod
    def make_zoning_schedule():

        # data vrom VU building
        zoning_lift1 = 6 * [None] + [1] + 2 * [2] + [6] + 5 * [2] + [14, 10, 17, 2, 8, 2, 14, 16, None]
        zoning_lift2 = 6 * [None] + [1, 2, 11] + 3 * [2] + [6] + 3 * [2] + [6, 8, 2, 14, 8, 8, 16, 13]
        zoning_lift3 = 8 * [None] + [2, 8, 2, 2, 8, 16, 16] + 3 * [2] + 2 * [6] + [2, 2, 16, 8]
        zoning_lift4 = 7 * [None] + [5] + 4 * [2] + [10, 2, 6, 2, 14, 2, 10, 16, 16, 8] + 2 * [None]
        zoning_lift5 = 7 * [None] + [1] + 4 * [2] + [12, 16, 2, 2, 14, 14, 2, 2, 16] + 3 * [None]
        zoning_lift6 = 6 * [None] + [1, 12] + 3 * [2] + [8, 6, 2, 16, 14, 6, 6] + 4 * [16] + [None] + [13]

        all_lifts = [zoning_lift1, zoning_lift2, zoning_lift3, zoning_lift4, zoning_lift5, zoning_lift6]

        # repeat, so resolution is every 30 minutes
        zonematrix = np.asarray([np.repeat(liftlist, 2) for liftlist in all_lifts])

        return zonematrix

    def get_zone(self, hour, minute):
        return self.zone_matrix[:, hour * 2 + (minute >= 30)]

    def act(self, state, env, info):

        if 'zoning' in info:

            if self.set_zoning:
                action = self.get_zone(env.time.hour, env.time.minute)
            else:
                action = [None] * self.env.n_elev
                print('ETD agent: forgot to set zoning while env is set to zoning mode')

            return action

        call_floor = state['floor']
        call_dir = 'up' if state['direction'] == 1 else 'down' if state['direction'] == -1 else None

        scores = []
        for elev in env.building.elevators:
            eta, penalty = elev.calculate_ETD_score(call_floor, call_dir)
            scores.append(eta + penalty)

        action = np.zeros(self.env.n_elev)
        action[np.argmin(scores)] = 1

        return action


if __name__ == '__main__':

    from environment import environment_continuous as environment

    env = environment.DiscreteEvent(4, 17)  # 2 elevators, 17 floors (15 floors + 2 basement floors)

    agent_type = 'ETDAgent'

    if agent_type == 'first_elev':
        agent = FirstElevAgent(env)
    elif agent_type == 'random':
        agent = RandomAgent(env)
    elif agent_type == 'sector':
        agent = SectorAgent(env)
    elif agent_type == 'closest':
        agent = ClosestAgent(env)
    elif agent_type == 'LeastBusyAgent':
        agent = LeastBusyAgent(env)
    elif agent_type == 'ETDAgent':
        agent = ETDAgent(env, set_zoning=True)
    else:
        raise NotImplementedError(f'Agent type {agent_type} not implemented')

    state, info = env.reset()

    if agent_type == 'sector':
        for ix, elev in enumerate(env.building.elevators):
            elev.rest_floor = agent.sectors[ix][0]
        env.building.elevators[0].rest_floor = 1  # ground floor

    rewards = {'reward': [],
               'duration': []}

    terminated = False
    while not terminated:
        action = agent.act(state, env, info)
        state, reward, terminated, info = env.step(action)
        rewards['reward'].append(reward)
        rewards['duration'].append(info['step_length'])

    print(f'Total reward: {sum(rewards["reward"])}')
