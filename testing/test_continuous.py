import datetime

from environment.environment_continuous import DiscreteEvent

env = DiscreteEvent(2, 17)  # 2 elevators, 17 floors (15 floors + 2 basement floors)


def test_idle_pickup_samefloor():
    data = [{'arrival_time': datetime.datetime(2023, 6, 7, 6, 1, 12),
             'floor': 1,
             'direction': 'up',
             'destination': 15,
             'time_waited': None},
            {'arrival_time': datetime.datetime(2023, 6, 7, 6, 1, 13),
             'floor': 1,
             'direction': 'down',
             'destination': 0,
             'time_waited': None},
            {'arrival_time': datetime.datetime(2023, 6, 7, 6, 1, 21),
             'floor': 10,
             'direction': 'up',
             'destination': 14,
             'time_waited': None}
            ]
    env.data = data
    state = env.reset()
    env.step([1, None])
    env.step([1, None])

    # elev 1 should have loaded both passengers, gone up first, ind is 1
    assert env.building.elevators[0].indicator == 1
    assert len(env.building.elevators[0].passengers) == 2
    assert env.building.elevators[0].next_stop == (15, None)


if __name__ == '__main__':
    test_idle_pickup_samefloor()
