import numpy as np
from rich import print

from environment.environment_discrete import DiscreteNocam
from environment.passenger import Passenger

env = DiscreteNocam(2, 4)


def test_datatypes():

    state = env.reset()
    action = np.array([2, 2])  # 2 = do nothing
    for _ in range(10):
        state, reward, done, info = env.step(action)

    assert isinstance(state, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert state['elev_position'].shape == (2,)
    assert state['elev_button'].shape == (2, 4)
    assert state['floor_button'].shape == (4, 2)
    assert state['time_of_day'] in [0, 1, 2]


def test_elevator_up_down():

    _ = env.reset()

    assert env.elevators[0].position == 0
    assert env.elevators[1].position == 0
    assert env.elevators[0].direction == 0
    assert env.elevators[1].direction == 0

    action = np.array([0, 0])  # 0 = go up

    _ = env.step(action)

    assert env.elevators[0].position == 1
    assert env.elevators[1].position == 1
    assert env.elevators[0].direction == 1
    assert env.elevators[1].direction == 1

    action = np.array([2, 2])  # stay

    _ = env.step(action)

    assert env.elevators[0].position == 1
    assert env.elevators[1].position == 1
    assert env.elevators[0].direction == 0
    assert env.elevators[1].direction == 0

    action = np.array([1, 1])  # 1 = go down

    _ = env.step(action)

    assert env.elevators[0].position == 0
    assert env.elevators[1].position == 0
    assert env.elevators[0].direction == -1
    assert env.elevators[1].direction == -1

    action = np.array([2, 2])  # stay

    _ = env.step(action)

    assert env.elevators[0].position == 0
    assert env.elevators[1].position == 0
    assert env.elevators[0].direction == 0
    assert env.elevators[1].direction == 0


def test_passenger_pickup_dropoff():

    env.reset()
    pazza = Passenger(0, 3)  # create passenger at floor 0, going to floor 3
    env.floor_passengers.append(pazza)

    env.elevators[0].position = 0
    env.elevators[0].direction = 0
    env.elevators[1].position = 3  # move other elevator away

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert pazza in env.elevators[0].passengers  # should be picked up
    assert pazza not in env.floor_passengers  # should be picked up

    assert env.elevators[0].indicator == 1  # should be signalling up

    env.elevators[0].position = 3
    env.elevators[0].direction = 0

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert pazza not in env.elevators[0].passengers  # should be dropped off
    assert env.elevators[0].indicator == 0  # should be signalling nothing


def test_elevator_full():

    env.reset()

    for _ in range(2 * env.elevators[0].capacity + 1):
        env.floor_passengers.append(Passenger(0, 3))

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert len(env.elevators[0].passengers) == env.elevators[0].capacity  # should be full
    assert len(env.elevators[1].passengers) == env.elevators[1].capacity  # should be full
    assert len([(passenger.get_floor() == 0) for passenger in env.floor_passengers]) == 1  # should be one left on floor

    assert env.elevators[0].indicator == 1  # should be signalling up
    assert env.elevators[1].indicator == 1  # should be signalling up

    env.elevators[0].position = 3
    env.elevators[1].position = 3

    action = np.array([2, 2])  # stay
    _= env.step(action, debugmode=True)

    assert len(env.elevators[0].passengers) == 0  # should be empty
    assert len(env.elevators[1].passengers) == 0  # should be empty

    env.reset()

    for _ in range(2 * env.elevators[0].capacity - 1):
        env.floor_passengers.append(Passenger(0, 3))

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert (len(env.elevators[0].passengers) == env.elevators[0].capacity) or \
           (len(env.elevators[1].passengers) == env.elevators[1].capacity)  # should be full
    assert (len(env.elevators[0].passengers) == env.elevators[0].capacity - 1) or \
           (len(env.elevators[1].passengers) == env.elevators[1].capacity - 1)  # should be alsmot full


def test_direction():

    env.reset()

    env.floor_passengers.append(Passenger(1, 0))

    env.elevators[0].position = 1

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert env.elevators[0].indicator == -1  # should be signalling down

    env.elevators[0].position = 0
    env.elevators[1].position = 3

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert env.elevators[0].indicator == 0  # should be signalling nothing

    env.floor_passengers.append(Passenger(0, 3))
    env.floor_passengers.append(Passenger(0, 3))
    env.floor_passengers.append(Passenger(0, -1))

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert len(env.elevators[0].passengers) == 2  # should not pick up the -1 passenger
    assert env.elevators[0].indicator == 1  # should be signalling up

    env.elevators[0].position = 2

    action = np.array([2, 2])  # stay
    _ = env.step(action, debugmode=True)

    assert env.elevators[0].indicator == 1  # should be signalling up

    # TODO: test if elevator can pick up more passengers on its way up


def test_mask_and_default_action():
    env.reset()

    env.elevators[0].position = 2
    env.elevators[0].direction = 1

    env.elevators[1].position = 3
    env.elevators[1].direction = -1

    env.floor_passengers.append(Passenger(3, 2))
    env.elevators[1].passengers.append(Passenger(3, 2))

    state, reward, done, info = env.step(np.array([0, 1]), debugmode=True)

    assert all(info['mask'][:, 0] == np.array([0, 0, 1]))
    assert all(info['mask'][:, 1] == np.array([0, 1, 1]))

    assert info['default_action'][1] == 2

    state, reward, done, info = env.step(np.array([2, 2]), debugmode=True)

    assert info['default_action'][0] == 1


def test_button_times():

    env.reset()

    env.floor_passengers.append(Passenger(1, 3))
    env.elevators[1].passengers.append(Passenger(3, 2))

    for _ in range(10):
        state, reward, done, info = env.step(np.array([2, 2]), debugmode=True)

    # make sure the floor button times are updated
    assert env.floor_button_times[1][0] == 10 # up button of floor 1
    assert sum(sum(env.floor_button_times)) == 10 # no other should be on
    assert env.elev_button_times[1][2] == 10 # 2nd floor button of elevator 1
    assert sum(sum(env.elev_button_times)) == 10 # no other should be on

    # pick up
    env.elevators[0].position = 1
    env.elevators[0].direction = 0

    # drop off
    env.elevators[1].position = 2
    env.elevators[1].direction = 0

    for _ in range(2):
        env.step(np.array([2, 2]), debugmode=True)

    assert env.floor_button_times[1][1] == 0 # this button should be 0
    assert sum(sum(env.floor_button_times)) == 0 # no other should be on
    assert env.elev_button_times[1][2] == 0 # this button should be 0
    assert env.elev_button_times[0][3] == 2


if __name__ == '__main__':
    test_datatypes()
    test_elevator_up_down()
    test_passenger_pickup_dropoff()
    test_elevator_full()
    test_direction()
    test_mask_and_default_action()
    test_button_times()
    print('[green]✔ All tests passed!')
