import numpy as np

import config
from .elevator import Elevator


class Building:

    def __init__(self, n_floors, n_elev):
        self.n_floors = n_floors
        self.n_elev = n_elev
        self.elevators = [Elevator(n_floors) for _ in range(n_elev)]

        self.floor_passengers = []
        self.elev_passengers = []
        self.floor_buttons = np.zeros((n_floors, 2))
        self.elev_buttons = np.zeros((n_elev, n_floors))

    def reset(self):

        self.floor_passengers = []
        self.elev_passengers = []
        self.floor_buttons = np.zeros((self.n_floors, 2))
        self.elev_buttons = np.zeros((self.n_elev, self.n_floors))

        for elevator in self.elevators:
            elevator.reset()

    def update_destinations(self, actions: np.ndarray, passenger):
        """
        updates the destination queue of the elevator that is assigned to the passenger
        :param actions: array of actions, one for each elevator. Expects 0 or 1.
        :param passenger: passenger object, used to determine the floor
        """
        for i, elevator in enumerate(self.elevators):
            if actions[i] == 1:
                elevator.add_to_destination_queue(passenger)

    def infra_step(self, episode_data, time):
        """
        processes lift behaviour after detination queues have been updated.
        :param episode_data: data structure to store episode data
        :param time: current time
        :return: reward, episode_data
        """
        reward = 0

        for elev in self.elevators:
            rew, self.floor_passengers = elev.infrastep(episode_data, self.floor_passengers, time)
            if config.RENDER:
                self.update_buttons()
            reward += rew

        # add floor passenger waiting time to reward of individual elevators
        reward += len(self.floor_passengers) * config.WAITING_PENALTY
        episode_data.rewards_wait_floor.append(len(self.floor_passengers) * config.WAITING_PENALTY)

        return reward

    def update_buttons(self):

        self.floor_buttons = np.zeros((self.n_floors, 2))

        for passenger in self.floor_passengers:
            if passenger.direction == 'up':
                self.floor_buttons[passenger.floor, 0] = 1
            elif passenger.direction == 'down':
                self.floor_buttons[passenger.floor, 1] = 1

        # update buttons in elevators
        self.elev_buttons = np.zeros((self.n_elev, self.n_floors))
        for i, elevator in enumerate(self.elevators):
            for passenger in elevator.passengers:
                self.elev_buttons[i, passenger.destination] = 1

    def get_stranded_passengers(self) -> list:
        """
        returns a list of passengers who are stranded on a floor because the
        elevator they were assigned to is full, or any other random problem.
        A passenger is stranded when no elevator contains the passenger's floor in their destination queue.
        :return: list of passengers
        """

        stranded_passengers = [passenger for passenger in self.floor_passengers if not any(
            (passenger.floor in [destination[0] for destination in elevator.destination_queue]
             or passenger.floor == elevator.current_floor)
            for elevator in self.elevators)]

        return stranded_passengers

    def set_zones(self, zones: np.ndarray):
        """ set zones decided by agent
        :param zones: array of zone indices, one for each elevator
        """
        for i, elevator in enumerate(self.elevators):
            # if agent chose 17 (no zone), set rest_floor to None
            elevator.rest_floor = zones[i] if zones[i] != 17 else None

    def get_elevator_distances(self, pass_floor) -> list:
        """
        for each elevator, gets the distance to the passenger
        :return:
        """

        positions = [elevator.position for elevator in self.elevators]  # in meters
        pass_floor = pass_floor * config.FLOOR_HEIGHT  # in meters
        distances = pass_floor - np.asarray(positions)  # positive if elevator is below passenger
        return distances

    def get_elevator_positions(self) -> list:
        """
        for each elevator, gets the distance to the passenger
        :return:
        """

        # convert position to floor number
        positions = [elevator.position for elevator in self.elevators]
        positions = [int(position / config.FLOOR_HEIGHT) for position in positions]
        return positions

    def get_elevator_speeds(self) -> list:

        speeds = [elevator.speed for elevator in self.elevators]
        # + for up, - for down
        speeds = [self.elevators[i].direction * speed for i, speed in enumerate(speeds)]
        return speeds

    def get_elevator_weights(self) -> list:
        weights = [elevator.weight for elevator in self.elevators]
        return weights

    def in_current_dir(self, floor, direction) -> bool:
        """
        returns the number of stops until the elevator is available for a new passenger
        :return: list of integers
        """
        for elevator in self.elevators:
            if elevator.indicator == 1 and direction == 'up' and elevator.position < floor:
                return True
            elif elevator.indicator == -1 and direction == 'down' and elevator.position > floor:
                return True

    def destination_queue_length(self) -> list:
        """
        returns the number of stops until the elevator is available for a new passenger
        :return: list of integers
        """
        return [len(elevator.destination_queue) for elevator in self.elevators]

    def get_ETDs(self, floor, direction) -> list:
        """
        returns the estimated time to destination for each elevator
        :return: list of floats
        """
        ETD = [sum(elevator.calculate_ETD_score(floor, direction)) for elevator in self.elevators]
        return ETD
