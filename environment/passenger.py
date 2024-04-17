import numpy as np


class Passenger(object):

    def __init__(self, floor, destination, arrival_time=None, direction=None):
        self.floor = floor
        self.destination = destination
        self.direction = direction
        self.arrival_time = arrival_time
        self.time_waited = 0
        self.time_travelled = 0
        self.weight = np.random.normal(75, 10)  # kg
        self.boarding_time = 0

    def __repr__(self):
        return f'Passenger(floor:{self.floor}, dest:{self.destination}, dir:{self.direction}) '

    def get_floor(self):
        # current floor
        return self.floor

    def get_destination(self):
        # final destination
        return self.destination

    def set_floor(self, floor):
        self.floor = floor

    def set_destination(self, destination):
        self.destination = destination

    def increment_time_waited(self):
        self.time_waited += 1

    def increment_time_travelled(self):
        self.time_travelled += 1
