import copy
import logging
import typing

import numpy as np

import config
from environment import helper_functions
from environment.passenger import Passenger

if typing.TYPE_CHECKING:
    from environment.environment_continuous import ElevatorMetrics

logger = logging.getLogger(__name__)


class Elevator(object):
    """
    A class representing an elevator.

    Attributes:
        position (int): The current position of the elevator.
        capacity (int): The maximum number of passengers the elevator can hold.
        passengers (List[Passenger]): The passengers currently in the elevator.
        direction (int): The direction the elevator is moving: -1 for down, 0 for stationary, 1 for up.
        indicator (int): The direction the elevator is signalling: -1 for down, 0 for stationary, 1 for up.
    """

    floor_height = config.FLOOR_HEIGHT  # meters
    max_speed = config.MAX_SPEED  # m/s
    door_open_time = config.DOOR_OPEN_TIME  # seconds
    door_close_time = config.DOOR_CLOSE_TIME  # seconds
    board_time = config.BOARD_TIME  # seconds per passenger
    acceleration = config.ACCELERATION  # m/s^2
    timedelta = config.TIMEDELTA  # seconds
    capacity = config.CAPACITY
    movement_penatly = config.MOVEMENT_PENALTY
    waiting_penalty = config.WAITING_PENALTY
    arrival_reward = config.ARRIVAL_REWARD
    loading_reward = config.LOADING_REWARD
    reward_scheme = config.REWARD_SCHEME
    full_penalty = config.ELEVATOR_FULL_PENALTY

    def __init__(self, n_floors=15, rest_floor=None):
        """
        The constructor for the Elevator class.
        :param n_floors: The number of floors in the building.
        """
        self.time = None
        self.n_floors = n_floors
        self.rest_floor = rest_floor
        self.stop_thresholds = self.get_stopping_thresholds()

        self.passengers = []
        self.direction = 0  # -1 = down, 0 = stationary, 1 = up
        self.indicator = 0  # -1 = down, 0 = off,        1 = up
        self.speed = 0
        self.stopped = True
        self.idle = True
        self.destination_queue = set()
        self.waiting_for_car_call = False
        self.reopen_doors = False
        self.next_positions = []
        self.next_speeds = []
        self.next_events = []
        self.current_floor = 1 if self.rest_floor is None else self.rest_floor
        self.position = self.floor_height * self.current_floor
        self.next_stop = None
        self.reward = 0

    def __repr__(self):
        return f'Elevator(pos:{self.position}, ' \
               f'indicator: {"up" if self.indicator == 1 else "down" if self.indicator == -1 else "stationary"}, ' \
               f'{self.passengers=}, {self.next_stop=}, {self.speed=}'

    def reset(self):
        """
        Reset the elevator to its initial state.
        """
        self.passengers = []
        self.direction = 0
        self.indicator = 0
        self.speed = 0
        self.idle = True
        self.stopped = True
        self.destination_queue = set()  # set to avoid duplicates
        self.waiting_for_car_call = False
        self.reopen_doors = False
        self.next_positions = []
        self.next_speeds = []
        self.next_events = []
        self.current_floor = 1 if self.rest_floor is None else self.rest_floor
        self.position = self.floor_height * self.current_floor
        self.next_stop = None
        self.reward = 0

    @property
    def weight(self):
        return sum([p.weight for p in self.passengers])

    def add_to_destination_queue(self, passenger):
        """
        Add a passenger to the destination queue of the elevator, and update next_stop and indicator.
        Special case when a passenger arrives at the floor where the elevator is currently stopped or idling:
        then, dont add to queue but set flag waiting_for_car_call or reopen_doors.
        :param passenger: passenger to add to destination queue. Can be hall call or car call
        """

        # car call
        if passenger in self.passengers:

            # if the passenger in the elev. presses the button of the floor we are already on, just reopen doors.
            # Dont add to passenger list.
            if passenger.get_destination() == self.current_floor and self.stopped:
                self.reopen_doors = True
                return

            self.destination_queue.add((passenger.get_destination(), None))

        # hall call
        else:

            # when receiving calls for same floor, dont add to queue. Set indicator and wait for car call
            # this can be either when idling, or when stopped for another reason
            if passenger.get_floor() == self.current_floor:

                # during stop and if in valid direction, set indicator and wait for car call as well
                # -> ensures indicator doesnt change during stop, even if often redundant
                if self.stopped and self._is_direction_valid(passenger, 'continuous'):
                    self.indicator = 1 if passenger.direction == 'up' else -1
                    self.waiting_for_car_call = True

                    # Additionally, if the doors are already closing, we need to reopen them
                    # not relevant in case of idling
                    if not self.idle and 'load unload' not in self.next_events:
                        self.reopen_doors = True

                    return

            # in other cases, set indicator and next stop and proceed as usual
            self.destination_queue.add((passenger.get_floor(), passenger.direction))

        # given the queue has changed, update next stop
        self.set_next_stop()

    def set_next_stop(self):
        """
        Decide what the next stop of the elevator should be, and update the indicator accordingly.
        :return:
        """
        self.next_stop = self.decide_next_stop()
        # if stopped and waiting for a car call (we just served a hall call), indicator is already set -> blocked
        if not self.waiting_for_car_call:
            self.update_indicator(case='continuous')

    def decide_next_stop(self):
        """
        Decide what the next stop of the elevator should be, and update the indicator accordingly.
        The logic is that we empty the queue in the direction we are going, and then go to the other direction.
        If no one is going in the other direction, we go to the lowest/highest floor of the current direction
        and start again.
        :return:
        """

        if self.indicator == 1:

            # check if still people going up
            calls_above = [p for p in self.destination_queue if round(self.position, 4) <= self.stop_thresholds[
                (p[0], 'up', self.speed)]]  # will also include calls for current floor

            if valid_calls_above := [p for p in calls_above if p[1] == 'up' or p[1] is None]:
                next_stop = sorted(valid_calls_above, key=lambda x: x[0])[0]
            # if not, check if people going down
            # this includes people going down in all floors, and car calls for floors below current floor
            elif down_calls := [p for p in self.destination_queue if (p[1] == 'down' or p[1] is None)]:
                # select highest floor
                next_stop = sorted(down_calls, key=lambda x: x[0])[-1]
            # check if people going up but below current floor
            elif calls_below := [p for p in self.destination_queue if p[1] == 'up']:
                # go to lowest floor
                next_stop = sorted(calls_below, key=lambda x: x[0])[0]
            # if no calls at all
            else:
                next_stop = None

        elif self.indicator == -1:

            calls_below = [p for p in self.destination_queue if round(self.position, 4) >= self.stop_thresholds[
                (p[0], 'down', self.speed)]]

            if valid_calls_below := [p for p in calls_below if p[1] == 'down' or p[1] is None]:
                next_stop = sorted(valid_calls_below, key=lambda x: x[0])[-1]
            elif up_calls := [p for p in self.destination_queue if (p[1] == 'up' or p[1] is None)]:
                next_stop = sorted(up_calls, key=lambda x: x[0])[0]
            elif calls_above := [p for p in self.destination_queue if p[1] == 'down']:
                next_stop = sorted(calls_above, key=lambda x: x[0])[-1]
            else:
                next_stop = None

        # if the indicator is 0, we are idle, or we are stopped and the next stop is the current floor.
        else:

            if len(self.destination_queue) == 0:
                next_stop = None
            elif len(self.destination_queue) == 1:
                next_stop = list(self.destination_queue)[0]

            # As the indicator switches to either 1 or -1 when the first call is
            # registered, we dont expect to see more than one call in the queue when the indicator is 0
            else:
                # for now, pick call closest to current floor
                next_stop = sorted(self.destination_queue,
                                   key=lambda x: abs(x[0] * self.floor_height - self.position))[0]
                if next_stop[0] != self.current_floor:
                    logger.warning(f'{self.time}: multiple calls while indicator is 0, next stop is not current floor')

        return next_stop

    def update_indicator(self, case='discrete'):
        """
        Update the indicator to reflect the direction the elevator is moving. Direction is determined by the
        destination of the first passenger in the elevator. This corresponds to the little arrows on the elevator
        in real life.
        :param case: The case to run the environment in: 'discrete' or 'continuous'.
        """

        if case == 'continuous':

            if self.next_stop is None or self.next_stop[0] == self.current_floor:
                self.indicator = 0
            elif self.next_stop[0] > self.current_floor:
                self.indicator = 1
            elif self.next_stop[0] < self.current_floor:
                self.indicator = -1
            else:
                raise ValueError("?")

        else:

            # if no one wants to go higher
            if self.indicator == 1 and all([p.get_destination() <= self.position for p in self.passengers]):
                self.indicator = 0  # attained destination

            # if no one wants to go lower
            elif self.indicator == -1 and all([p.get_destination() >= self.position for p in self.passengers]):
                self.indicator = 0  # attained destination

            elif self.indicator == 0:
                if len(self.passengers) > 0:
                    if self.passengers[0].get_destination() > self.position:  # first passenger determines direction
                        self.indicator = 1
                    elif self.passengers[0].get_destination() < self.position:
                        self.indicator = -1

    def remove_from_destination_queue(self):
        """
         Removes a destination from the destination queue. Also responsible for setting waiting_for_car_call to True
         if removing a car call from queue
         This function is complicated because it is sometimes tricky to know what call we are serving. For example,
         if we arrive at floor 5 going up, we dont serve a hall call going down, unless there are no more calls going
         up.
         """

        # calls at current floor
        relevant_calls = [call for call in self.destination_queue if call[0] == self.current_floor]
        car_calls = [call for call in relevant_calls if call[1] is None]
        # halls calls in right direction
        hall_calls_to_remove = [call for call in relevant_calls if (call[1] == 'up' and self.indicator == 1) or
                                (call[1] == 'down' and self.indicator == -1)]

        # if we removed a hall call, that means we are serving a hall call and waiting for a car call
        # -> block indicator
        if hall_calls_to_remove:
            self.waiting_for_car_call = True
            self.indicator = 1 if hall_calls_to_remove[0][1] == 'up' else -1

        # calls to remove are car calls and hall calls in right direction
        calls_to_remove = car_calls + hall_calls_to_remove

        # remove calls
        self.destination_queue = {call for call in self.destination_queue if call not in calls_to_remove}

        # update next stop, because the dest. queue changed.
        # Will also update indicator if not waiting for car call
        self.set_next_stop()

        # case where we removed hall call / car call in current dir, but there are no more calls in
        # that direction. In that case, we should take hall call of opposite direction
        if self.next_stop is not None and self.next_stop[0] == self.current_floor:
            # get all calls in relevant_calls that are not in calls_to_remove
            other_dir = [call for call in relevant_calls if call not in calls_to_remove]
            if other_dir:
                self.waiting_for_car_call = True
                self.indicator = 1 if other_dir[0][1] == 'up' else -1
            self.destination_queue = {call for call in self.destination_queue if call not in other_dir}

            self.set_next_stop()

        # should not be possible, we explicitly remove first calls in right direction and then opposite,
        # none should remain at current floor
        if self.next_stop is not None and self.next_stop[0] == self.current_floor:
            logger.warning('next stop is still current floor')

    def infrastep(self, episode_data, floor_passengers, time):

        self.time = time
        self.reward = 0

        # idling: not moving, not handling (dis)embarking
        if self.idle:

            if self.waiting_for_car_call:
                self.idle = False
                self.compute_stop_trace(floor_passengers)
            elif self.next_stop is not None:
                self.idle = False
                self.stopped = False

        # stopped: handling (dis)embarking. if empty, go back to idling. Else, go to moving
        elif self.stopped:

            ended, floor_passengers = self.take_stop_step(episode_data, floor_passengers)

            if ended:

                # re-trigger stop if reopen_doors flag was set. Just recompute stop trace
                if self.reopen_doors:

                    self.reopen_doors = False
                    self.compute_stop_trace(floor_passengers)

                else:

                    # check if we were waiting for call car. If so, unblock indicator
                    if self.waiting_for_car_call:
                        self.waiting_for_car_call = False
                        # in case nobody got in, for some reason. Normally, updates next stop when car call registered,
                        # but if nobody got in, this is not done.
                        # The closing of the doors is deadline for waiting for car
                        # call, so if nobody got in, we must update next stop here
                        self.set_next_stop()

                    # if there are remaining calls, serve them
                    if self.next_stop is not None:
                        if self.next_stop[0] == self.current_floor:
                            # this happens is exceptional cases, where for example 1 passenger presses the Down button
                            # but pressed a higher floor than the current floor. Then, passenger 2 does not get in
                            # while the elevator actually will go up. The call remains in the queue, and after the end
                            # of the stop, this call becomes the next stop. The solution is to remove the call from
                            # dest. queue (fixes indicator) and compute stop trace again.
                            self.remove_from_destination_queue()
                            self.compute_stop_trace(floor_passengers)
                        else:
                            # go to move mode.
                            self.stopped = False

                    # else, go to rest floor or go to idle mode
                    else:

                        if self.rest_floor is None or self.current_floor == self.rest_floor:
                            self.idle = True
                        # go to rest floor
                        else:
                            # create dummy call to idle floor
                            rest_passenger = Passenger(self.rest_floor, None, None, None)
                            self.add_to_destination_queue(rest_passenger)
                            self.stopped = False

        # if moving
        else:

            # when moving, add movement penalty
            self.reward += self.movement_penatly
            episode_data.rewards_moving.append(self.movement_penatly)

            ended = self.take_move_step()  # episode_data)

            if ended:
                # update current floor: is updated everytime after arrival NOTE: means this is not accurate while moving
                self.current_floor = int(round((self.position / self.floor_height), 0))
                # assert self.next_stop[0] == self.current_floor
                self.stopped = True
                # remove current floor from destination queue. Can be hall call or car call
                self.remove_from_destination_queue()
                self.compute_stop_trace(floor_passengers)

        # add waiting penalty for each passenger waiting in elevator
        self.reward += self.waiting_penalty * len(self.passengers)
        episode_data.rewards_wait_elev.append(self.waiting_penalty * len(self.passengers))

        return self.reward, floor_passengers

    def _is_direction_valid(self, passenger, mode='discrete') -> bool:
        """
        Check if the direction of the elevator is valid for the passenger to enter.
        :param passenger: The passenger to check.
        :return: True if the passenger can enter, False otherwise.
        """
        if mode == 'discrete':
            if self.indicator == 0:
                return True
            elif self.indicator == 1 and passenger.get_destination() > self.position:
                return True
            elif self.indicator == -1 and passenger.get_destination() < self.position:
                return True
            else:
                return False

        elif mode == 'continuous':

            # sometimes, passengers press the wrong button
            if self.indicator == 1 and passenger.direction == 'up' and \
                    passenger.get_destination() <= self.current_floor:
                return True
            elif self.indicator == -1 and passenger.direction == 'down' and \
                    passenger.get_destination() >= self.current_floor:
                return True

            elif self.indicator == 0:
                return True
            elif self.indicator == 1 and passenger.direction == 'up':
                return True
            elif self.indicator == -1 and passenger.direction == 'down':
                return True
            else:
                return False
        else:
            raise ValueError(f'Invalid mode: {mode}')

    def load_unload(self, environment_passengers: list, episode_data: 'ElevatorMetrics', case='discrete') \
            -> (list, int):
        """
        Load and unload passengers from the elevator. Unloading happens first, then loading. Passengers are removed
        from the environment if they are loaded into the elevator.
        Also takes care of updating the indicator for the elevator.
        :param environment_passengers: The list of passengers in the environment.
        :param episode_data: The episode data object to add passenger data to.
        :param case: The case to run the environment in: 'discrete' or 'continuous'.
        :return: The updated list of passengers in the environment and the number of passengers unloaded.
        """

        # unload passengers
        arrived_passengers = [p for p in self.passengers if p.get_destination() == self.current_floor]
        for p in arrived_passengers:
            p.time_waited = (p.boarding_time - p.arrival_time).seconds
            p.time_travelled = (self.time - p.boarding_time).seconds
            episode_data.add_passenger_data(p.time_waited, p.time_travelled)
        num_unloaded = len(arrived_passengers)
        self.passengers = [p for p in self.passengers if p not in arrived_passengers]

        # update indicator so new passengers can enter
        # (in case of continuous, is already done when arriving at floor)
        if case == 'discrete':
            self.update_indicator(case='discrete')

        # load passengers
        passengers_to_load = []
        for p in environment_passengers:
            if len(self.passengers) >= self.capacity:
                logger.info(f'{self.time}: Elevator is full, cannot load more passengers')
                self.reward += self.full_penalty
                episode_data.rewards_full.append(self.full_penalty)
                break
            if p.get_floor() == self.current_floor and self._is_direction_valid(p, 'continuous'):
                self.passengers.append(p)
                # keep track of loaded passengers for reward
                passengers_to_load.append(p)
                p.boarding_time = self.time
                if case == 'continuous':
                    # update queue to include new passenger
                    self.add_to_destination_queue(p)

                elif case == 'discrete':
                    self.update_indicator(case='discrete')

        environment_passengers = [p for p in environment_passengers if p not in self.passengers]

        rew_load, rew_unload = self.process_rewards(arrived_passengers, passengers_to_load)
        self.reward += rew_load + rew_unload
        episode_data.rewards_loading.append(rew_load)
        episode_data.rewards_arrival.append(rew_unload)

        return environment_passengers, num_unloaded, len(passengers_to_load)

    def calculate_ETD_score(self, floor: int, direction: str):
        """
        calculate time till dest. floor, considering own direction and direction of call
        """

        # first, order destination queue:
        # first, calls in own direction and before thresholds, then opposite, then same dir but after thresholds

        destination_queue = copy.deepcopy(self.destination_queue)

        passeng = (floor, direction)
        already_serving = True if passeng in self.destination_queue else False
        destination_queue.add(passeng)

        if self.waiting_for_car_call:
            # add phantom passenger to queue
            destination_queue.add((self.current_floor, 'up' if self.indicator == 1 else 'down'))

        stop_time = 10  # seconds

        if self.indicator == 1:

            # first order the queue (first in dir, then opposite dir, then same dir but after thresholds)
            calls_above = [p for p in destination_queue if
                           (p[1] == 'up' or p[1] is None) and
                           # either above threshold or below threshold but decelerating
                           ((round(self.position, 4) <= self.stop_thresholds[(p[0], 'up', self.speed)]) or
                            ((self.position < p[0] * self.floor_height) and self.speed < self.max_speed))]
            calls_above.sort(key=lambda x: x[0])
            calls_other_dir = [p for p in destination_queue if p not in calls_above and
                               (p[1] == 'down' or p[1] is None)]
            calls_other_dir.sort(key=lambda x: x[0], reverse=True)
            calls_same_dir = [p for p in destination_queue if p not in calls_above and p not in calls_other_dir]
            calls_same_dir.sort(key=lambda x: x[0])

            if passeng in calls_above:
                # passenger can go straight to passeng (same dir)
                distance = passeng[0] * self.floor_height - self.position
                calls_before = calls_above[:calls_above.index(passeng)]
                pass_after = (([] if calls_above[-1] == passeng else calls_above[calls_above.index(passeng) + 1:]) +
                              calls_other_dir + calls_same_dir)
            elif passeng in calls_other_dir:
                # the elevator must first go to the highest floor, then to the passeng
                highest_floor = max([p[0] for p in destination_queue])
                # distance is distance to highest floor + distance from highest floor to passeng
                distance = (highest_floor * self.floor_height - self.position) + \
                           (highest_floor - passeng[0]) * self.floor_height
                calls_before = calls_above + calls_other_dir[:calls_other_dir.index(passeng)]
                pass_after = ([] if calls_other_dir[-1] == passeng else
                              calls_other_dir[calls_other_dir.index(passeng) + 1:]) + calls_same_dir
            elif passeng in calls_same_dir:
                # the elevator must first go to the highest floor of the opposite direction (if applicable), then
                # to the lowest floor of the same direction, then to the passeng
                lowest_floor = min([p[0] for p in destination_queue])
                if calls_above or calls_other_dir:
                    highest_floor = max([p[0] for p in destination_queue])
                    distance = (highest_floor * self.floor_height - self.position) + \
                               (highest_floor - lowest_floor) * self.floor_height + \
                               (passeng[0] - lowest_floor) * self.floor_height
                else:
                    # distance is distance to lowest floor + distance from lowest floor to passeng
                    distance = (self.position - lowest_floor * self.floor_height) + \
                               (passeng[0] - lowest_floor) * self.floor_height

                calls_before = calls_above + calls_other_dir + calls_same_dir[:calls_same_dir.index(passeng)]
                pass_after = ([] if calls_same_dir[-1] == passeng else
                              calls_same_dir[calls_same_dir.index(passeng) + 1:])
            else:
                raise ValueError('passenger not in destination queue')

            if distance < 0:
                distance = -distance
                logger.warning(f'negative distance for ETD calc: {distance}')

            # based on order of appearance of passeng in queue, calculate eta and penalty
            eta = distance / self.max_speed + len(calls_before) * stop_time
            penalty = 0 if already_serving else stop_time * len(pass_after)

        elif self.indicator == -1:

            calls_below = [p for p in destination_queue if
                           (p[1] == 'down' or p[1] is None) and
                           # above threshold or below threshold but decelerating
                           ((round(self.position, 4) >= self.stop_thresholds[(p[0], 'down', self.speed)]) or
                            ((self.position > p[0] * self.floor_height) and self.speed < self.max_speed))]
            calls_below.sort(key=lambda x: x[0], reverse=True)
            calls_other_dir = [p for p in destination_queue if p not in calls_below and
                               (p[1] == 'up' or p[1] is None)]
            calls_other_dir.sort(key=lambda x: x[0])
            calls_same_dir = [p for p in destination_queue if p not in calls_below and p not in calls_other_dir]
            calls_same_dir.sort(key=lambda x: x[0], reverse=True)

            if passeng in calls_below:
                distance = self.position - passeng[0] * self.floor_height
                calls_before = calls_below[:calls_below.index(passeng)]
                pass_after = (([] if calls_below[-1] == passeng else calls_below[calls_below.index(passeng) + 1:]) +
                              calls_other_dir + calls_same_dir)
            elif passeng in calls_other_dir:
                lowest_floor = min([p[0] for p in destination_queue])
                # distance is distance to lowest floor + distance from lowest floor to passeng
                distance = (self.position - lowest_floor * self.floor_height) + \
                           (passeng[0] - lowest_floor) * self.floor_height
                calls_before = calls_below + calls_other_dir[:calls_other_dir.index(passeng)]
                pass_after = ([] if calls_other_dir[-1] == passeng else
                              calls_other_dir[calls_other_dir.index(passeng) + 1:]) + calls_same_dir
            elif passeng in calls_same_dir:
                highest_floor = max([p[0] for p in destination_queue])
                if calls_below or calls_other_dir:
                    lowest_floor = min([p[0] for p in destination_queue])
                    distance = (self.position - lowest_floor * self.floor_height) + \
                               (highest_floor - lowest_floor) * self.floor_height + \
                               (highest_floor - passeng[0]) * self.floor_height
                else:
                    # distance is distance to highest floor + distance from highest floor to passeng
                    distance = (highest_floor * self.floor_height - self.position) + \
                               (highest_floor - passeng[0]) * self.floor_height

                calls_before = calls_below + calls_other_dir + calls_same_dir[:calls_same_dir.index(passeng)]
                pass_after = ([] if calls_same_dir[-1] == passeng else
                              calls_same_dir[calls_same_dir.index(passeng) + 1:])
            else:
                raise ValueError('passenger not in destination queue')

            if not distance >= 0:
                distance = -distance
                logger.warning(f'negative distance: {distance}')

            eta = distance / self.max_speed + len(calls_before) * stop_time
            penalty = 0 if already_serving else stop_time * len(pass_after)

        else:

            # assert self.destination_queue == set()
            distance = abs(passeng[0] * self.floor_height - self.position)
            eta = distance / self.max_speed
            penalty = 0

        return eta, penalty

    #
    # def compute_movement_trace(self, destination, position, speed):
    #     """
    #     compute positions between current positions and destination
    #     Uses self.position for current position, self.next_stop for destination
    #     vf = vi + at
    #     d = (v1 + v2) / 2 * t
    #     :return: list of positions and speeds
    #     """
    #
    #     # own cache function
    #     if (destination, position, speed) in self.compute_movement_trace_cache:
    #         return self.compute_movement_trace_cache[(destination, position, speed)]
    #
    #     speeds = []
    #     positions = []
    #
    #     reached_threshold = False
    #
    #     if position < destination * self.floor_height:
    #         # as long as not reached destination
    #         # round, because sometimes it stops just shy or goes just over
    #         while round(position, 3) < destination * self.floor_height:
    #             if position < self.stop_thresholds[(destination, 'up', speed)] and not reached_threshold:
    #                 # accelerate or maintain speed
    #                 # vf = vi + at (but max_speed if v > max_speed)
    #                 new_speed = min(speed + self.acceleration * self.timedelta, self.max_speed)
    #             else:
    #                 if not reached_threshold:  # make sure you only do this once
    #                     # correct position for overshooting
    #                     position = self.stop_thresholds[(destination, 'up', speed)]
    #                     reached_threshold = True
    #                 # decelerate
    #                 # vf = vi + at (but 0 if v < 0)
    #                 new_speed = max(speed - self.acceleration * self.timedelta, 0)
    #             # update position
    #             # d = (v1 + v2) / 2 * t
    #             position += (speed + new_speed) / 2 * self.timedelta
    #             # update speed
    #             speed = new_speed
    #
    #             positions.append(position)
    #             speeds.append(speed)
    #
    #     elif position > destination * self.floor_height:
    #         # as long as not reached destination
    #         while round(position, 3) > destination * self.floor_height:
    #             if position > self.stop_thresholds[(destination, 'down', speed)] and not reached_threshold:
    #                 # accelerate or maintain speed
    #                 new_speed = min(speed + self.acceleration * self.timedelta, self.max_speed)
    #             else:
    #                 if not reached_threshold:
    #                     # correct position for overshooting
    #                     position = self.stop_thresholds[(destination, 'down', speed)]
    #                     reached_threshold = True
    #                 # decelerate
    #                 new_speed = max(speed - self.acceleration * self.timedelta, 0)
    #             # update position
    #             position -= (speed + new_speed) / 2 * self.timedelta
    #             # update speed
    #             speed = new_speed
    #
    #             positions.append(position)
    #             speeds.append(speed)
    #     else:
    #         positions = []
    #         speeds = []
    #
    #     self.compute_movement_trace_cache[(destination, position, speed)] = [positions, speeds]
    #
    #     return positions, speeds

    # @cache
    def compute_next_pos_speed(self, destination, position, speed):  # , episode_data):
        """
        compute next position and speed between
        Uses self.position for current position, self.next_stop for destination
        vf = vi + at
        d = (v1 + v2) / 2 * t
        :return: next position and speed
        """

        if position < destination * self.floor_height:  # if going up
            if position < self.stop_thresholds[(destination, 'up', speed)]:  # if not past threshold
                # accelerate or maintain speed
                new_speed = min(speed + self.acceleration * self.timedelta, self.max_speed)
                new_pos = position + (speed + new_speed) / 2 * self.timedelta
                # episode_data.energy_consumption.append(self.acceleration)
            else:
                # decelerate
                new_speed = max(speed - self.acceleration * self.timedelta, 0)
                new_pos = position + (speed + new_speed) / 2 * self.timedelta
                # correct for overshooting
                if new_pos > destination * self.floor_height:
                    new_speed = 0
                    new_pos = destination * self.floor_height

        elif position > destination * self.floor_height:  # if going down
            if position > self.stop_thresholds[(destination, 'down', speed)]:
                # accelerate or maintain speed
                new_speed = min(speed + self.acceleration * self.timedelta, self.max_speed)
                new_pos = position - (speed + new_speed) / 2 * self.timedelta
                # episode_data.energy_consumption.append(self.acceleration)
            else:
                # decelerate
                new_speed = max(speed - self.acceleration * self.timedelta, 0)
                new_pos = position - (speed + new_speed) / 2 * self.timedelta
                # correct for overshooting
                if new_pos < destination * self.floor_height:
                    new_speed = 0
                    new_pos = destination * self.floor_height

        else:
            new_speed = 0
            new_pos = position
            assert new_pos == destination * self.floor_height

        return new_pos, new_speed

    def compute_stop_trace(self, floor_passengers):
        """
        computes total stopping time: opening doors, letting in and out passengers, closing doors
        List is list of events that happen during the stop
        :return:
        """

        # door opening
        opening_steps = int(self.door_open_time / self.timedelta)

        # amount of passengers that get in and out * constant
        passengers_to_board = len([p for p in floor_passengers if p.get_destination() == self.current_floor])
        passengers_to_unboard = len([p for p in self.passengers if p.get_destination() == self.current_floor])
        load_steps = int((1 + self.board_time * passengers_to_board) / self.timedelta)
        unload_steps = int((1 + self.board_time * passengers_to_unboard) / self.timedelta)

        # door closing
        closing_steps = int(self.door_close_time / self.timedelta)
        # only have 1 event for load unload, rest is empty (serves as cue for load_unload function)
        self.next_events = ['door open'] * opening_steps + ['load unload'] + [''] * (load_steps + unload_steps) + \
                           ['door close'] * closing_steps

    def take_move_step(self):  # , episode_data):

        self.position, self.speed = self.compute_next_pos_speed(self.next_stop[0], self.position, self.speed)
        # ,episode_data)

        # if moved, add acceleration to episode data

        if self.speed == 0 or self.position == self.next_stop[0] * self.floor_height:
            # correct for imprecisions
            self.position = self.next_stop[0] * self.floor_height
            self.speed = 0
            self.direction = 0
            return True

        self.direction = 1 if self.next_stop[0] * self.floor_height > self.position else -1

        return False

    def take_stop_step(self, episode_data, floor_passengers):

        if not self.next_events:
            return True, floor_passengers

        event = self.next_events.pop(0)
        if event == 'load unload':
            floor_passengers, num_unloaded, num_loaded = \
                self.load_unload(floor_passengers, episode_data, case='continuous')

        return False, floor_passengers

    def process_rewards(self, unloaded_pass, loaded_pass):

        if config.REWARD_SCHEME == 'TIME':
            # reward is negative of time waited by passengers that arrived
            # -> the more they wait, the worse. Reward is always negative
            rew_unload = sum([-(p.time_waited + p.time_travelled) for p in unloaded_pass]) * self.arrival_reward
            return (self.loading_reward * len(loaded_pass)), rew_unload

        elif config.REWARD_SCHEME == 'PASSENGER':
            # passenger arrival is a fixed reward
            return (self.loading_reward * len(loaded_pass)), (self.arrival_reward * len(unloaded_pass))

        else:
            raise NotImplementedError

    def get_stopping_thresholds(self) -> dict:
        """
        Returns a dictionary of stopping thresholds for every floor, dependent on direction and current speed.
        key is floor+direction+speed, value is x position of stopping threshold
        NOTE: floor starts at 0
        :return: dict of stopping thresholds
        """

        # precompute stopping thresholds for all possible speeds
        possible_speeds = np.arange(0, self.max_speed + self.acceleration * self.timedelta,
                                    self.acceleration * self.timedelta)

        # vf^2 = vi^2 + 2ad,
        # vf^2 = 0,
        # 0 = vi^2 + 2ad,
        # vi^2 = -2ad,
        # d = vi^2 / -2a ( but we discard the - because the acceleration is positive here)
        stopping_distances = possible_speeds ** 2 / (2 * self.acceleration)

        # custom dict to take 3 args and round float
        stopping_thresholds = helper_functions.FloatKeyDictionary(1)
        for ix, speed in enumerate(possible_speeds):
            for floor in range(self.n_floors):
                # going up
                stopping_thresholds[(floor, 'up', speed)] = \
                    floor * self.floor_height - stopping_distances[ix]
                # going down
                stopping_thresholds[(floor, 'down', speed)] = \
                    floor * self.floor_height + stopping_distances[ix]

        return stopping_thresholds
