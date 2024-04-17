import numpy as np

import environment.var_mapping as vm


class BaseAgent(object):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action = np.zeros(self.env.n_elev)

    def do_automatic_actions(self, info):
        for i, action in enumerate(info['default_action']):
            if action is not None:
                self.action[i] = action

    def correct_illegal_actions(self, elevator, state):
        if self.action[elevator] == vm.ACTION_UP and state['elev_position'][elevator] == self.env.n_floors - 1:
            self.action[elevator] = vm.ACTION_STAY
        elif self.action[elevator] == vm.ACTION_DOWN and state['elev_position'][elevator] == 0:
            self.action[elevator] = vm.ACTION_STAY


class ConventionalAgent(BaseAgent):

    def __init__(self, env):
        super().__init__(env)

    def act(self, state, info):

        self.action = np.zeros(self.env.n_elev)

        self.do_automatic_actions(info)

        # elevators that need action
        for elevator in [i for i in range(self.env.n_elev) if info['default_action'][i] is None]:

            # if stationary
            if state['elev_direction'][elevator] == 0:

                # if buttons pressed above:
                if np.any(state['elev_button'][elevator, state['elev_position'][elevator]:]):
                    self.action[elevator] = vm.ACTION_UP

                # if buttons pressed below:
                elif np.any(state['elev_button'][elevator, :state['elev_position'][elevator]]):
                    self.action[elevator] = vm.ACTION_DOWN

                # if no buttons pressed:
                else:
                    # go to nearest floor with button pressed
                    floors_with_buttons = np.unique(np.where(state['floor_button'] == 1)[0])
                    if len(floors_with_buttons) == 0:
                        self.action[elevator] = vm.ACTION_STAY
                        continue
                    nearest = np.argmin(np.abs(floors_with_buttons - state['elev_position'][elevator]))
                    self.action[elevator] = vm.ACTION_UP if \
                        floors_with_buttons[nearest] > state['elev_position'][elevator] else vm.ACTION_DOWN

                    # correct for case where elevator is at the top floor or bottom floor
                    self.correct_illegal_actions(elevator, state)

            # if moving up
            elif state['elev_direction'][elevator] == vm.DIR_UP:

                # if nobody above and elevator is empty, stop
                if np.sum(state['elev_button'][elevator]) == 0 and \
                        np.sum(state['floor_button'][state['elev_position'][elevator]:]) == 0:
                    self.action[elevator] = vm.ACTION_STAY
                    continue

                # if person would like to go up at the floor we are passing, stop
                if state['floor_button'][state['elev_position'][elevator], 0] == 1:
                    self.action[elevator] = vm.ACTION_STAY
                else:
                    self.action[elevator] = vm.ACTION_UP

            # if moving down
            elif state['elev_direction'][elevator] == vm.DIR_DOWN:

                # if nobody below and elevator is empty, stop
                if np.sum(state['elev_button'][elevator]) == 0 and \
                        np.sum(state['floor_button'][:state['elev_position'][elevator]]) == 0:
                    self.action[elevator] = vm.ACTION_STAY
                    continue

                # if person would like to go down at the floor we are passing, stop
                if state['floor_button'][state['elev_position'][elevator], 1] == 1:
                    self.action[elevator] = vm.ACTION_STAY
                else:
                    self.action[elevator] = vm.ACTION_DOWN

        return self.action


class SECTOR(BaseAgent):

    def __init__(self, env):
        super().__init__(env)
        self.sectors = self.assign_sectors()

    def assign_sectors(self):

        floors = np.arange(self.env.n_floors)
        return np.array_split(floors, self.env.n_elev)

    def act(self, state, info):

        self.action = np.zeros(self.env.n_elev)

        self.do_automatic_actions(info)

        for elevator in [i for i in range(self.env.n_elev) if info['default_action'][i] is None]:

            # if stationary
            if state['elev_direction'][elevator] == vm.DIR_STAY:

                # if elevator buttons pressed above:
                if np.any(state['elev_button'][elevator, state['elev_position'][elevator]:]):
                    self.action[elevator] = vm.ACTION_UP

                # if elevator buttons pressed below:
                elif np.any(state['elev_button'][elevator, :state['elev_position'][elevator]]):
                    self.action[elevator] = vm.ACTION_DOWN

                # if no buttons pressed, go to best floor in own sector
                else:
                    self.go_to_sector(elevator, state, direction=vm.DIR_STAY)

            # if moving up
            elif state['elev_direction'][elevator] == vm.DIR_UP:

                # if person would like to go up at the floor we are passing, stop
                if state['floor_button'][state['elev_position'][elevator], 0] == 1:
                    self.action[elevator] = vm.ACTION_STAY
                    continue

                # if nobody wants go go in or out, continue route
                elif np.sum(state['elev_button'][elevator]) > 0 and \
                        state['floor_button'][state['elev_position'][elevator], 0] == 0:
                    self.action[elevator] = vm.ACTION_UP

                # if elevator is empty, was going to own sector
                else:
                    self.go_to_sector(elevator, state, direction=vm.DIR_UP)

                self.correct_illegal_actions(elevator, state)

            # if moving down
            elif state['elev_direction'][elevator] == vm.DIR_DOWN:

                # if person would like to go down at the floor we are passing, stop
                if state['floor_button'][state['elev_position'][elevator], 1] == 1:
                    self.action[elevator] = vm.ACTION_STAY
                    continue

                # if nobody wants go go in or out, continue route
                elif np.sum(state['elev_button'][elevator]) > 0 and \
                        state['floor_button'][state['elev_position'][elevator], 1] == 0:
                    self.action[elevator] = vm.ACTION_DOWN

                # if elevator is empty, was going to own sector
                else:
                    self.go_to_sector(elevator, state, direction=vm.DIR_DOWN)

                self.correct_illegal_actions(elevator, state)

        return self.action

    def go_to_sector(self, elevator, state, direction):

        # get floors with buttons pressed in own sector
        floors_with_buttons = np.intersect1d(
            np.where(state['floor_button'] == 1)[0], self.sectors[elevator])

        if len(floors_with_buttons) > 0:

            # go to furthest floor within own sector
            # (this ensures that every floor within the sector is visited)
            target = int(np.argmax(np.abs(np.subtract(floors_with_buttons, state['elev_position'][elevator]))))

        # go chill in own sector if no buttons pressed
        # if no button pressed, get closest floor in own sector. This prevents oscillation
        else:
            floors_with_buttons = self.sectors[elevator]
            target = int(np.argmin(np.abs(floors_with_buttons - state['elev_position'][elevator])))

        # if already in own sector, chill out
        if floors_with_buttons[target] == state['elev_position'][elevator]:
            self.action[elevator] = vm.ACTION_STAY
        else:
            self.action[elevator] = vm.ACTION_UP if floors_with_buttons[target] \
                                                    > state['elev_position'][elevator] else vm.ACTION_DOWN

        # minor correction for illegal direction switches
        if direction == vm.DIR_UP and self.action[elevator] == vm.ACTION_DOWN:
            self.action[elevator] = vm.ACTION_STAY
        elif direction == vm.DIR_DOWN and self.action[elevator] == vm.ACTION_UP:
            self.action[elevator] = vm.ACTION_STAY

        # correct for case where elevator is at the top floor or bottom floor
        self.correct_illegal_actions(elevator, state)
