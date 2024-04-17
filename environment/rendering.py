import pygame

import config


class Rendering:

    def __init__(self, env, mode='discrete'):

        pygame.init()
        self.mode = mode
        self.env = env
        self.windowsize = (1300, 700)
        self.window = pygame.display.set_mode(self.windowsize)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 25)
        self.n_floors = env.n_floors
        self.n_elevators = env.n_elev
        self.floor_height = 0.8 * self.windowsize[1] / self.n_floors
        self.floor_width = 0.8 * self.windowsize[0] / self.n_elevators
        self.floor_y_positions = []
        self.floor_x_positions = []
        self.shaft_width = 0.2 * self.windowsize[0] / self.n_elevators
        self.shaft_x_positions = []
        self.elevator_boxes = []
        self.elevator_button_boxes = []
        self.floor_button_boxes = []
        self.floor_button_labels = []
        self.elevator_button_labels = []
        self.init_window()

    def init_window(self):

        # draw background
        self.draw_background()

        # draw elevators
        if self.mode == 'discrete':
            for i, elevator in enumerate(self.env.elevators):
                rect = pygame.Rect(self.shaft_x_positions[i][0],
                                   self.floor_y_positions[self.n_floors - elevator.position - 1][
                                       0] + 0.1 * self.floor_height,
                                   self.shaft_width,
                                   0.8 * self.floor_height)
                pygame.draw.rect(self.window, (255, 255, 255), rect)
                self.elevator_boxes.append(rect)

        elif self.mode == 'continuous':
            for i, elevator in enumerate(self.env.building.elevators):
                rect = pygame.Rect(self.shaft_x_positions[i][0],
                                   ((self.env.n_floors * config.FLOOR_HEIGHT - elevator.position - 3) /
                                    (self.env.n_floors * config.FLOOR_HEIGHT) * self.windowsize[1]) +
                                   0.1 * self.floor_height,
                                   self.shaft_width,
                                   0.8 * self.floor_height)
                pygame.draw.rect(self.window, (255, 255, 255), rect)
                self.elevator_boxes.append(rect)

        pygame.display.update()

    def render(self):
        """
        Renders the environment at every step
        :return: None
        """
        # if space is pressed, pause the simulation
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for event_ in pygame.event.get():
                            if event_.type == pygame.KEYDOWN:
                                if event_.key == pygame.K_SPACE:
                                    paused = False
                                    break

        if self.mode == 'discrete':
            self.clock.tick(2)
        else:
            self.clock.tick(config.TIME_SPEED_UP_FACTOR / config.TIMEDELTA)

        self.draw_background()

        if self.mode == 'continuous':
            self.env.floor_passengers = self.env.building.floor_passengers
            self.env.elevators = self.env.building.elevators
            self.env.floor_buttons = self.env.building.floor_buttons
            self.env.elev_buttons = self.env.building.elev_buttons

        # get elevator positions
        if self.mode == 'discrete':
            for i, elevator in enumerate(self.env.elevators):
                self.elevator_boxes[i].y = self.floor_y_positions[self.n_floors - elevator.position - 1][
                                               0] + 0.1 * self.floor_height
        elif self.mode == 'continuous':
            for i, elevator in enumerate(self.env.building.elevators):
                # position is in meters, convert to pixels
                self.elevator_boxes[i].y = ((self.env.n_floors * config.FLOOR_HEIGHT - elevator.position - config.FLOOR_HEIGHT) /
                                            (self.env.n_floors * config.FLOOR_HEIGHT) * self.windowsize[1]) + \
                                           0.1 * self.floor_height
        # draw elevators
        for rect in self.elevator_boxes:
            pygame.draw.rect(self.window, (255, 255, 255), rect)

        # draw passengers
        for floor in range(self.n_floors):
            # get amount of passengers on floor
            for i, passenger in enumerate(
                    [passenger for passenger in self.env.floor_passengers if passenger.floor == floor]):
                color = (255, 0, 0) if passenger.destination < floor else (0, 255, 0)
                pygame.draw.circle(self.window, color,
                                   (100 + 20 * i,
                                    self.floor_y_positions[self.n_floors - floor - 1][0] + 0.5 * self.floor_height),
                                   10)

        # draw elevator passengers
        for i, elevator in enumerate(self.env.elevators):
            self.window.blit(self.font.render('{}'.format(len(elevator.passengers)), True, (0, 0, 0)),
                             (self.elevator_boxes[i].x + 0.5 * self.elevator_boxes[i].width,
                              self.elevator_boxes[i].y + 0.5 * self.elevator_boxes[i].height))

        # draw floor buttons
        # get leftmost elevator x coord
        leftmost_elevator_x = min([rect.x for rect in self.elevator_boxes])
        for floor in range(self.n_floors):
            if self.env.floor_buttons[floor, 0]:
                pygame.draw.circle(self.window, (0, 255, 0),
                                   (leftmost_elevator_x - 30,
                                    self.floor_y_positions[self.n_floors - floor - 1][0] + 0.3 * self.floor_height),
                                   4)
            if self.env.floor_buttons[floor, 1]:
                pygame.draw.circle(self.window, (255, 0, 0),
                                   (leftmost_elevator_x - 30,
                                    self.floor_y_positions[self.n_floors - floor - 1][0] + 0.7 * self.floor_height),
                                   4)

        # draw elevator indicators
        for i, elevator in enumerate(self.env.elevators):
            if elevator.indicator == 1:
                rect = pygame.Rect(self.elevator_boxes[i].x + 0.3 * self.elevator_boxes[i].width,
                                   self.elevator_boxes[i].y + 0.1 * self.elevator_boxes[i].height,
                                   0.1 * self.elevator_boxes[i].width,
                                   0.1 * self.elevator_boxes[i].height)
                pygame.draw.rect(self.window, (0, 255, 0), rect)
            elif elevator.indicator == -1:
                rect = pygame.Rect(self.elevator_boxes[i].x + 0.6 * self.elevator_boxes[i].width,
                                   self.elevator_boxes[i].y + 0.1 * self.elevator_boxes[i].height,
                                   0.1 * self.elevator_boxes[i].width,
                                   0.1 * self.elevator_boxes[i].height)
                pygame.draw.rect(self.window, (255, 0, 0), rect)

        # draw exiting passengers
        # get rightmost elevator x coord
        if self.mode == 'discrete':
            rightmost_elevator_x = max([rect.x for rect in self.elevator_boxes]) + self.shaft_width
            for i, ix in enumerate(self.env.exiting_passengers):
                pygame.draw.circle(self.window, (255, 0, 0),
                                   (rightmost_elevator_x + 30 + 20 * i,
                                    self.floor_y_positions[self.n_floors - ix - 1][0] + 0.5 * self.floor_height),
                                   10)
        elif self.mode == 'continuous':
            rightmost_elevator_x = max([rect.x for rect in self.elevator_boxes]) + self.shaft_width
            # get all passengers inside elevators that are stopped, and are at their destination
            i = 0  # to prevent drawing on top of each other
            for elevator in self.env.elevators:
                for passenger in elevator.passengers:
                    if passenger.destination == elevator.current_floor and elevator.stopped:
                        pygame.draw.circle(self.window, (255, 0, 0),
                                           (rightmost_elevator_x + 30 + 20 * i,
                                            self.floor_y_positions[self.n_floors - passenger.destination - 1][0] + 0.5 * self.floor_height),
                                           10)
                        i += 1

        # draw current time
        if self.mode == 'continuous':
            self.window.blit(self.font.render('Time: {}'.format(self.env.time), True, (255, 255, 255)), (20, 20))

        # draw queues of elevator 1
        if self.mode == 'continuous':
            self.window.blit(self.font.render('Queue: {}'.format(self.env.elevators[0].destination_queue),
                                              True, (255, 255, 255)), (20, 40))

        # draw doors opening and closing
        if self.mode == 'continuous':
            if self.env.elevators[0].stopped and not self.env.elevators[0].idle:
                if self.env.elevators[0].next_events and self.env.elevators[0].next_events[0] == 'door open':
                    self.window.blit(self.font.render('Opening', True, (255, 255, 255)), (20, 60))
                elif self.env.elevators[0].next_events and self.env.elevators[0].next_events[0] == 'door close':
                    self.window.blit(self.font.render('Closing', True, (255, 255, 255)), (20, 60))

        # draw next_stop
        if self.mode == 'continuous':
            self.window.blit(self.font.render('Next stop: {}'.format(self.env.elevators[0].next_stop),
                                              True, (255, 255, 255)), (20, 80))

        # draw if waiting for hall call
        if self.mode == 'continuous':
            if self.env.elevators[0].waiting_for_car_call:
                self.window.blit(self.font.render('Waiting for hall call',
                                                  True, (255, 255, 255)), (20, 100))

        pygame.display.update()

    def draw_background(self):
        """Draws background for rendering."""

        if self.mode == 'continuous':
            self.env.elevators = self.env.building.elevators
            self.env.floor_passengers = self.env.building.floor_passengers

        # draw background
        self.window.fill((0, 0, 0))

        # draw shafts
        self.shaft_width = 60
        self.shaft_x_positions = [[self.window.get_width() / 2 - 20 - (self.shaft_width + 20) * (i + 1),
                                   self.window.get_width() / 2 - 20 - (self.shaft_width + 20) * (
                                           i + 1) + self.shaft_width]
                                  for i in range(len(self.env.elevators) // 2)] + \
                                 [[self.window.get_width() / 2 + 40 + (self.shaft_width + 20) * i,
                                   self.window.get_width() / 2 + 40 + (self.shaft_width + 20) * i + self.shaft_width]
                                  for i in range(len(self.env.elevators) // 2)]
        if len(self.env.elevators) % 2 == 1:
            self.shaft_x_positions.append(
                [(self.window.get_width() / 2) - (self.shaft_width / 2),
                 (self.window.get_width() / 2) + (self.shaft_width / 2)])

        for shaft in self.shaft_x_positions:
            rect = pygame.Rect(shaft[0], 0, self.shaft_width, self.window.get_height())
            pygame.draw.rect(self.window, (127, 127, 127), rect)

        # draw floors
        self.floor_height = self.windowsize[1] / self.n_floors
        self.floor_y_positions = [[self.floor_height * i, self.floor_height * (i + 1)] for i in range(self.n_floors+1)]

        [pygame.draw.line(self.window, (200, 200, 20), (0, self.floor_height * i),
                          (self.window.get_width(), self.floor_height * i), 1) for i in range(self.n_floors)]
