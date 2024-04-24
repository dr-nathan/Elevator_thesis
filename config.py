# Description: Configuration file for the environment

# general
NAME = 'Combinatorial - 2x busy'

# pygame
RENDER = False

# reproducibility
SET_SEED = False

# agent (for discrete case only)
AGENT = "DQNMulti"
AGENTS = ["DQNSingle", "DQNMulti", "conventional", "sector"]

# reward function
MOVEMENT_PENALTY = -0.01
WAITING_PENALTY = -0.01
ELEVATOR_FULL_PENALTY = -10
ARRIVAL_REWARD = 2
LOADING_REWARD = 2  # 2
ZERO_ELEV_PENALTY = -100  # when no elev responds to call. only possible in case of Branching assigning agent
DISCOUNTING_SCHEME = 'fixed'  # 'variable' | 'fixed'
BUSYNESS_MULTIPLIER = 2  # 1 means normal busy-ness, 0.5 means half as busy, etc.

FILL_WITH_BASELINE = False  # (unused, not implemented)

REWARD_SCHEME = 'PASSENGER'  # ['PASSENGER', 'SUM', 'TIME', 'SQUARED']
# Time means that the reward is the sum of the time elapsed since begin of
# button press / passenger arrival.
# Sum means that the reward is the sum of pressed buttons / passengers at time t only
# passenger means that the reward is the number of passengers that arrived at time t only
# time_squared means time, but waiting times are squared

# NN parameters
BATCH_SIZE = 32
LR = 5e-4
NN_SIZE = 'large'  # 'small' | 'large'
STATE_SIZE = 'large'  # 'small' | 'large'
STATE_POSITION = 'position'  # 'distance' | 'position'
STATE_ETD = 'ETD'  # 'STA' | 'ETD'  # STA = stops till available, ETD = estimated time to destination

# assigning network parameters
NN_TYPE_ASSIGN = 'duel_comb'  # 'duel_comb | 'comb' | 'branch'
MAX_ELEVS_RESPONDING = 2  # only relevant for comb and duel_comb
# next 2 lines only relevant in case of 'branch'
NN_ASSIGN_AGG = 'sum'  # 'sum' | 'mean' | 'none
NN_ASSIGN_USE_ADV = True

# zoning network parameters
LEARN_ZONING = False
NN_TYPE_ZONE = 'branching'  # 'branching' | 'sequential'
NN_ZONING_AGG = 'sum'  # 'sum' | 'mean'
NN_ZONING_USE_ADV = True

LEARN_INTERVAL = 10  # forward-backward pass every x steps
TARGET_NETWORK_UPDATE_INTERVAL = 300  # update target network every x steps
DISCOUNT_FACTOR = 0.95  # 0.95

NORMALIZE_STATE = True
NORMALIZE_REWARD = False
CLIP_GRADIENTS = True
LOSS_FN = 'Huber'  # 'MSE' | 'Huber'

# run parameters
N_EPISODES = 10000000

STEPS = 1000
RUN_TYPE: str = "multiple_runs"  # "single_run" | "multiple_runs"
RUNS = 20
TIMEDELTA = 0.1  # seconds
TIME_SPEED_UP_FACTOR = 10  # 1 means real time, 10 means 10x speedup. Only relevant for RENDER = True

# Building and elevator parameters
CAPACITY = 8
FLOOR_HEIGHT = 3  # meters
MAX_SPEED = 2.5  # m/s
DOOR_OPEN_TIME = 2  # seconds
DOOR_CLOSE_TIME = 2  # seconds
BOARD_TIME = 1  # seconds per passenger
ACCELERATION = 1  # m/s^2
