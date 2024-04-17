
# Code base for my master thesis about improving the VU elevators using reinforcement learning

## Agents

The Agent map contains scripts for training and running RL agents in the VU elevator environment. Discrete folder
contains agents trained on the earlier discrete version of the environment, most recent experiments are in the
continuous folder. In the continuous folder, RLAgents.py is the training script, saving trained agent files and 
result plots in the Data map. 

neural_nets.py contains the neural network architectures used in the agents. 

To train a new RL agent, run agent/continuous_agents/RLagents_cont.py while setting 'train' to True
Setting 'train' to False will run the agent once on the env and extract detailed metrics on performance of the 
last trained agent. Set all parameters in config.py.

Baseline agents are determined in RuleBasedAgents.py.

## Comparing

The trained agents can be moved to comparing/agents_to_compare_cont to compare them against each-other 
and against the baseline agents. Run comparing/compareCont.py to compare the agents. Run 
comparing/CompareContRuleBasedAgents.py to compare all baseline agents against each-other.

## Environment

All scripts relevant for the functioning of the elevator environment are in the environment map. The most recent 
and relevant environment is built in environment_continuous.py, which uses building.py which in turns uses 
elevator.py for actual elevator logic. The environment script is also dependent on variables set in config.py, 
and uses functions that are located in environment/helper_functions.py.

environment/passenger.py takes care of the passenger logic, and environment/rendering.py is used for visualizing the
environment as a GUI.

To import new data, put the raw data file in environment/data/raw and run process_raw_data.py. This will create a 
JSON file in environment/data/JSON. Rename the relevant file to passenger_data.json, this is the one 
environment_continuous.py imports. The validation data can be decided manually, 
see agents/continuous_agents/RLagents_cont.py for an example.

## Config file

Some params are decided on-the-spot when needed for relevant functions, but most important run params are in
config.py.

## Plots

All results obtained via simulation are stored in the plots folder. The plots are created by the scripts in the
other modules.

## Other

The main_cont.py, main.py, param_search.py, parameter_tuning.py files and testing map were used for earlier experiments and are 
not relevant anymore. main_cont.py can still be used to quickly run a single agent on the environment and compare
it to baselines however. But this function was mostly taken over by the comparing module.

test.sh was used to be able to run experiments on the VU servers, but that turned out to be slower than running
experiments on my own computer.