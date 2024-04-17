
params = {'loss_fn': ['Huber, MSE'],
          'state_representation': ['full', 'reduced'],
          'reward_scheme': ['time', 'passenger', 'sum', 'squared'],
          'learn_zoning': [True, False],
          'regularization': [None, 'L1', 'L2'],
          'NN_size': ['small', 'large'],
          'NN_type': ['comb', 'duel_comb'],
          'learning_rate': [5e-4, 5e-3],
          'discount_factor': [0.9, 0.99]}

# gridsearch is totally infeasible, so we must use evolutionary search methods




