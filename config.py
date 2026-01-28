class Config:
    '''
    a template version for configurations
    '''

    SNR_NORM_FACTOR = 5.0
    ANGLE_NORM_FACTOR = 100

    TRAIN_PARAMS = {
        'actor_lr': 0.00001,
        'critic_lr': 0.001,
        'multiplier_lr': 0.00001,
        'mcchooser_lr': 0.001,
        'capacity': 100000,
        'batch_size': 128,
        'epsilon': 1,
        'sigma': 0.4,
        'r_1': 20000,
        'r_2': 50000,
        'initial_epsilon': 1,
        'final_epsilon': 0.01,
        'initial_sigma': 0.3,
        'final_sigma': 0,
    }

    AGENT_INIT = {
        'actor': {
            'bias': 0.2,
            'weight_min': -3e-3,
            'weight_max': 3e-3
        },
        'multiplier': {
            'bias': 1.0,
            'weight_min': -3e-3,
            'weight_max': 3e-3
        },
        'dqn': {
            'bias': 0.0,
            'weight_min': -3e-3,
            'weight_max': 3e-3
        }
    }