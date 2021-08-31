import numpy as np
# from multiagent.environment import MultiAgentEnv
# import multiagent.scenarios as scenarios
# import gym_vecenv


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
