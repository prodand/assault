import numpy as np

INPUT_SIZE = 128
ACTION_SIZE = 7


def build_input(frames):
    return np.array([frames[0]]).reshape((1, INPUT_SIZE)) / 255


def prepare_input(states):
    result = states[0].reshape(INPUT_SIZE, 1)
    for i in range(1, len(states)):
        result = np.column_stack((result, states[i].reshape(INPUT_SIZE, 1)))
    return result.T


def prepare_action_rewards(action_rewards):
    size = len(action_rewards)
    discounted_rewards = np.zeros((size, 2), dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, (action, rw) in enumerate(reversed(action_rewards)):
        discounted_reward = rw + discounted_reward * gamma
        discounted_rewards[size - index - 1] = [action, discounted_reward]
    return discounted_rewards
