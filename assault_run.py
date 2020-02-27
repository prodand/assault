import time

import gym as gym
import numpy as np

from funs import INPUT_SIZE, build_input, prepare_input, prepare_action_rewards
from network import McNetwork

env = gym.make("Assault-ram-v0").env

network = McNetwork(INPUT_SIZE)

MAX_ABS_REWARD = 21


def evaluate(steps, reward, done, lives_diff):
    if done:
        return -21.0, True, True
    if lives_diff < 0:
        return -42.0, True, False
    if reward > 0:
        return reward, True, False
    if steps == 25:
        return -0.5, True, False
    return reward if reward != 0.0 else -0.5, False, False


if __name__ == '__main__':
    init_state = env.reset()
    frames_seq = list([init_state, init_state, init_state])
    lives = 4
    epoch = 1
    while True:
        rd = env.render()
        states = list()
        action_rewards = list()
        done = False
        steps = 0
        while True:
            input_frame = build_input(frames_seq)
            predictions = network.predict(input_frame)
            action = np.argmax(predictions)
            state, reward, done, info = env.step(action)
            steps += 1
            my_reward, end, reset = evaluate(steps, reward, done, info['ale.lives'] - lives)
            lives = info['ale.lives']

            frames_seq.pop(0)
            frames_seq.append(state)
            states.append(input_frame)
            action_rewards.append((action, my_reward / MAX_ABS_REWARD))

            print(epoch, reward, my_reward, predictions[action], action)
            env.render()
            if epoch % 50 == 0:
                time.sleep(0.004)
            if reset:
                init_state = env.reset()
                frames_seq = list([init_state, init_state, init_state])
                epoch += 1
            if end:
                break

        network.train(prepare_input(states), prepare_action_rewards(action_rewards))
