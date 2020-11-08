from config import CONFIG
from utils import plot
import random
from JaywalkerCrosswalk import *
from collections import *

random.seed(1234)


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env):
        self.env = env
        self.episode = 2000
        self.horizon = 30

    def act(self, observation, reward, done):
        return self.env.action_space.sample()

    def state_to_index(self, state):
        # i, j, k == > i * (maxj + 1) * (maxk + 1) + j * (maxk + 1) + k
        max_i = env.max_position
        max_j = env.max_velocity
        max_k = 2
        i = state[0]
        j = state[1]
        k = state[2]
        index = i * (max_j + 1) * (max_k + 1) + j * (max_k + 1) + k
        return index

    def train(self):
        rewardperep = []
        for i in range(self.episode):
            totalreward = 0
            for j in range(self.horizon):
                action = self.act(0, 0, 0)
                next_state, reward, done, info = self.env.step(action)
                totalreward += reward
                if done:
                    break

            rewardperep.append(totalreward)

        plot(rewardperep, 'results/random_traffic_returns.png')


if __name__ == '__main__':
    random_runs = defaultdict(list)

    n_steps = 10000
    RANDOM_RUNS = 5

    for _ in range(RANDOM_RUNS):
        # Train
        env = JaywalkerCrosswalkEnv(CONFIG)
        model = RandomAgent(env)
        model.train()

        # Test
        env.reset()
        state = model.state_to_index(env.state)

        total_reward = 0
        step_to_goal = 0
        speeding = False  # check if speeding in the trajectory
        success = False
        for step in range(n_steps):
            action = model.act(state, 0, 0)
            # print("Step {}".format(step + 1))
            # print("Action: ", env.actions_list[action])
            state, reward, done, info = env.step(action)
            state = model.state_to_index(state)
            total_reward += reward

            # speeding if reward < -3
            if reward < -3:
                speeding = True

            # print('obs=', state, 'current reward=', reward, 'total reward=', total_reward, 'done=', done)
            # env.render(mode='console')
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                # print("Goal reached!", "reward=", reward, 'total reward=', total_reward)
                step_to_goal = step
                if reward == 40:
                    success = True
                break

        random_runs['total_reward'].append(total_reward)
        random_runs['steps'].append(step_to_goal)
        random_runs['speeding'].append(1 if speeding else 0)
        random_runs['success'].append(1 if success else 0)

    print(random_runs)
    # calculate average total reward
    avg_reward = sum(random_runs['total_reward']) / RANDOM_RUNS
    print("Average Total Reward: {}".format(avg_reward))
    avg_steps = sum(random_runs['steps']) / RANDOM_RUNS
    print("Average Steps: {}".format(avg_steps))
    sucess_rate = sum(random_runs['success']) / RANDOM_RUNS
    print("Success Rate: {}".format(sucess_rate))
    speeding_rate = sum(random_runs['speeding']) / RANDOM_RUNS
    print("Speeding Rate: {}".format(speeding_rate))
