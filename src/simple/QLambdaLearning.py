import numpy as np
from SimpleCrosswalkEnv import *
from config import CONFIG
import random
from utils import plot
from collections import *

random.seed(1234)
np.random.seed(1234)


class QLamdaLearning:
    def __init__(self, env, lamb):
        self.env = env
        self.statespace = env.observation_space
        self.actionspace = env.action_space.n
        self.lr = 0.7  # 0.02 in original
        self.discountfactor = 0.95
        self.episode = 2000
        self.horizon = 100
        self.eps = 0.8
        self.decay_factor = 0.9
        position = int(env.max_position + 1)
        v = int(env.max_velocity - env.min_velocity + 1)
        self.Qmat = np.zeros((position * v, self.actionspace))
        self.Nmat = np.zeros((position * v, self.actionspace))
        self.lamb = lamb

    def state_to_index(self, state):
        index = state[1] + (env.max_velocity - env.min_velocity + 1) * state[0]
        return index

    def act(self, state, deterministic):
        if deterministic:
            action = np.argmax(self.Qmat[state, :])
        else:
            epsgreedy = random.uniform(0, 1)
            if epsgreedy > self.eps:
                action = np.argmax(self.Qmat[state, :])
            else:
                self.eps *= self.decay_factor
                action = self.env.action_space.sample()
        return action

    def update(self, curstate, curaction, reward, nextstate):
        nextu = np.amax(self.Qmat[nextstate])
        delta = reward + self.discountfactor * nextu - self.Qmat[curstate][curaction]
        self.Nmat[curstate][curaction] += 1
        a = self.actionspace
        s = self.Qmat.shape[0]
        for i in range(s):
            for j in range(a):
                self.Qmat[i][j] += self.lr * delta * self.Nmat[i][j]
                self.Nmat[i][j] = self.discountfactor * self.lamb * self.Nmat[i][j]

    def train(self):
        rewardperep = []
        for i in range(self.episode):
            state = self.state_to_index(self.env.reset())
            totalreward = 0
            for j in range(self.horizon):
                action = self.act(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.state_to_index(next_state)
                self.update(state, action, reward, next_state)
                totalreward += reward
                state = next_state
                if done == True:
                    break
            rewardperep.append(totalreward)

        plot(rewardperep, 'results/qlambdalearning_returns.png')


if __name__ == "__main__":
    random_runs = defaultdict(list)

    n_steps = 10000
    RANDOM_RUNS = 5

    for _ in range(RANDOM_RUNS):
        # Train
        env = SimpleCrosswalkEnv(CONFIG)
        model = QLamdaLearning(env, lamb=0.7)
        model.train()

        # Test
        env.reset()
        state = model.state_to_index(env.state)

        total_reward = 0
        step_to_goal = 0
        speeding = False  # check if speeding in the trajectory
        success = False
        for step in range(n_steps):
            action = model.act(state, deterministic=True)
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
