import numpy as np
import gym
from gym import spaces


class SimpleCrosswalkEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, config):
        """
        HighwayETC Environment
        :param config: see config.py
        """
        super(SimpleCrosswalkEnv, self).__init__()

        # action space
        # ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
        self.actions_list = config['actions_list']
        self.action2delta = {0: 0., 1: 1., 2: 2., 3: -1., 4: -2.}
        self.action_space = spaces.Discrete(len(self.actions_list))

        # state space: [position, velocity]
        self.state_features = config['state_features']
        self.init_state = config['init_state']  # [0, 3]
        self.goal_state = config['goal_state']  # [19, 3]
        self.max_position = config['max_position']
        self.min_velocity = config['min_velocity']
        self.max_velocity = config['max_velocity']
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.max_position, self.max_velocity]),
                                            shape=(2,),
                                            dtype=np.float32)

        # rewards_dict
        self.rewards_dict = config['rewards_dict']

        # current state np.array([position, velocity])
        self.state = np.array(self.init_state)
        self.prev_state = self.state

        # ETC position (10 hard code)
        self.crosswalk_pos = config['crosswalk_pos']
        self.crosswalk_max_velocity = config['crosswalk_max_velocity']

    def reset(self):
        """
        Return observation as np.array
        :return: observation (np.array)
        """
        self.state = np.array(self.init_state)
        return self.state

    def _get_reward(self, action):
        # return reward, done
        reward = 0
        done = False

        # reward_goal
        if self.state[0] >= self.goal_state[0]:
            if self.state[1] == self.goal_state[1]:
                reward += self.rewards_dict['goal_with_good_velocity']
            else:
                reward += self.rewards_dict['goal_with_bad_velocity']
            done = True
        else:
            reward += self.rewards_dict['per_step_cost']

        # limit speed when close to ETC
        if self.prev_state[0] <= self.crosswalk_pos <= self.state[0]:
            # penalize overspeeding over ETC
            if self.state[1] > self.crosswalk_max_velocity:
                reward += self.rewards_dict['over_speed_near_crosswalk']

        return reward, done

    def step(self, action):
        # print(self.state)
        # 0=no_change, 1=speed_up, 2=speed_up_up, 3=slow_down, 4=slow_down_down

        # remember prev state
        self.prev_state = self.state

        # update velocity (cannot be negative)
        self.state[1] += self.action2delta[action]
        self.state[1] = max(self.state[1], 0)
        self.state[1] = min(self.state[1], self.max_velocity)

        # update position
        # new_pos = old_pos + velocity
        self.state[0] += self.state[1]
        self.state[0] = min(self.max_position, self.state[0])

        reward, done = self._get_reward(action)

        return self.state, reward, done, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # print(self.state)

    def close(self):
        pass
