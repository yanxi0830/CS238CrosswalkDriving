import matplotlib.pyplot as plt
import pandas as pd


def plot(raw_rewardsperep, save_path):
    plt.grid(linestyle='-.')
    returns_smoothed = pd.Series(raw_rewardsperep).rolling(10, min_periods=10).mean()
    plt.plot(raw_rewardsperep, linewidth=0.5, label='reward per episode')
    plt.plot(returns_smoothed, linewidth=2.0, label='smoothed reward (over window size=10)')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward v.s. Training Episode")
    plt.savefig(save_path)
