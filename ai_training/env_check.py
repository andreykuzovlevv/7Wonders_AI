from stable_baselines3.common.env_checker import check_env
from .sw_gym_env import SevenWondersEnv

env = SevenWondersEnv()

check_env(env)