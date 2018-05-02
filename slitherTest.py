import gym
import universe

env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)
observation_n = env.reset()