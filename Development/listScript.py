import csv
import gym
import numpy as np
import universe # loads universe envs into gym registry
                # much less details for universe envs and adds 2K+ envs
                # comment out universe import if uneeded

# mostly combination of links 1 & 2 w/ action_space.sample() simple sampling provided
# 1. https://github.com/openai/gym/issues/106#issuecomment-226675545
# 2. https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from
# other refs:
# 3. https://discuss.openai.com/t/is-there-information-on-what-actions-and-observations-really-are/1259
#   esoteric games such as physics games (eg. Ant) don't seem to implement get_action_meanings()
#   can't test w/o getting mujoco trial or paid license
# 4. https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf

def SampleActionSpace(env):
    env_sample = []
    for i in range(100):
        ass = env.action_space.sample()
        env_sample.append(env.action_space.sample())
    return "[" + str(round(np.min(env_sample), 1)) + "..." + str(round(np.max(env_sample), 1)) + "]"


class NullE:
    def __init__(self):
        self.observation_space = self.action_space = self.reward_range = "N/A"

envall = gym.envs.registry.all()
with open('envList.csv','w') as csvfile:
    listwriter = csv.writer(csvfile, delimiter = ',')
    table ='Environment Id|Observation Space|Action Space|Reward Range|tStepL|Trials|rThresh|Action Meanings'
    listwriter.writerow(table.split('|'))

    for e in envall:
        action_meanings = "N/A"
        try:
            env = e.make()
            action_meanings = "N/A"
            try:
                action_meanings = env.unwrapped.get_action_meanings()
            except:
                act_space = str(env.action_space)
                if "universe" not in act_space and "Tuple" not in act_space:
                    action_meanings = SampleActionSpace(env)
        except:
            env = NullE()
            continue  # Skip these for now
        table = '{}|{}|{}|{}|{}|{}|{}|{}'.format(e.id,  # |{}|{}|{}
                                                      env.observation_space, env.action_space, env.reward_range,
                                                      e.timestep_limit, e.trials, e.reward_threshold,
                                                      str(action_meanings))  # ,
        listwriter.writerow(table.split('|'))
    print(table)