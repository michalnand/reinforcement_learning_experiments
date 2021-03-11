import gym
import gym_anytrading

import numpy
import time

import RLAgents

import quantstats as qs
import pandas as pd

import models.ppo_baseline.src.model            as Model
import models.ppo_baseline.src.config           as Config


class ScoreWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

       
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward/100.0

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

def Wrapper(env):
    env = ScoreWrapper(env)
    
    return env

path = "models/ppo_baseline/"

config  = Config.Config()

envs = RLAgents.MultiEnvSeq('forex-v0', ScoreWrapper, config.actors)

agent = RLAgents.AgentPPO(envs, Model, Config) 

max_iterations = 10*(10**6) 

#trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 1000)
#trainig.run() 

agent.load(path)
agent.disable_training()

steps = 0
while True:
    history = envs.envs[0].history
    reward, done = agent.main()

    if steps%100 == 0:
        envs.render(0)  

    steps+= 1

    if done:
        break


df = gym_anytrading.datasets.FOREX_EURUSD_1H_ASK

start_index = 24
end_index   = len(df)
    

qs.extend_pandas()


net_worth = pd.Series(history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

print(returns)


qs.reports.full(returns)