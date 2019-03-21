
import gym
import numpy as np
from gym import spaces
import time


#build the enviroment
env = gym.make('FrankaReacher-v0')
env.reset()
env.render()




#print infos
print("action space: tourg action on the right arm joints [1, 2, 7, 3, 4, 5, 6]")
print(env.action_space)
print("action upper bound")
print(env.action_space.high)
print("action lower bound")
print(env.action_space.low)


print("observationspace: ")
print(env.observation_space)
print("observationspace upper bound")
print(env.observation_space.high)
print("observationspace lower bound")
print(env.observation_space.low)




for _ in range(5000): # run for 1000 steps
    env.render()
    #get info from the world
    action = np.zeros(9) #do nothing
    #action = env.action_space.sample()*0.2 # pick a random action
    action[6]=-0.0
    action[7]=-0.8
    action[8]=-0.8
    #print("performing action:")
    print(action)
    #action=[0.20530831813812256, -27.898210525512695, 1.2628244161605835, 22.035024642944336, 0.11711292713880539, 1.5159826278686523, 0.01829119771718979, 0.0, 0.0]
    observation, reward, done, info = env.step(action)
    #time.sleep(0.1)
   # print(reward)
    #print("observing")
    #print(observation)
    #print("reward")
    #print(reward)
    