
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate, count 
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import plotting_results

env_name='MountainCar-v0'
env=gym.make(env_name)
num_episodes=1000
max_timestep=200

SEED = (12, 34, 56, 78, 90)
episode_reward_result=[]
step_reward_result=[]
episode_length_result=[]
total_step_result=[]
fig, axs = plt.subplots(sharey=False, sharex=False)

for seed in SEED:
    print(seed)
    episode_reward=[]
    episode_length=[]
    env.seed(seed)
    for i_episode in range(num_episodes):
        env.reset()
        accumulate_reward=0.0
        for t in count():
            action=env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            accumulate_reward=accumulate_reward+reward
            if done:
                episode_reward.append(accumulate_reward)
                episode_length.append(t)
                break


    episode_reward_result.append(episode_reward)
    # step_reward_result.append(step_reward)
    episode_length_result.append(episode_length)
    # total_step_result.append(total_step)



plotting_results.plot_only_return(result=episode_reward_result,n=10,axs=axs,color='r',label='AC')

#save data
episode_reward_result=np.array(episode_reward_result)
# step_reward_result_Q_learning=np.array(step_reward_result,dtype=float)
episode_length_result=np.array(episode_length_result)

my_path = os.path.dirname(os.path.abspath(__file__))# Figures out the absolute path for you in case your working directory moves around.
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.npz'
np.savez(os.path.join(my_path, my_file), 
    episode_reward_result=episode_reward_result, 
    episode_length_result=episode_length_result)


plt.subplots_adjust(hspace=1)
# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))
