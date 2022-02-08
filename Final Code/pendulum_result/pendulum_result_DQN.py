import gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import plotting_results
import DQN.DQN_v2 as DQN 

env_name='our_environments:discrete_singlependulum-v0'
env=gym.make(env_name)
hidden_dims=(512,128)
num_episodes=1000
# max_timestep=200
discount_factor=1.0
network_learning_rate=0.0005
start_epsilon=0.005
end_epsilon=1e-10
decay_steps=8000
type_of_decay='exp'
seed=20
SEED = (45,  8, 16, 78, 90)
with_target_Q=[True, True, False, False]
step_update_target=10

with_buffer=[True, False, False, True]
max_size=100000
batch_size=300

color=['b', 'r', 'g', 'y']
label=['DQN_with_targetQ_with_buffer','DQN_with_targetQ_without_buffer','DQN_without_targetQ_without_buffer','DQN_without_targetQ_with_buffer' ]
label=['1','2','3','4']

fig, axs = plt.subplots(sharey=False, sharex=False)

for i in range(4):
    print(i,with_target_Q[i],with_buffer[i],color[i],label[i])

    episode_reward_result=[]
    step_reward_result=[]
    episode_length_result=[]
    total_step_result=[]

    for seed in SEED:
        print(seed)
        episode_reward,step_reward,episode_length,total_step=DQN.DQN(env_name,
                hidden_dims,
                num_episodes,
                discount_factor,
                network_learning_rate,
                start_epsilon,
                end_epsilon,
                decay_steps,
                type_of_decay,
                with_target_Q[i],
                step_update_target,
                with_buffer[i],
                max_size,
                batch_size,
                seed)
    
        episode_reward_result.append(episode_reward)
        step_reward_result.append(step_reward)
        episode_length_result.append(episode_length)
        total_step_result.append(total_step)

    label=['1','2','3','4']
    # plotting_results.plot_subplot_episode_reward(result=episode_reward_result,n=10,axs=axs,i=0,color=color[i],label=label[i])
    # plotting_results.plot_subplot_episode_length(result=episode_length_result,n=10,axs=axs,i=1,color=color[i],label=label[i])
    plotting_results.plot_only_return(result=episode_reward_result,n=10,axs=axs,color=color[i],label=label[i])

    #save data
    episode_reward_result=np.array(episode_reward_result)
    # step_reward_result_Q_learning=np.array(step_reward_result,dtype=float)
    episode_length_result=np.array(episode_length_result)

    label=['DQN_with_targetQ_with_buffer','DQN_with_targetQ_without_buffer','DQN_without_targetQ_without_buffer','DQN_without_targetQ_with_buffer' ]
    my_path = os.path.dirname(os.path.abspath(__file__))# Figures out the absolute path for you in case your working directory moves around.
    my_file = os.path.splitext(os.path.basename(__file__))[0]+'_'+label[i]+'.npz'
    np.savez(os.path.join(my_path, my_file), 
        episode_reward_result=episode_reward_result, 
        episode_length_result=episode_length_result)


plt.subplots_adjust(hspace=1)
# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))

