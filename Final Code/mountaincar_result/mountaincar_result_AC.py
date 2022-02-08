#AC
import gym
import numpy as np
import matplotlib.pyplot as plt
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import plotting_results
import AC.AC_v2 as AC 

env_name='MountainCar-v0'
env=gym.make(env_name)
if_action_discrete=True

hidden_dims=(512,128)
# hidden_dims=(32,32)
num_episodes=1000
max_timestep=200
discount_factor=1.0
actor_learning_rate=0.0009
critic_learning_rate=0.05
actor_nework_max_grad_norm=3
critic_max_grad_norm=3
entropy_loss_weight=10
start_epsilon=0.0000
end_epsilon=0.000000
decay_steps=30000
type_of_decay='constant'

SEED = (45,  8, 16, 78, 90)
episode_reward_result=[]
step_reward_result=[]
episode_length_result=[]
total_step_result=[]
fig, axs = plt.subplots(sharey=False, sharex=False)

for seed in SEED:
    print(seed)
    episode_reward,episode_length=AC.AC(env_name,
            if_action_discrete,
            hidden_dims,
            num_episodes,
            max_timestep,
            discount_factor,
            actor_learning_rate,
            actor_nework_max_grad_norm,
            critic_learning_rate,
            critic_max_grad_norm,
            entropy_loss_weight,
            start_epsilon,
            end_epsilon,
            decay_steps,
            type_of_decay,
            seed)

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
my_file = os.path.splitext(os.path.basename(__file__))[0]+'_'+'AC'+'.npz'
np.savez(os.path.join(my_path, my_file), 
    episode_reward_result=episode_reward_result, 
    episode_length_result=episode_length_result)


plt.subplots_adjust(hspace=1)
# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))
