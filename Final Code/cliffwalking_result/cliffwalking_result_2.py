import gym
import numpy as np
import matplotlib.pyplot as plt


import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import plotting_results
import TD.Q_table as Q_table
import TD.Sarsa as Sarsa


env=gym.make('CliffWalking-v0')
num_episodes=400
max_timestep=100
discount_factor=1
lr=[2,1, 0.1, 0.01]
color=['darkorange', 'blue','g', 'c']
start_epsilon=0.1
end_epsilon=0.0001
decay_steps=1000
type_of_decay='exp'
SEED=[245,  80, 416, 491, 467]

fig, axs = plt.subplots(sharey=False, sharex=False)

for i in range(4):
    print(i)
    alpha=lr[i]
    
    print(color[i])

    #Q-learning
    episode_reward_result_Q_learning=[]
    step_reward_result_Q_learning=[]
    episode_length_result_Q_learning=[]

    for seed in SEED:
        Q,episode_reward,episode_length,step_reward=Q_table.q_learning(env, num_episodes, 
                        max_timestep,
                        discount_factor, 
                        alpha,
                        start_epsilon,
                        end_epsilon,
                        decay_steps,
                        type_of_decay,
                        seed)
        episode_reward_result_Q_learning.append(episode_reward)
        step_reward_result_Q_learning.append(step_reward)
        episode_length_result_Q_learning.append(episode_length)


    plotting_results.plot_only_return(result=episode_reward_result_Q_learning,n=10,axs=axs,color=color[i],label='lr='+str(alpha))
    #save data
    episode_reward_result=np.array(episode_reward_result_Q_learning)
    # step_reward_result_Q_learning=np.array(step_reward_result,dtype=float)
    episode_length_result=np.array(episode_reward_result_Q_learning)

    my_path = os.path.dirname(os.path.abspath(__file__))# Figures out the absolute path for you in case your working directory moves around.
    my_file = os.path.splitext(os.path.basename(__file__))[0]+'_'+'Q_table_lr='+str(alpha)+'.npz'
    np.savez(os.path.join(my_path, my_file), 
        episode_reward_result=episode_reward_result, 
        episode_length_result=episode_length_result)


    episode_reward_result_Q_learning=np.array(episode_reward_result_Q_learning)
    print('Overall performance:'+str(episode_reward_result_Q_learning.mean())+'+-'+str(episode_reward_result_Q_learning.std()))
episode_reward_result_Q_learning=np.array(episode_reward_result_Q_learning)
optimal=np.ones(episode_reward_result_Q_learning.shape[1])*(-13)

axs.plot(optimal,'-.' ,color='r', label='Threshold', linewidth=2)
axs.set_title('Performance of Tabular Q-learning in the Cliff-Walking environment')
axs.set_ylabel('Mean accumulated reward over past 10 episodes')    
axs.legend()
# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))


#random policy
start_epsilon=1
end_epsilon=1
episode_reward_result_random=[]
step_reward_result_random=[]
episode_length_result_random=[]

for seed in SEED:
    Q,episode_reward,episode_length,step_reward=Q_table.q_learning(env, num_episodes, 
                    max_timestep,
                    discount_factor, 
                    alpha,
                    start_epsilon,
                    end_epsilon,
                    decay_steps,
                    type_of_decay,
                    seed)
    episode_reward_result_random.append(episode_reward)
    step_reward_result_random.append(step_reward)
    episode_length_result_random.append(episode_length)

plotting_results.plot_only_return(result=episode_reward_result_random,n=10,axs=axs,color='black',label='random')
episode_reward_result_random=np.array(episode_reward_result_random)
print('Overall performance:'+str(episode_reward_result_random.mean())+'+-'+str(episode_reward_result_random.std()))

#save data
episode_reward_result=np.array(episode_reward_result_random)
# step_reward_result_Q_learning=np.array(step_reward_result,dtype=float)
episode_length_result=np.array(episode_reward_result_random)

my_path = os.path.dirname(os.path.abspath(__file__))# Figures out the absolute path for you in case your working directory moves around.
my_file = os.path.splitext(os.path.basename(__file__))[0]+'_'+'random'+'.npz'
np.savez(os.path.join(my_path, my_file), 
    episode_reward_result=episode_reward_result, 
    episode_length_result=episode_length_result)
