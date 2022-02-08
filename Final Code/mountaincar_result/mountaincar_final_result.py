import numpy as np
import matplotlib.pyplot as plt
import os,sys 

cwd = os.getcwd() 
print(cwd)
#load data
episode_reward_result_AC=np.load("Final Code/mountaincar_result/mountaincar_result_AC_AC.npz")['episode_reward_result']
episode_reward_result_DQN=np.load('Final Code/mountaincar_result/mountaincar_result_DQN_DQN_with_targetQ_with_buffer.npz')['episode_reward_result']
episode_reward_result_random=np.load('Final Code/mountaincar_result/pure_random_action.npz')['episode_reward_result']

# print('Pure random policy:'+'\n mean:'+str(episode_reward_result_random.mean())+'\n std:'+str(episode_reward_result_random.std()))


parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import plotting_results
#ploty
fig, axs = plt.subplots(sharey=False, sharex=False)

plotting_results.plot_only_return(result=episode_reward_result_AC,n=10,axs=axs,color='g',label='AC',threshold=195)
episode_reward_result_AC=np.array(episode_reward_result_AC)
print('AC Overall performance:'+str(episode_reward_result_AC.mean())+'+-'+str(episode_reward_result_AC.std()))

plotting_results.plot_only_return(result=episode_reward_result_DQN,n=10,axs=axs,color='b',label='DQN',threshold=195)
episode_reward_result_DQN=np.array(episode_reward_result_DQN)
print('DQN Overall performance:'+str(episode_reward_result_DQN.mean())+'+-'+str(episode_reward_result_DQN.std()))

plotting_results.plot_only_return(result=episode_reward_result_random,n=10,axs=axs,color='darkorange',label='Random Policy')
episode_reward_result_random=np.array(episode_reward_result_random)
print('Random Overall performance:'+str(episode_reward_result_random.mean())+'+-'+str(episode_reward_result_random.std()))



plt.plot(np.ones(1000)*(-110),'-.',color='r',label='Threshold')
axs.set_title('Performances of DQN and AC in the mountain car environment')
axs.set_ylabel('Mean accumulated reward over past 10 episodes')   
axs.legend()
plt.subplots_adjust(hspace=1)
# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))



