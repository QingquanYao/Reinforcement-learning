import numpy as np
import matplotlib.pyplot as plt
import os,sys 


#load data
episode_reward_result_AC=np.load("Final Code/cartpole_result/cartpole_result_AC_AC.npz")['episode_reward_result']
episode_reward_result_DQN=np.load('Final Code/cartpole_result/cartpole_result_DQN_DQN_with_targetQ_with_buffer.npz')['episode_reward_result']
episode_reward_result_random=np.load('Final Code/cartpole_result/pure_random_action.npz')['episode_reward_result']

# print('Pure random policy:'+'\n mean:'+str(episode_reward_result_random.mean())+'\n std:'+str(episode_reward_result_random.std()))


parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  

import plotting_results
#ploty
fig, axs = plt.subplots(3,1,sharey=False, sharex=True,figsize=(7,8)) 
fig.supylabel('Mean accumulated reward over past 10 episodes')
fig.supxlabel('Episodes')



plotting_results.plot_only_return(result=episode_reward_result_AC,n=10,axs=axs[0],color='g',label='AC',threshold=195)
episode_reward_result_AC=np.array(episode_reward_result_AC)
print('AC Overall performance:'+str(episode_reward_result_AC.mean())+'+-'+str(episode_reward_result_AC.std()))

plotting_results.plot_only_return(result=episode_reward_result_DQN,n=10,axs=axs[0],color='b',label='DQN',threshold=195)
episode_reward_result_DQN=np.array(episode_reward_result_DQN)
print('DQN Overall performance:'+str(episode_reward_result_DQN.mean())+'+-'+str(episode_reward_result_DQN.std()))

plotting_results.plot_only_return(result=episode_reward_result_random,n=10,axs=axs[0],color='darkorange',label='Random Policy')
episode_reward_result_random=np.array(episode_reward_result_random)
print('Random Overall performance:'+str(episode_reward_result_random.mean())+'+-'+str(episode_reward_result_random.std()))



axs[0].plot(np.ones(1000)*195,'-.',color='r',label='Threshold')
axs[0].set_title('Performances of DQN and AC in the cart-pole environment')
# axs[0].set_ylabel('Mean accumulated reward over past 10 episodes')   
# axs[0].legend()
# plt.subplots_adjust(hspace=1)




#load data
episode_reward_result_AC=np.load("Final Code/mountaincar_result/mountaincar_result_AC_AC.npz")['episode_reward_result']
episode_reward_result_DQN=np.load('Final Code/mountaincar_result/mountaincar_result_DQN_DQN_with_targetQ_with_buffer.npz')['episode_reward_result']
episode_reward_result_random=np.load('Final Code/mountaincar_result/pure_random_action.npz')['episode_reward_result']


plotting_results.plot_only_return(result=episode_reward_result_AC,n=10,axs=axs[1],color='g',threshold=195)
episode_reward_result_AC=np.array(episode_reward_result_AC)
print('AC Overall performance:'+str(episode_reward_result_AC.mean())+'+-'+str(episode_reward_result_AC.std()))

plotting_results.plot_only_return(result=episode_reward_result_DQN,n=10,axs=axs[1],color='b',threshold=195)
episode_reward_result_DQN=np.array(episode_reward_result_DQN)
print('DQN Overall performance:'+str(episode_reward_result_DQN.mean())+'+-'+str(episode_reward_result_DQN.std()))

plotting_results.plot_only_return(result=episode_reward_result_random,n=10,axs=axs[1],color='darkorange')
episode_reward_result_random=np.array(episode_reward_result_random)
print('Random Overall performance:'+str(episode_reward_result_random.mean())+'+-'+str(episode_reward_result_random.std()))



axs[1].plot(np.ones(1000)*(-110),'-.',color='r')
axs[1].set_title('Performances of DQN and AC in the mountain car environment')
# axs[1].set_ylabel('Mean accumulated reward over past 10 episodes')   
# axs[1].legend()


#load data
episode_reward_result_AC=np.load("Final Code/pendulum_result/pendulum_result_AC_AC.npz")['episode_reward_result']
episode_reward_result_DQN=np.load('Final Code/pendulum_result/pendulum_result_DQN_DQN_with_targetQ_with_buffer.npz')['episode_reward_result']
episode_reward_result_random=np.load('Final Code/pendulum_result/pure_random_action.npz')['episode_reward_result']


plotting_results.plot_only_return(result=episode_reward_result_AC,n=10,axs=axs[2],color='g',threshold=195)
episode_reward_result_AC=np.array(episode_reward_result_AC)
print('AC Overall performance:'+str(episode_reward_result_AC.mean())+'+-'+str(episode_reward_result_AC.std()))

plotting_results.plot_only_return(result=episode_reward_result_DQN,n=10,axs=axs[2],color='b',threshold=195)
episode_reward_result_DQN=np.array(episode_reward_result_DQN)
print('DQN Overall performance:'+str(episode_reward_result_DQN.mean())+'+-'+str(episode_reward_result_DQN.std()))

plotting_results.plot_only_return(result=episode_reward_result_random,n=10,axs=axs[2],color='darkorange')
episode_reward_result_random=np.array(episode_reward_result_random)
print('Random Overall performance:'+str(episode_reward_result_random.mean())+'+-'+str(episode_reward_result_random.std()))


axs[2].plot(np.ones(1000)*(-150),'-.',color='r')
axs[2].set_title('Performances of DQN and AC in the nverted pendulum environment')
# axs[2].set_ylabel('Mean accumulated reward over past 10 episodes')   
# axs[2].legend()


# plt.subplots_adjust(hspace=1)
fig.legend(loc='center right')





# #save plot
my_path = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.splitext(os.path.basename(__file__))[0]+'.pdf'
plt.savefig(os.path.join(my_path, my_file))



