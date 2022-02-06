# from TD import EGreedyPolicy   
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import EGreedyPolicy
import gym
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def q_learning(env, num_episodes, 
                max_timestep,
                discount_factor, 
                alpha,
                start_epsilon,
                end_epsilon,
                decay_steps,
                type_of_decay='linear',
                seed=None):
    
    # Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    # while following an epsilon-greedy policy
    
    # The final action-value function.
    # A matrix that maps state-action pair (s,a) to action-value Q[s][a].
    Q = np.zeros((env.observation_space.n,env.action_space.n), dtype=np.float64)

    episode_reward=[]
    step_reward=[]
    episode_length=[]
    total_step=0
    policy=EGreedyPolicy.EpsilonGreedy(env,start_epsilon,end_epsilon,decay_steps,type_of_decay)

    for i_episode in range(num_episodes):
        
        # Reset the environment and pick the first state
        if seed is not None:
            env.seed(seed)
        current_state = env.reset()
        
        accumulate_reward=0.0

        for t in count():

            # env.render()
            # Take a step
            total_step+=1
            action = policy.return_action(current_state,Q,total_step)

            next_state, reward, done, _ = env.step(action)
            step_reward.append(reward)
            accumulate_reward=accumulate_reward+reward
            
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[current_state][action]
            Q[current_state][action] += alpha * td_delta
            current_state = next_state    
            
           
            if done or t>= max_timestep :
                
                episode_reward.append(accumulate_reward)
                episode_length.append(t)
                break
    print('total step of Q-table:'+str(total_step))
    return Q,episode_reward,episode_length,step_reward
