import gym
from itertools import count
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from TD import EGreedyPolicy

def Sarsa_learning(env, num_episodes, 
                max_timestep,
                discount_factor, 
                alpha,
                start_epsilon,
                end_epsilon,
                decay_steps,
                type_of_decay='linear',
                seed=None):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        max_timestep: max number of time step in one episodes
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
        seed: to generate deterministic random number
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A matrix that maps state-action pair (s,a) to action-value Q[s][a].
    Q = np.zeros((env.observation_space.n,env.action_space.n), dtype=np.float64)

    episode_reward=[]
    step_reward=[]
    episode_length=[]
    policy=EGreedyPolicy.EpsilonGreedy(env,start_epsilon,end_epsilon,decay_steps,type_of_decay)


    for i_episode in range(num_episodes):
        
        # Reset the environment and pick the first state
        if seed is not None:
            env.seed(seed)
        current_state = env.reset()
        total_step=0
        accumulate_reward=0.0
        action = policy.return_action(current_state,Q,total_step)

        for t in count():

            # env.render()
            # Take a step
            total_step+=1
            next_state, reward, done, _ = env.step(action)
            step_reward.append(reward)
            accumulate_reward=accumulate_reward+reward
            
            
            # TD Update
            best_next_action =  policy.return_action(next_state,Q,total_step)

            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[current_state][action]
            Q[current_state][action] += alpha * td_delta
            current_state = next_state    
            action=best_next_action
            if done or t>= max_timestep : 
                episode_reward.append(accumulate_reward)
                episode_length.append(t)
                break
    print('total step of Sarsa:'+str(total_step))
    return Q,episode_reward,episode_length,step_reward
