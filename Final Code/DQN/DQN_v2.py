import gym
import os
import random
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
import os
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import replaybuffer
import network
import EGreedyPolicy

# import torchvision.transforms as T


def DQN(env_name,
        hidden_dims,
        num_episodes,
        discount_factor,
        network_learning_rate,
        start_epsilon,
        end_epsilon,
        decay_steps,
        type_of_decay,
        with_target_Q,
        step_update_target,
        with_buffer,
        max_size,
        batch_size,
        seed=None):


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    #create environment
    env=gym.make(env_name)
    #create Q network and target Q network
    n_action=env.action_space.n
    n_obervation=env.observation_space.shape[0]

    Q_network = network.Estimator(input_dim=n_obervation,output_dim=n_action,hidden_dims=hidden_dims).to(device=device)
    if with_target_Q is True:
        Q_target_network=Q_network.to(device=device)

    # creat a replay buffer
    buffer=replaybuffer.ReplayBuffer(max_size,batch_size)

    #statics
    episode_reward=[]
    step_reward=[]
    episode_length=[]
    total_step=0

    policy=EGreedyPolicy.EpsilonGreedy(env,start_epsilon,end_epsilon,decay_steps,type_of_decay)

    for i_episode in range(num_episodes):
        
        # if i_episode%100==0:
        #     print(i_episode/num_episodes*100)
        # Reset the environment and pick the first state
        if seed is not None:
            env.seed(seed)
        current_state = env.reset()
        current_state=torch.from_numpy(current_state)
        accumulate_reward=0.0

        for t in count() :
            # env.render()
            # Take a step
            total_step+=1

            Q=Q_network(current_state)
            action = policy.return_action(Q,total_step)

            next_state, reward, done, _ = env.step(action)

            if with_buffer is True:
                experience = (current_state, action, reward, next_state, float(done))
                buffer.store(experience)
            
            step_reward.append(reward)
            accumulate_reward=accumulate_reward+reward

            next_state=torch.from_numpy(next_state)
            
            #update network
            if with_buffer is True:
                # print('with buffer')
                min_samples = buffer.batch_size * buffer.n_warmup_batches
                if len(buffer) > min_samples:
                    experiences = buffer.sample()
                    experiences = Q_network.load(experiences)
                    states, actions, rewards, next_states, is_terminals = experiences
                    if with_target_Q is True:
                        max_a_q_sp = Q_target_network(next_states).detach().max(1)[0].unsqueeze(1)
                    else:
                        max_a_q_sp = Q_network(next_states).max(1)[0].unsqueeze(1)
                    target_q_sa = rewards + (discount_factor * max_a_q_sp * (1 - is_terminals))
                    q_sa = Q_network(states).gather(1, actions)

                    td_error = q_sa - target_q_sa
                    value_loss = td_error.pow(2).mean()
                    value_optimizer=optim.RMSprop(Q_network.parameters(), lr=network_learning_rate)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()    
            else:
                # print('without buffer ')
                if with_target_Q is True:
                    # print('with target Q')
                    with torch.no_grad():
                        best_next_action=torch.argmax(Q_target_network(next_state)).item()  
                        td_target = reward + discount_factor * Q_target_network(next_state)[best_next_action]
                else:
                    # print('without target Q')
                    best_next_action = torch.argmax(Q_network(next_state)).item()    
                    td_target = reward + discount_factor * Q_network(next_state)[best_next_action]

                criterion = torch.nn.MSELoss(reduction='sum')
                # optimizer = torch.optim.RMSprop(Q_network.parameters(), lr=network_learning_rate)
                loss = criterion(Q_network(current_state)[action],td_target)
                Q_network.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(Q_network.parameters(), 
                #                 critic_max_grad_norm)
                with torch.no_grad():
                    for param in Q_network.parameters():
                        param -= network_learning_rate * param.grad
                # optimizer.step()


            current_state = next_state    
        


            if with_target_Q is True:
                if total_step%step_update_target==0:
                    Q_target_network=Q_network


            if done or t>200: 
                episode_reward.append(accumulate_reward)
                episode_length.append(t)
                break

    return episode_reward,step_reward,episode_length,total_step
