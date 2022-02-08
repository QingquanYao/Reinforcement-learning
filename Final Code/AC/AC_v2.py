import gym
import os
import random
import numpy as np
from itertools import count
# import matplotlib.pyplot as plt

# from tqdm import tqdm

import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import network
import EGreedyPolicy

# env_name='Pendulum-v1'
# # env_name='CartPole-v0'
# # env_name="our_environments:discrete_singlependulum-v0"
# env=gym.make(env_name)
# if_action_discrete=False

# hidden_dims=(128,256,512,256,128)
# # hidden_dims=(32,32)
# num_episodes=5000
# max_timestep=200
# discount_factor=1.0
# actor_learning_rate=0.000001
# actor_nework_max_grad_norm=3
# critic_learning_rate=0.000001
# critic_max_grad_norm=3
# entropy_loss_weight=0.001
# start_epsilon=1
# end_epsilon=0.1
# decay_steps=10000
# type_of_decay='linear'
# seed=20

def AC(env_name,
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
        seed):

    env=gym.make(env_name)
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

    if if_action_discrete is True:
        n_action=env.action_space.n
    else:
        n_action=env.action_space.shape[0]

    n_obervation=env.observation_space.shape[0]


    if if_action_discrete is True:
        Actor_network = network.NeuralNetwork(input_dim=n_obervation,output_dim=n_action, hidden_dims=hidden_dims).to(device=device)
    else:
        Actor_network = network.NeuralNetwork(input_dim=n_obervation,output_dim=2,hidden_dims=hidden_dims).to(device=device)

    Critic_network = network.NeuralNetwork(input_dim=n_obervation,output_dim=1,hidden_dims=hidden_dims).to(device=device)


    # if with_target_Q is True:
    #     Q_target_network=Q_network.to(device=device)

    # # creat a replay buffer
    # buffer=replaybuffer.ReplayBuffer(max_size,batch_size)

    #statics
    episode_reward=[]
    step_reward=[]
    episode_length=[]
    total_step=0


    if seed is not None:
        env.seed(seed)

    average_reward=0
    action_selection=EGreedyPolicy.EpsilonGreedy(env,start_epsilon,end_epsilon,decay_steps,type_of_decay)

    for i_episode in range(num_episodes):
        # print(i_episode)
        current_state = env.reset()
        current_state=torch.from_numpy(current_state)
        accumulate_reward=0.0
        # I=1
        for t in count():
            # print(t)
            # env.render()
            # t=1
            if if_action_discrete is True:
                action_preferences=Actor_network(current_state)
                # policy=torch.div(torch.exp(action_preferences),torch.sum(torch.exp(action_preferences)))
                # policy=torch.nn.Softmax(action_preferences)
                policy=torch.distributions.Categorical(logits=action_preferences)
                action=policy.sample()
                log_policy=policy.log_prob(action)
                action=action.item()
                action=action_selection.return_action(i_episode,action,random_action=env.action_space.sample())
                entropy=policy.entropy()
                

            else:
                # print(t)
                # print('current state:')
                # print(current_state)
                # print(Actor_network.state_dict())
                mean, std=Actor_network(current_state)
                # print(mean)s
                std=torch.exp(std)
                print(std,mean)
                policy=torch.distributions.normal.Normal(mean, std)
                action=policy.sample()
                log_policy=policy.log_prob(action).unsqueeze(-1)
                action=action.item()
                action=np.array([action])
                action=action_selection.return_action(i_episode,action,random_action=env.action_space.sample())
                
                entropy=policy.entropy().unsqueeze(-1)
                
            
            # print(action)
            # print(type(action))
            next_state, reward, done, _ = env.step(action)
            step_reward.append(reward)
            accumulate_reward=accumulate_reward+reward
            if t>0:
                average_reward=np.array(accumulate_reward/t)
                average_reward=np.average(average_reward)
        
            
            next_state=torch.from_numpy(next_state)
            if done:
                V_next=0
            else:
                V_next=Critic_network(next_state)

            TD_error=(reward+discount_factor*V_next)-Critic_network(current_state)
            # average_reward=average_reward+alpha_R*TD_delta
            Critic_loss=TD_error.pow(2).mean()

            Critic_network.zero_grad()
            Critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(Critic_network.parameters(), 
                            critic_max_grad_norm)
            with torch.no_grad():
                for param in Critic_network.parameters():
                    param -= critic_learning_rate * param.grad
            # Critic_optimizer = torch.optim.RMSprop(Critic_network.parameters(), lr=critic_learning_rate, momentum=0.9)
            # # Zero gradients, perform a backward pass, and update the weights.
            # Critic_optimizer.zero_grad()
            # Critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(Critic_network.parameters(), 
            #                             critic_max_grad_norm)
            # Critic_optimizer.step()

            # print('TD_error:')
            # print(TD_error.squeeze().detach())
            # Actor_loss=-(I*TD_error.squeeze().detach()*log_policy+entropy.detach()*entropy_loss_weight).mean()
            
            # Actor_loss=-(I*TD_error.squeeze().detach()*log_policy)
            Actor_loss=-(TD_error.squeeze().detach()*log_policy)
            
            Actor_network.zero_grad()
            Actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(Actor_network.parameters(), 
                                            actor_nework_max_grad_norm)
            with torch.no_grad():
                for param in Actor_network.parameters():
                    param -= actor_learning_rate * param.grad

            # print(Actor_loss)            
            # Actor_optimizer = torch.optim.Adam(Actor_network.parameters(), lr=actor_learning_rate)
            # # Zero gradients, perform a backward pass, and update the weights.
            # Actor_optimizer.zero_grad()
            # # for para in Actor_network.parameters():
            # #     print(para)
            # Actor_loss.backward()
            # # for para in Actor_network.parameters():
            # #     print(para)
            # # print(Actor_optimizer.state_dict())
            # torch.nn.utils.clip_grad_norm_(Actor_network.parameters(), 
            #                                 actor_nework_max_grad_norm)
            # Actor_optimizer.step()    
            # # for para in Actor_network.parameters():
            # #     print(para)

            if done or t>max_timestep:
                # print(accumulate_reward)
                episode_reward.append(accumulate_reward)
                episode_length.append(t)
                break
            
            # I=discount_factor*I
            current_state=next_state

        

    return episode_reward,episode_length

# episode_reward=np.array(episode_reward)
# plt.plot(episode_reward)
# plt.show()


