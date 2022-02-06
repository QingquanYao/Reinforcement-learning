import numpy as np
import torch
import random
class EpsilonGreedy():
    def __init__(self,env,start_epsilon,end_epsilon,decay_steps,type_of_decay):
        self.env=env
        self.start_epsilon=start_epsilon
        self.end_epsilon=end_epsilon
        self.decay_steps=decay_steps
        self.type_of_decay=type_of_decay # should be a string


    def return_action(self,t,action,random_action):
        '''
        return an action by epsilon greedy policy
        the type of decay of epsilon can be 'linear' or 'exp'
        '''
        if self.type_of_decay=='linear':
            epsilon=self.linear_epsilon_decay(t)
        elif self.type_of_decay=='exp':
            epsilon=self.exponential_epsilon_decay(t)
        elif self.type_of_decay=='constant':
            epsilon=self.start_epsilon

        if np.random.random()< epsilon:
            
            action=random_action
        else:
            action =action # item returns a int that is the value of tensor
        return action

    def linear_epsilon_decay(self,t):
        epsilons=np.linspace(self.start_epsilon,self.end_epsilon,self.decay_steps)
        epsilon = self.end_epsilon if t >= self.decay_steps else epsilons[t]
        return epsilon

    def exponential_epsilon_decay(self,t):
        epsilons = 0.01 / np.logspace(-2, 0, self.decay_steps, endpoint=False) - 0.01
        epsilons = epsilons * (self.start_epsilon - self.end_epsilon) + self.end_epsilon
        epsilon = self.end_epsilon if t >= self.decay_steps else epsilons[t]
        return epsilon