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


    def return_action(self,Q,t):
        '''
        return an action by epsilon greedy policy
        the type of decay of epsilon can be 'linear' or 'exp'
        '''
        if self.type_of_decay=='linear':
            epsilon=self.linear_epsilon_decay(t)
        elif self.type_of_decay=='sine':
            epsilon=self.sine_epsilon_fluctuation(t)
        elif self.type_of_decay=='exp':
            epsilon=self.exponential_epsilon_decay(t)
        elif self.type_of_decay=='constant':
            epsilon=self.start_epsilon

        with torch.no_grad():
            if np.random.random()< epsilon:
                action=random.randrange(Q.shape.numel())
            else:
                action =torch.argmax(Q).item() # item returns a int that is the value of tensor
            
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

    def sine_epsilon_fluctuation(self,t):
        epsilons=np.linspace(0,100,self.decay_steps)
        epsilons = (np.sin(epsilons)+1)/2*(self.start_epsilon-self.end_epsilon)+self.end_epsilon
        epsilon = self.end_epsilon if t >= self.decay_steps else epsilons[t]
        return epsilon