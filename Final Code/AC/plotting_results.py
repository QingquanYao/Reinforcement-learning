import matplotlib.pyplot as plt
import numpy as np
import os


def moving_average(x, n):
    '''matrix version: input a matrix, return moving agerverage along each row 
        this version can also solve sequence problem(1-d array)
    '''
    x=np.array(x) 
    y1=np.zeros(x.shape)
    if x.shape[1]<=n:
        for i in range(x.shape[1]): 
            y1[:,i]=np.average(x[:,:i+1],axis=1)
        return y1
    else:
        for i in range(x.shape[1]): 
            if i<n-1:
                y1[:,i]=np.average(x[:,:i+1],axis=1)
            else: 
                for j in range(x.shape[0]):
                    y1[j,n-1:]=np.convolve(x[j,:], np.ones(n), 'valid') / n
        return y1

def convert(result,n):
    #convert pure reuslt to moving average resutl over n episodes or steps
    # and return max,min,mean 
    result=np.array(result)
    r_max=result.max(axis=0)
    r_min=result.min(axis=0)
    r_mean=result.mean(axis=0)
    return r_max,r_min,r_mean

def calculate_ratio_of_optimal(episode_reward_result,optimal_value,n):
    # this is used to calculate the precentage of optimal policy occurring in previous n episodes 
    episode_reward_result=np.array(episode_reward_result)
    ratio_of_optimal=np.zeros(shape=(episode_reward_result.shape))
    for j in range(episode_reward_result.shape[1]):
        if j-n>=0:
            ratio_of_optimal[:,j]=np.count_nonzero(episode_reward_result[:,j-n:j]==optimal_value,axis=1)/n
        elif j>0:
            ratio_of_optimal[:,j]=np.count_nonzero(episode_reward_result[:,:j]==optimal_value,axis=1)/j
        else:
            ratio_of_optimal[:,j]=np.count_nonzero(episode_reward_result[:,:j]==optimal_value,axis=1)
    return ratio_of_optimal

def plot_subplot_episode_reward(result,n,axs,i,color,label):
    #color and label are strings
    ma_result=moving_average(result,n)
    mar_max,mar_min,mar_mean=convert(ma_result,n)
    axs[i].plot(mar_max, color, linewidth=1)
    axs[i].plot(mar_min, color, linewidth=1)
    axs[i].plot(mar_mean, color, label=label, linewidth=2)
    x=np.arange(np.size(mar_mean)) 
    axs[i].fill_between(x, mar_min, mar_max, facecolor=color, alpha=0.3)
    axs[i].set_title('Moving Average Return over '+ str(n)+' Episodes')
    axs[i].set_xlabel('Episodes')
    axs[i].set_ylabel('Moving Average\n Return')    
    axs[i].legend()

def plot_subplot_episode_length(result,n,axs,i,color,label):
    #color and label are strings
    ma_result=moving_average(result,n)
    mar_max,mar_min,mar_mean=convert(ma_result,n)
    axs[i].plot(mar_max, color, linewidth=1)
    axs[i].plot(mar_min, color, linewidth=1)
    axs[i].plot(mar_mean, color, label=label, linewidth=2)
    x=np.arange(np.size(mar_mean)) 
    axs[i].fill_between(x, mar_min, mar_max, facecolor=color, alpha=0.3)
    axs[i].set_title('Moving Average Length of Episode over '+ str(n)+' Episodes')
    axs[i].set_xlabel('Episodes')
    axs[i].set_ylabel('Moving Average \nLength of Episode')
    axs[i].legend()

def plot_subplot_step_reward(result,n,axs,i,color,label):
    #color and label are strings
    ma_result=moving_average(result,n)
    mar_max,mar_min,mar_mean=convert(ma_result,n)
    axs[i].plot(mar_max, color, linewidth=1)
    axs[i].plot(mar_min, color, linewidth=1)
    axs[i].plot(mar_mean, color, label=label, linewidth=2)
    x=np.arange(np.size(mar_mean)) 
    axs[i].fill_between(x, mar_min, mar_max, facecolor=color, alpha=0.3)
    axs[i].set_title('Moving Average Step Reward over '+ str(n)+' Steps')
    axs[i].set_xlabel('Steps')
    axs[i].set_ylabel('Moving Average \n Step Reward')
    axs[i].legend()

def plot_subplot_ratio_of_optimal(result,optimal_value,n,axs,i,color,label):
    #color and label are strings
    ratio_of_optimal=calculate_ratio_of_optimal(episode_reward_result=result,optimal_value=optimal_value,n=n)
    mar_max,mar_min,mar_mean=convert(ratio_of_optimal,n)
    axs[i].plot(mar_max, color, linewidth=1)
    axs[i].plot(mar_min, color, linewidth=1)
    axs[i].plot(mar_mean, color, label=label, linewidth=2)
    x=np.arange(np.size(mar_mean)) 
    axs[i].fill_between(x, mar_min, mar_max, facecolor=color, alpha=0.3)
    axs[i].set_title('Ratio of Optimal Policy over'+ str(n)+' Episodes')
    axs[i].set_xlabel('Episodes')
    axs[i].set_ylabel('Ratio of optimal policy ')
    axs[i].legend()