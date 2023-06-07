#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:26:20 2023

@author: lay0005
"""
import numpy as np
from QLearningWSTL import Q_learning
import matplotlib.pyplot as plt
from Labeling import Human_labeling

def percentage(reg, anom):
    return (reg/(reg+anom))*100

def plot_curve_with_confidence(true_spec,op, ours,op_ours, BL1,op1,BL2,op2, num_rollout, runs):
    dataset = []
    for i in range(runs):
        x_trace,y_trace, rewards = Q_learning(ours,op_ours,num_rollout)
        dataset.append(rewards)
           
    # Calculate average and confidence interval
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    n = len(dataset)
    confidence = 1.96 * std / np.sqrt(n)  # 95% confidence interval (assuming normal distribution)
    # Create x-axis values
    x = np.arange(len(mean))
    # Plot curve with shaded confidence interval
    plt.plot(x, mean, label='Average')
    plt.fill_between(x, mean - confidence, mean + confidence, alpha=0.3)
    # Add labels and title

    #baseline2
    dataset = []
    for i in range(runs):
        x_trace,y_trace, rewards = Q_learning(BL1,op1,num_rollout)
        dataset.append(rewards)
           
    # Calculate average and confidence interval
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    n = len(dataset)
    confidence = 1.96 * std / np.sqrt(n)  # 95% confidence interval (assuming normal distribution)
    # Create x-axis values
    x = np.arange(len(mean))
    # Plot curve with shaded confidence interval
    plt.plot(x, mean, label='Average')
    plt.fill_between(x, mean - confidence, mean + confidence, alpha=0.3)
    #compute number of safe traces
    x_r,y_r,x_a,y_a = Human_labeling(true_spec,op,x_trace,y_trace)  # Label rollout traces from RL
    percentage_safe = percentage(len(x_r.columns),len(x_a.columns))
    print('Percentage of safe rollout traces(Baseline 1): ', percentage_safe)  
    # Add labels and title
    
    #Baseline 3
    dataset = []
    for i in range(runs):
        x_trace,y_trace, rewards = Q_learning(BL2,op2,num_rollout)
        dataset.append(rewards)
           
    # Calculate average and confidence interval
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    n = len(dataset)
    confidence = 1.96 * std / np.sqrt(n)  # 95% confidence interval (assuming normal distribution)
    # Create x-axis values
    x = np.arange(len(mean))
    # Plot curve with shaded confidence interval
    plt.plot(x, mean, label='Average')
    plt.fill_between(x, mean - confidence, mean + confidence, alpha=0.3)
    #compute number of safe trace
    x_r,y_r,x_a,y_a = Human_labeling(true_spec,op,x_trace,y_trace)  # Label rollout traces from RL
    percentage_safe = percentage(len(x_r.columns),len(x_a.columns))
    print('Percentage of safe rollout traces(Baseline 2): ', percentage_safe)  

    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Average Cumulative Reward per Episode')
    plt.grid()
    plt.legend(['Ours','Baseline 1','Baseline 2'])
    plt.show()
        
    
