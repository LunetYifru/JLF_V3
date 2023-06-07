#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:11:54 2023

@author: lay0005
"""
import random
import pandas as pd
from QLearningWSTL import state_transition
# Define the actions
goal_loc = [(4,4),(4,5),(5,4),(5,5)]
def initial_trace():
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down,right, left
    x_trace = []
    y_trace = []
    while len(x_trace) < 200 :
        current_state = (0,0)
        states= [current_state]
        while current_state not in goal_loc:
            action = random.choice(actions)#explore                              
            # Determine the next state following the transition probablities        
            next_state = state_transition(current_state,action)        
            # Store the current state
            # Update the state
            current_state = next_state
            states.append(current_state)  
            
        if len(states) <=20:
            x_vals = []
            y_vals = []
            for j in states:
                x_vals.append(j[0])
                y_vals.append(j[1])
            x_trace.append(x_vals)
            y_trace.append(y_vals)
        else:
            continue
        # Display the grid environment with the rollout traces in black, goal position in green, and obstacle in red
    x_trace = pd.DataFrame(x_trace).T
    y_trace = pd.DataFrame(y_trace).T
    x_trace.fillna(method='ffill', inplace=True)
    y_trace.fillna(method='ffill', inplace=True)
    return x_trace, y_trace
#x_trace, y_trace = initial_trace()