#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:03:00 2023


@author: lay0005
"""


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import rtamt
import math
from GPyOptPars import size

# Adjust from string to rtamt-format spec (trace)
def specification(spec_str):
    spec = rtamt.StlDenseTimeSpecification()
    spec.name = 'STL dense-time online monitor'
    spec.declare_var('x', 'float')
    spec.declare_var('y', 'float')
    spec.spec = spec_str
    spec.parse()
    return spec

# Adjust from string to rtamt-format spec (state)
def specification_cs(spec_str):
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.name = 'STL_obstacle'
    spec.declare_var('x', 'float')
    spec.declare_var('y', 'float')
    spec.spec = spec_str
    spec.parse()
    return spec

# Robustness of a trace
def robustness(dataset, spec):
    rob = spec.evaluate(['x',dataset[0]],['y',dataset[1]])
    return [i[1] for i in rob]

# Robustness of a 'state'
def robustness_cs(spec,current_state):  # Online Monitor
    rob = spec.update(0, [('x', current_state[0]) , ('y', current_state[1])])
    return rob


grid_size = 6
  
def state_transition(current_state, action):
    if action == (0,-1):#down
        suggested_state = (current_state[0] + action[0], max(0,current_state[1] + action[1]))
    elif action == (0,1):#up
        suggested_state = (current_state[0] + action[0], min(grid_size-1,current_state[1] + action[1]))
    elif action ==(-1,0):#left
        suggested_state = (max(0,current_state[0] + action[0]), current_state[1] + action[1])
    elif action == (1,0):#right
        suggested_state = (min(grid_size-1,current_state[0] + action[0]), current_state[1] + action[1])

    # up, down,right, left
    states = [(current_state[0] + 0, min(grid_size-1,current_state[1] + 1)), (current_state[0] + 0, max(0,current_state[1]  - 1)), ((min(grid_size-1,current_state[0] + 1), current_state[1] + 0)) ,(max(0,current_state[0] -1), current_state[1] + 0)]

    num_states = len(states)
    transition_probabilities = [0.9 if state == suggested_state else (0.1/(num_states-1)) for state in states]
    next_state = random.choices(states,weights=transition_probabilities)
    return next_state[0]


def Q_learning(spec, operation, num_rollout):
    rewards = []
    eps = []
    # spec= 'F(x>=4 & y>=4) & G(not((x>={} & x<{} & y>={} & y<{}) or (x>{} & x<={} & y>{} & y<={})))'
    if 'F' in spec and 'G' in spec:
        spec_no_tmp = spec.replace ('F','').replace('G','')
    elif 'F' in  spec:
        spec_no_tmp = spec.replace ('F','')
    elif 'G' in spec:
        spec_no_tmp = spec.replace ('G','')
    else:
        print('Check STL for existence of temporal operators! (or alter code)')
    print(spec_no_tmp)    
    print('Running RL...')
    # goal = specification('F[0,10](x>=4 & y>=4)')
    # Define the actions
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down,right, left  
    # Initialize Q-table
    Q = np.zeros((grid_size, grid_size, len(actions)))
   
    # Define the training parameters
    alpha = 0.1
    gamma = 0.9
    max_steps = 20
    # num_episodes = 1200
    # decay_rate = -0.002
    # num_episodes = 100
    # decay_rate = -0.015
    num_episodes = 1000
    decay_rate = -0.003
    # Train the agent
    for episode in range(num_episodes):
        cum_r = 0
        eps.append(episode)
        print(episode)
        # if episode%10 == 0:
        #     print(episode)
       
        # Initialize the state
        current_state = (0,0)
        epsilon = np.exp(decay_rate*episode)
        states_actions = []
        # Terminate the episode if the agent reaches the goal or the maximum number of steps is reached
        for step in range(max_steps):
            # Choose an action using epsilon-greedy exploration
            if random.uniform(0, 1) < epsilon: #explore
                action = random.choice(actions)
            else:
                action = actions[np.argmax(Q[current_state[0], current_state[1]])] #exploit
                                 
            # Determine the next state following the transition probablities        
            next_state = state_transition(current_state,action)

            reward = robustness_cs(specification_cs(spec_no_tmp), next_state)
            cum_r  += reward
            # Update the Q-value  
            Q[current_state[0], current_state[1], actions.index(action)] = Q[current_state[0], current_state[1], actions.index(action)] + alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[current_state[0], current_state[1], actions.index(action)])
            # Store the current state and action
            states_actions.append((current_state, action))
            # Update the state
            current_state = next_state
            
        ################### STL Rewrad ##################
        x_tr = []
        y_tr = []
        for i in range(len(states_actions)):
            x_tr.append([i,states_actions[i][0][0]])
            y_tr.append([i,states_actions[i][0][1]])          
        dataset = (x_tr,y_tr)
        
        # # rob_spec = robustness(dataset, spec)[0]
        # rob_spec = robustness(dataset, spec)[0]
        # # rob_obs = robustness(dataset, spec_obs)[0]
        
    
        if operation == '>' or operation == '<' or operation == 'not':
            rob_spec = min(robustness(dataset, specification(spec)))
        elif operation == '&':
            first_part = spec[:spec.find('&', spec.find('&') + 1)]
            second_part = spec[spec.find('&', spec.find('&') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob_spec = min(robustness(dataset, specification(spec)))
            else:
                rob1 = robustness(dataset, specification(first_part))
                rob2 = robustness(dataset, specification(second_part))
                rob_spec = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
        elif operation == '|' :
            first_part = spec[:spec.find('|', spec.find('|') + 1)]
            second_part = spec[spec.find('|', spec.find('|') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob_spec = min(robustness(dataset, specification(spec)))
            else:
                rob1 = robustness(dataset, specification(first_part))
                rob2 = robustness(dataset, specification(second_part))
                rob_spec = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
        elif operation == 'G' or operation == 'F' or operation == 'U':            
            rob_spec =robustness(dataset, specification(spec))[0]

        
        if rob_spec >= 0:
            reward = 100
        else:
            reward = 0
           
        print('reward sparse:',reward)
        rewards.append(cum_r+reward)
        # print(rob_goal,reward)
           
        # Update the Q-values at the end of the episode
        for i, (state, action) in enumerate(states_actions):
            # print(state, action, states_actions[i],i,states_actions[i+1 if i<len(states_actions)-1 else i][0][0])
            
            Q[state[0], state[1], actions.index(action)] = (1-alpha) * Q[state[0], state[1], actions.index(action)] + 0.1 * reward
            
            # Q[state[0], state[1], actions.index(action)] += alpha * (reward + gamma * np.max(Q[states_actions[i+1 if i<len(states_actions)-1 else i][0][0],
            #                        states_actions[i+1 if i<len(states_actions)-1 else i][0][1], :]) - Q[state[0], state[1], actions.index(action)])

    # Plotting cumulative reward per episode
    # plt.plot(eps, rewards)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.grid()
    # plt.show()
        
    # Generating 200 rollout traces
    rollout_traces = []
    actionnn = []
    for i in range(num_rollout):
        current_state = (0, 0)
        rollout = [current_state]
        act = []
        st = 0
        #for k in range(30):
        while st < 20:#rob_goal < 0 and st <20:
            # print(current_state)
            action = actions[np.argmax(Q[current_state[0], current_state[1]])]
            act.append(action)
            next_state = state_transition(current_state,action)
            current_state = next_state
            rollout.append(current_state)
            st+=1
        actionnn.append(act)
        rollout_traces.append(rollout)
    # print(rollout_traces)
    # print(act)      
    
    # Organize rollout traces into a dataset
    x_trace = []
    y_trace = []
    for i in range(len(rollout_traces)):
        x_vals = []
        y_vals = []
        for j in rollout_traces[i]:
            x_vals.append(j[0])
            y_vals.append(j[1])
        x_trace.append(x_vals)
        y_trace.append(y_vals)
    x_trace = pd.DataFrame(x_trace).T
    y_trace = pd.DataFrame(y_trace).T
    return x_trace,y_trace, rewards





#plt.plot(x_trace[i],y_trace[i],'*-')
#plt.grid()    
#plt.show()

# Display the grid environment with the rollout traces in black, goal position in green, and obstacle in red
def plot(x_trace,y_trace,num_rollout, grid_size):
    x_plt = []
    y_plt = []
    # fig = plt.figure()
    for i in range(len(x_trace.columns)):
        x_plt.append([j+0.5 for j in x_trace[i]])
        y_plt.append([j+0.5 for j in y_trace[i]])
   
    for i in range(num_rollout):
        fig = plt.figure()
        goal = matplotlib.patches.Rectangle((grid_size-2,grid_size-2), 2, 2, color='green')
        obstacle1 = matplotlib.patches.Rectangle((1,2), 3, 1, color='black')
        obstacle2 = matplotlib.patches.Rectangle((3,3), 1, 1, color='black')
        ax = fig.add_subplot(111)
        ax.add_patch(goal)
        ax.add_patch(obstacle1)
        ax.add_patch(obstacle2)
    #    plt.plot(x_plt[i],y_plt[i],'*-')
        plt.plot(x_plt[i],y_plt[i],'c*-')
        plt.grid()
        plt.show()
#plt.plot(x_trace[i],y_trace[i],'*-')
#plt.grid()    
#plt.show()
