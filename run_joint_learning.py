#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:08:47 2023

@author: lay0005
"""

# from STL_inference import GA,initialize
from GPyOptPars import GP_opt
from InitialDataset import initial_trace
from Labeling import Human_labeling
from QLearningWSTL import Q_learning, plot_curve_with_confidence
import pandas as pd
from plot import plot_curve_with_confidence, percentage

print('Initializing data...')
x_trace,y_trace = initial_trace() #Random trace generation (initial)

true_spec = 'F(x>=4 & y>=4) & G(not((x>=1 & x<4 & y>=2 & y<3) or (x>=4 & x<5 & y>=3 & y<4)))' #spec helping as the 'human'
op = '&'
num_rollout = 200
x_reg,y_reg,x_anom,y_anom = Human_labeling(true_spec,op,x_trace,y_trace) # labeling traces
num_safe = len(x_reg.columns) #number of safe rollout traces
num_unsafe = len(x_anom.columns) #number of safe rollout traces
percentage_safe = percentage(num_safe,num_unsafe)

print('Number of safe traces: ',num_safe,'( of ',num_safe+num_unsafe,')')
rng = [min(min(x_reg.min()),min(y_reg.min()),min(x_anom.min()),min(y_anom.min())), max(max(x_reg.max()),max(y_reg.max()),max(x_anom.max()),max(y_anom.max())),len(x_reg)] # Upper and lower bounds for GP

print('Begining bi-level optimization...')
it = 1 #iteration number
STLs = ['Random']
safe_p = [percentage_safe]
size_xreg = [len(x_reg.columns)]

while percentage_safe < 98.0:
    # pop_df = initialize(x_reg, y_reg, x_anom, y_anom, rng)
    # infered_STL = GA(pop_df,x_reg, y_reg, x_anom, y_anom, rng) #infer full STL from traces (template+parameters)
    rng = [min(min(x_reg.min()),min(y_reg.min()),min(x_anom.min()),min(y_anom.min())), max(max(x_reg.max()),max(y_reg.max()),max(x_anom.max()),max(y_anom.max())),len(x_reg)] # Upper and lower bounds for GP
    spec_template ='F(x>=4 & y>=4) & G(not((x>={} & x<{} & y>={} or y<{}) & (x>={} & x<{} & y>={} & y<{})))'
    infered_STL = GP_opt(spec_template,'&', x_reg,y_reg,x_anom,y_anom,rng) #Infer parameters given STL template
    print('----Infered STL is----', infered_STL)
    STLs.append(infered_STL)
    x_trace,y_trace,rewards = Q_learning(infered_STL,'&',num_rollout) # Q learning and generate rollout traces based on infered STL
    # x_reg,y_reg,x_anom,y_anom = Human_labeling(spec,op,x_trace,y_trace) # Label rollout traces from RL
    x_r,y_r,x_a,y_a = Human_labeling(true_spec,op,x_trace,y_trace)  # Label rollout traces from RL
    percentage_safe = percentage(len(x_r.columns),len(x_a.columns))
    safe_p.append(percentage_safe)
    # Appending new data
    x_reg = pd.concat([x_reg, x_r], axis=1, ignore_index=True)
    y_reg = pd.concat([y_reg, y_r], axis=1, ignore_index=True)
    x_anom = pd.concat([x_anom, x_a], axis=1, ignore_index=True)
    y_anom = pd.concat([y_anom, y_a], axis=1, ignore_index=True)
    
    size_xreg.append(len(x_reg.columns))
    num_safe = len(x_r.columns) #number of safe rollout traces
    num_unsafe = len(x_a.columns)
    print('Iteration number: ',it,'---',percentage_safe,'% safe traces')
    it += 1

print('Final STL is: ', infered_STL)

from plot import plot_curve_with_confidence, percentage
our_STL = infered_STL
# Baseline #1 : Known STL goals, but not constraints
STL_unknown_constraints = 'F(x>=4 and y>=4)'
#Baseline #2 : Every part of the STL known (ideal situation)
STL_known_constraints = 'F(x>=4 & y>=4) & G(not((x>=1 & x<4 & y>=2 & y<3) or (x>=4 & x<5 & y>=3 & y<4)))'

plot_curve_with_confidence(true_spec,'&', our_STL,'&', STL_unknown_constraints,'F',STL_known_constraints,'&', 200, 10)

#Baseline #3: Classical rewards with known goal and obstacle locations

#**To be written
