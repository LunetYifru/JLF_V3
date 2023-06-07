#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:40:42 2023

@author: lay0005
"""
import rtamt
import math
import pandas as pd
from GPyOptPars import robustness,specification,signal,size
#from InitialDataset import initial_trace
#x_trace, y_trace = initial_trace()

def Human_labeling(spec,operation,x_trace,y_trace):
    x_reg = []
    y_reg = []
    x_anom = []
    y_anom= []
    
    for i in range(len(x_trace.columns)):
        if operation == '>' or operation == '<' or operation == 'not':
            rob = min(robustness(signal(x_trace,y_trace,i), specification(spec)))
        elif operation == '&':

            first_part = spec[:spec.find('&', spec.find('&') + 1)]
            second_part = spec[spec.find('&', spec.find('&') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(x_trace,y_trace,i), specification(spec)))
            else:
                rob1 = robustness(signal(x_trace,y_trace,i), specification(first_part))
                rob2 = robustness(signal(x_trace,y_trace,i), specification(second_part))
                rob = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
        elif operation == '|' :
            first_part = spec[:spec.find('&', spec.find('&') + 1)]
            second_part = spec[spec.find('&', spec.find('&') + 1)+1:]
            if size(first_part) == 1 or size(second_part) == 1:
                rob = min(robustness(signal(x_trace,y_trace,i), specification(spec)))
            else:
                rob = robustness(signal(x_trace,y_trace,i), specification(spec))[0]
        elif operation == 'G' or operation == 'F' or operation == 'U':            
            rob =robustness(signal(x_trace,y_trace,i), specification(spec))[0]


        if rob >=0:
            x_reg.append([j for j in x_trace[i]])
            y_reg.append([j for j in y_trace[i]])
        if rob <0:
            x_anom.append([j for j in x_trace[i]])
            y_anom.append([j for  j in y_trace[i]])

             
    x_reg = pd.DataFrame(x_reg).T
    y_reg = pd.DataFrame(y_reg).T
    x_anom = pd.DataFrame(x_anom).T
    y_anom = pd.DataFrame(y_anom).T
    return x_reg,y_reg,x_anom,y_anom
         
#x_reg,y_reg,x_anom,y_anom = Human_labling('F(x>=4 & y>=4)','F',x_trace,y_trace)