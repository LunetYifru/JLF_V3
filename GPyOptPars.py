from cmath import inf
from math import exp
from random import seed
import GPyOpt
import numpy as np
from scipy.stats import norm
import pandas as pd
import math
import rtamt

def robustness(dataset, spec):
    rob = spec.evaluate(['x',dataset[0]],['y',dataset[1]])
    return [i[1] for i in rob]

def specification(spec_str):
    spec = rtamt.StlDenseTimeSpecification()
    spec.name = 'STL discrete-time online Python monitor'
    spec.declare_var('x', 'float')
    spec.declare_var('y', 'float')
    spec.spec = spec_str
    spec.parse()
    return spec


def signal(x,y,i):
    A = []
    B = []
    for j in range(len(x[i])):       
        A.append([j,x[i][j]])
        B.append([j,y[i][j]])
    return (A,B)

def size(formula):
    ops = '<>FG&|Un'
    size = 0
    for i in formula:
        if i in ops:
            size += 1
    return size
 
  
def GP_opt(spec,operation, x_reg,y_reg,x_anom,y_anom,rng):

    def pars(x):
        t1 = 0
        t2 = inf
        
        # choose variables in the same way as the search "space" below was designed
        c1 = int(x[:, 0])
        c2 = int(x[:, 1])
        c3 = int(x[:, 2])
        c4 = int(x[:, 3])
        c5 = int(x[:, 4])
        c6 = int(x[:, 5])
        c7 = int(x[:, 6])
        c8 = int(x[:, 7])

            
        # Decide formatting based on template    
        phi = spec.format(c1,c2,c3,c4,c5,c6,c7,c8)
       
        rob_reg = np.empty(0)
        rob_anom = np.empty(0)

        if t1 < t2:
            for i in range(len(x_reg.columns)):
                 if operation == '>' or operation == '<' or operation == 'not':
                     rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                 elif operation == '&':
                     first_part = phi[:phi.find('&', phi.find('&') + 1)]
                     second_part = phi[phi.find('&', phi.find('&') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_reg,y_reg,i), specification(first_part))
                         rob2 = robustness(signal(x_reg,y_reg,i), specification(second_part))
                         rob_r = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))

                 elif operation == '|' :
                     first_part = phi[:phi.find('|', phi.find('|') + 1)]
                     second_part = phi[phi.find('|', phi.find('|') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_reg,y_reg,i), specification(first_part))
                         rob2 = robustness(signal(x_reg,y_reg,i), specification(second_part))
                         rob_r = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))
                 elif operation == 'G' or operation == 'F' or operation == 'U':            
                     rob_r =robustness(signal(x_reg,y_reg,i), specification(phi))[0]
                 rob_reg = np.append(rob_reg, rob_r)
                 
            for i in range(len(x_anom.columns)):
                 if operation == '>' or operation == '<' or operation == 'not':
                     rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                 elif operation == '&':
                     first_part = phi[:phi.find('&', phi.find('&') + 1)]
                     second_part = phi[phi.find('&', phi.find('&') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_anom,y_anom,i), specification(first_part))
                         rob2 = robustness(signal(x_anom,y_anom,i), specification(second_part))
                         rob_a = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))

                 elif operation == '|' :
                     first_part = phi[:phi.find('|', phi.find('|') + 1)]
                     second_part = phi[phi.find('|', phi.find('|') + 1)+1:]
                     if size(first_part) == 1 or size(second_part) == 1:
                         rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
                     else:
                         rob1 = robustness(signal(x_anom,y_anom,i), specification(first_part))
                         rob2 = robustness(signal(x_anom,y_anom,i), specification(second_part))
                         rob_a = min(math.copysign(1,rob1[0]), math.copysign(1,rob2[0]))

                 elif operation == 'G' or operation == 'F' or operation == 'U':            
                     rob_a =robustness(signal(x_anom,y_anom,i), specification(phi))[0]
                 rob_anom = np.append(rob_anom, rob_a)
                        
            pos = 0
            neg = 0
            for i in rob_reg:
                if i>= 0: pos +=1 
            for i in rob_anom:
                 if i< 0:neg +=1
            rob_reg_av = np.average(rob_reg)
            rob_anom_av = np.average(rob_anom)    
#            A = abs(rob_reg_av-rob_anom_av)
#            print('--------',pos,neg)
            A = (pos/len(x_reg.columns)+neg/len(x_anom.columns))*100 + 10*abs(rob_reg_av-rob_anom_av)
            
#            print('t1<t2')
            if pos + neg > (len(x_reg.columns) + len(x_anom.columns)):
                A += 100           
        elif t2<=t1:
            A = -100
        print('A is:',A,'with phi :',phi)
        return -A
    #Design space based on template of the STL
    domain = tuple([i for i in range(int(rng[0]),int(rng[1])+1)]) # search space for parameters (happens to be the same for all parameters in the grid world)
    space = [{'name': 'c1', 'type': 'discrete', 'domain': domain},
            {'name': 'c2', 'type': 'discrete', 'domain': domain},
            {'name': 'c3', 'type': 'discrete','domain': domain},
            {'name': 'c4', 'type': 'discrete', 'domain': domain},
            {'name': 'c5', 'type': 'discrete','domain': domain},
            {'name': 'c6', 'type': 'discrete', 'domain': domain},
            {'name': 'c7', 'type': 'discrete','domain': domain},
            {'name': 'c8', 'type': 'discrete', 'domain': domain}]

    feasible_region = GPyOpt.Design_space(space=space)
    bounds = feasible_region.get_bounds()
    print(bounds)

    initial_design = GPyOpt.experiment_design.initial_design(
        'random', feasible_region, 30)

    # print(initial_design)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(pars)

    # --- CHOOSE the model type
    model = GPyOpt.models.GPModel(
        exact_feval=True, optimize_restarts=10, verbose=False)

    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(
        feasible_region)

    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(
        model, feasible_region, optimizer=aquisition_optimizer)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # BO object
    bo = GPyOpt.methods.ModularBayesianOptimization(
        model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time = None
    max_iter = 30
    tolerance = 1e-8     # distance between two consecutive observations

    # Run the optimization
    bo.run_optimization(max_iter=max_iter, max_time=max_time,
                        eps=tolerance, verbosity=False)

    # bo.plot_acquisition()
    # bo.plot_convergence()
    
    # Final spec with parameters
    phi = spec.format(int(bo.x_opt[0]),int(bo.x_opt[1]),int(bo.x_opt[2]),int(bo.x_opt[3]),int(bo.x_opt[4]),int(bo.x_opt[5]),int(bo.x_opt[6]),int(bo.x_opt[7])) #c1,c2,c3,c4,c5,c6,c7,c8

#     # Best found value
#     if operation == '<' or operation == '>':
#         phi = spec.format(bo.x_opt[0]) #c
#     elif operation =='G' or operation == 'F':
#         phi = spec.format(int(bo.x_opt[0]),int(bo.x_opt[1]),bo.x_opt[2],bo.x_opt[3]) #t1,t2,c
#     elif operation == 'U':
#         phi = spec.format(int(bo.x_opt[0]),int(bo.x_opt[1]),bo.x_opt[2],bo.x_opt[3]) #t1,t2,c1,c2

    return phi
#
#phi = GP_opt('G[{1},{2}](y > {0:.2f})','G', input1=x_reg, input2=y_reg,input3=x_anom, input4=y_anom, rng_mod=rng_mod)
#print(phi)
