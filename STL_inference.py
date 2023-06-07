import random
import rtamt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from GPyOptPars import GP_opt,robustness,specification,signal
    
#
####################    Import Data    ##############################
#x_reg = pd.read_excel(r'x1_reg.xlsx', header=None)
#y_reg = pd.read_excel(r'x2_reg.xlsx', header=None)
#x_anom = pd.read_excel(r'x1_anom.xlsx', header=None)
#y_anom = pd.read_excel(r'x2_anom.xlsx', header=None)

 ###########  Random initialization of primary population and GP_opt to update parameters  ##########
def initialize(x_reg,y_reg,x_anom,y_anom,rng_mod):

    print('-------------Initializing population--------------')
    population = {}
    pop_df = pd.DataFrame()
    op = ['>','<','>','<','G','G','G','G','F','F','F','F','U','U']
    
    population['phi1'] = GP_opt('x > {0:.2f}',op[0], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi2'] = GP_opt('x < {0:.2f}',op[1], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi3'] = GP_opt('y > {0:.2f}',op[2], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi4'] = GP_opt('y < {0:.2f}',op[3], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi5'] = GP_opt('G[{0},{1}](x > {2:.2f})',op[4], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi6'] = GP_opt('G[{0},{1}](x < {2:.2f})',op[5], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi7'] = GP_opt('G[{0},{1}](y > {2:.2f})',op[6], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi8'] = GP_opt('G[{0},{1}](y < {2:.2f})',op[7], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi9'] = GP_opt('F[{0},{1}](x > {2:.2f})',op[8], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi10'] = GP_opt('F[{0},{1}](x < {2:.2f})',op[9], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi11'] = GP_opt('F[{0},{1}](y > {2:.2f})',op[10], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi12'] = GP_opt('F[{0},{1}](y < {2:.2f})',op[11], x_reg,y_reg,x_anom,y_anom,rng_mod)   
    population['phi13'] = GP_opt('(x<{2:.2f})U[{0},{1}](y>{3:.2f})',op[12], x_reg,y_reg,x_anom,y_anom,rng_mod)
    population['phi14'] = GP_opt('(y>{2:.2f})U[{0},{1}](x>{3:.2f})',op[13], x_reg,y_reg,x_anom,y_anom,rng_mod)

        

    pop_df['Name'] = population.keys()
    pop_df['Formula'] = population.values()
    pop_df['Operation'] = op
    
 ##########  initialization of formulae with binary operators ##########
    pop_ext = {}
    N = []  
    for i in range(14, 50):
        random.seed()
        rand = random.randint(1, 2) 
        
        num = random.sample(range(0, 14), 2) 
#        while ('x' in pop_df.iloc[num[0]]['Formula'] and 'x' in pop_df.iloc[num[1]]['Formula']) or ('y' in pop_df.iloc[num[0]]['Formula'] and 'y' in pop_df.iloc[num[1]]['Formula']) or (num in N):
#               num = random.sample(range(0, 11), 2)        
        while num in N:
            num = random.sample(range(0, 14), 2) 
        N.append(num)
        if rand == 1:
             op = '&'
             pop_ext['phi' + str(i)] = '('+pop_df.iloc[num[0]]['Formula']+')'+ op +'('+pop_df.iloc[num[1]]['Formula']+')'
        elif rand == 2:
             op = '|'
             pop_ext['phi' + str(i)] = '('+pop_df.iloc[num[0]]['Formula']+')'+ op +'('+pop_df.iloc[num[1]]['Formula']+')'
#        elif rand == 3:
#             op = 'not'
#             pop_ext['phi' + str(i)] = op +'('+pop_df.iloc[num[0]]['Formula']+')'

#        print(pop_ext)        
        pop_df = pop_df.append({'Name': 'phi' + str(i),'Formula': pop_ext['phi'+ str(i)],'Operation': op}, ignore_index=True)   

    population = {**population, **pop_ext}    
    print('Length of Data Frame ---', len(pop_df))    
    return pop_df


# ##############################     computing parameters for formula           ##################

def compute_rob(phi,operation,x_reg,y_reg,x_anom,y_anom):    
    print(phi)
    rob_reg = np.empty(0)
    rob_anom = np.empty(0)
        
    for i in range(len(x_reg.columns)):
         if operation == '>' or operation == '<' or operation == 'not':
             rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
         elif operation == '&':
             first_part = phi[:phi.find('&')]
             second_part = phi[phi.find('&')+1:]
             if size(first_part) == 1 or size(second_part) == 1:
                 rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
             else:
                 rob_r = robustness(signal(x_reg,y_reg,i), specification(phi))[0]
         elif operation == '|' :
             first_part = phi[:phi.find('|')]
             second_part = phi[:phi.find('|')+1:]
             if size(first_part) == 1 or size(second_part) == 1:
                 rob_r = min(robustness(signal(x_reg,y_reg,i), specification(phi)))
             else:
                 rob_r = robustness(signal(x_reg,y_reg,i), specification(phi))[0]
         elif operation == 'G' or operation == 'F' or operation == 'U':            
             rob_r =robustness(signal(x_reg,y_reg,i), specification(phi))[0]
         rob_reg = np.append(rob_reg, rob_r)
         
    for i in range(len(x_anom.columns)):
         if operation == '>' or operation == '<' or operation == 'not':
             rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
         elif operation == '&':
             first_part = phi[:phi.find('&')]
             second_part = phi[phi.find('&')+1:]
             if size(first_part) == 1 or size(second_part) == 1:
                 rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
             else:
                 rob_a = robustness(signal(x_anom,y_anom,i), specification(phi))[0]
         elif operation == '|' :
             first_part = phi[:phi.find('|')]
             second_part = phi[:phi.find('|')+1:]
             if size(first_part) == 1 or size(second_part) == 1:
                 rob_a = min(robustness(signal(x_anom,y_anom,i), specification(phi)))
             else:
                 rob_a = robustness(signal(x_anom,y_anom,i), specification(phi))[0]
         elif operation == 'G' or operation == 'F' or operation == 'U':            
             rob_a =robustness(signal(x_anom,y_anom,i), specification(phi))[0]
         rob_anom = np.append(rob_anom, rob_a)
    
    pos = 0
    neg = 0
    for i in rob_reg:
        if i>= 0: pos +=1 
    for i in rob_anom:
         if i< 0:neg +=1
    
    results = [pos, neg]
    print( results)
    # rob_diff = np.absolute(np.average(rob_reg) - np.average(rob_anom))   
    return results

def size(formula):
    ops = '<>FG&|Un'
    size = 0
    for i in formula:
        if i in ops:
            size += 1
    return size

def fitness_func(results, size,x_reg,y_reg,x_anom,y_anom): 
    if size<=5:
       Fitness_score = (results[0]/len(x_reg.columns) + results[1]/len(x_anom.columns) +0.5*((results[0] +results[1])/(len(x_reg.columns)+len(x_anom.columns))))*100
    else:
        # decrease robustness by (size-3)X10 % for sizes greater than 2
        Fitness_score = ((results[0]/len(x_reg.columns) + results[1]/len(x_anom.columns))*100 ) *(1-((size-5)/10))
    return Fitness_score
 
# ###################      Compute robustness, size, and fitness       ###########################

def evaluate(pop_df,x_reg,y_reg,x_anom,y_anom):
     print('Evaluating formulae in population...')
     pop_size = []
     pop_fit = []
     pop_res = []
     for i in range(len(pop_df)):
         formula = pop_df.iloc[i]['Formula']
         operation = pop_df.iloc[i]['Operation']
         pop_res.append(compute_rob(formula, operation,x_reg,y_reg,x_anom,y_anom)) #robustness(results)
         pop_size.append(size(formula)) #size
         pop_fit.append(fitness_func(pop_res[-1], pop_size[-1],x_reg,y_reg,x_anom,y_anom)) #fitness
         
     pop_df['Size'] = pop_size
     pop_df['Fitness'] = pop_fit
     pop_df['Results'] = pop_res
     pop_df_evalulated = pop_df
    
     return pop_df_evalulated   
  
#####################      Rearrange  population with descending order of fitness  ############

def rank(pop_df_evaluated):
    pop_df_evaluated['Rank'] = pop_df_evaluated['Fitness'].rank(ascending=0)
    pop_df_evaluated = pop_df_evaluated.set_index('Rank')
    pop_df_evaluated = pop_df_evaluated.sort_index()
    pop_df_evaluated.reset_index(inplace=True)
    pop_df_evaluated.drop(columns=['Rank'])
    pop_df_ranked = pop_df_evaluated
    return pop_df_ranked


####################       New generation parents (Top 25)   ################
def top_half(pop_df_ranked):
    pop_df_new = pop_df_ranked.head(25)
    return pop_df_new

#print(XX)
# ##########################  Mutation and Crossover Functions   ###########

def mutate(index, pop_df_new,x_reg,y_reg,x_anom,y_anom,rng_mod):
     formula = pop_df_new.iloc[index]['Formula']
     op1 = pop_df_new.iloc[index]['Operation']
     print('Formula to be mutated: ',formula)       
     if op1 == '>' and size(formula) == 1:
         op = '<'
         phi=formula[:formula.find('>')]+op+'{0:.2f}'
         phi = GP_opt(phi,op, x_reg,y_reg,x_anom,y_anom,rng_mod)
         return phi,op 
     elif op1 == '<' and size(formula) == 1:
         op = '>'
         phi=formula[:formula.find('<')]+op+'{0:.2f}'
         phi = GP_opt(phi,op, x_reg,y_reg,x_anom,y_anom,rng_mod)
         return phi,op        
     elif op1 == 'F':
         op ='G'
         phi=formula.replace(re.findall(re.escape('F')+"(.*)"+re.escape(']'),formula)[0],'[{0},{1}',1).replace('F',op,1)
         x = re.findall(r"[><\s]\s*(\d+\.\d+|\d+)", phi)
         phi = phi.replace(x[0],'{2:.2f}',1)
         phi = GP_opt(phi,op, x_reg,y_reg,x_anom,y_anom,rng_mod)     # alter temporal operators
         return phi,op
     elif op1 == 'G' :
         op ='F'
         phi=formula.replace(re.findall(re.escape('G')+"(.*)"+re.escape(']'),formula)[0],'[{0},{1}',1).replace('G',op,1)
         x = re.findall(r"[><\s]\s*(\d+\.\d+|\d+)", phi)
         phi = phi.replace(x[0],'{2:.2f}',1)
         phi = GP_opt(phi,op, x_reg,y_reg,x_anom,y_anom,rng_mod)
         return phi,op
     elif op1 == '&':
         op = '|'

         phi = formula.replace('&',op,1)
         return phi,op
     elif op1 == '|':
         op = '&'

         phi = formula.replace('|',op,1)
         return phi,op
     elif op1 == 'U':
         op = 'U'
#         A = re.split(r'U\[\d+,\d+\]',formula)
#         if size(A[0])==1 and size(A[1])==1:
#            digits = re.findall(r'(\d+\.?\d*|\.\d+)', formula)
#            replacements = ['{2:.2f}','{0}','{1}','{3:.2f}']
#            for i in range(4):
#                d = digits[i]
#                formula = re.sub(fr'\b{d}(?!:)\b', replacements[i], formula)
#            phi = formula
#            
         if '>' in formula:
            phi = formula.replace('>','<',1)
         elif '<' in formula:
            phi = formula.replace('<','>',1)
#         else:
#            phi = A[1]+'U[{0},{1}]'+A[0]
#            
#         phi = GP_opt(phi,op, x_reg,y_reg,x_anom,y_anom,rng_mod)              
         return phi,op
     else:
         return formula,op1

def crossover(index1,index2,pop_df_new,x_reg,y_reg,x_anom,y_anom,rng_mod):
    
     formula1 = pop_df_new.iloc[index1]['Formula']
     formula2 = pop_df_new.iloc[index2]['Formula']
     while formula1 == formula2:
         formula1 = pop_df_new.iloc[random.sample(range(0,25), 1)[0]]['Formula']
         
     op1 = pop_df_new.iloc[index1]['Operation']
     op2 = pop_df_new.iloc[index2]['Operation']
     print('Formula for crossover : ',formula1,op1,'______',formula2,op2)  
#     print(formula1)
#     print(formula2)
#     
     # if ((op1 == '<') or (op1 == '>')) and ((op2 == '<') or (op2 == '>')):
     #     phi1 = formula1.split(op1)[0]+op1+'{2:.2f})U[{0},{1}]'+formula2.split(op2)[0]+op2+'{3:.2f}'
     if op1 =='&' :
         op = ['&','&']
         if (op2 =='|' ):
             phi1 = formula1[:formula1.find('&')]+op[0]+formula2[formula2.find('|')+1:]
             phi2 = formula1[formula1.find('&')+1:]+op[1]+formula2[:formula2.find('|')]
             
         elif ( op2 =='&'):
             phi1 = formula1[:formula1.find('&')]+op[0]+formula2[formula2.find('&')+1:]
             phi2 =  formula1[formula1.find('&')+1:]+op[1]+ formula2[:formula2.find('&')]
             
         elif ( op2 == 'G') or (op2 == 'F') or ((op2 =='<') and (size(formula2)==1)) or ((op2 =='>') and (size(formula2)==1))  or ((op2 == 'not' and size(formula2)==1)):
             phi1 = formula1[:formula1.find('&')]+op[0]+'('+formula2+')'
             phi2 = formula1[formula1.find('&')+1:]+op[1]+'('+formula2+')'
             if  formula1[:formula1.find('&')] == formula2 or  formula1[formula1.find('&')+1:] == formula2:
                 phi1 = formula1
                 phi2 = formula2
             
         elif op2 == 'U':
             A = re.split(r'U\[\d+,\d+\]',formula2)
             phi1 = formula1[:formula1.find('&')]+op[0]+A[1]
             phi2 = A[0]+op[1]+formula1[formula1.find('&')+1:]            
         return [phi1,phi2,op]
     elif op1 == '|':
         op = ['|','|']
         if ( op2 =='&'):
             phi1 = formula1[:formula1.find('|')]+op[0]+formula2[formula2.find('&')+1:]
             phi2 = formula1[formula1.find('|')+1:]+op[1]+formula2[:formula2.find('&')]
             
         elif (op2 =='|' ):
             phi1 = formula1[:formula1.find('|')]+op[0]+formula2[formula2.find('|')+1:]
             phi2 =  formula1[formula1.find('|')+1:]+op[1]+ formula2[:formula2.find('|')]
             
         elif ( op2 == 'G') or (op2 == 'F') or ((op2 =='<') and (size(formula2)==1)) or ((op2 =='>') and (size(formula2)==1))  or ((op2 == 'not' and size(formula2)==1)):
             phi1 = formula1[:formula1.find('|')]+op[0]+'('+formula2+')'
             phi2 = formula1[formula1.find('|')+1:]+op[1]+'('+formula2+')' 
             if  formula1[:formula1.find('|')] == formula2 or  formula1[formula1.find('|')+1:] == formula2:
                 phi1 = formula1
                 phi2 = formula2
         elif op2 == 'U':
             A = re.split(r'U\[\d+,\d+\]',formula2)
             phi1 = formula1[:formula1.find('|')]+op[0]+A[1]
             phi2 = A[0]+op[1]+formula1[formula1.find('|')+1:]
         return [phi1,phi2,op]
     elif op1 == 'U':
         op = ['U','U']
         A = re.split(r'U\[\d+,\d+\]',formula1)
         if op2 =='&':
             phi1 = A[0]+'U[{0},{1}]'+formula2[formula2.find('&')+1:]
             phi2 = formula2[:formula2.find('&')]+'U[{0},{1}]'+A[1]
             if 'U' in formula2[formula2.find('&')+1:]:
                 B =  re.split(r'U\[\d+,\d+\]',formula2[formula2.find('&')+1:])
                 phi1 =  A[0]+'U[{0},{1}]'+ B[1]
                 phi2 = B[0]+'U[{0},{1}]'+ A[1]
             if 'U' in formula2[:formula2.find('&')]:
                 B =  re.split(r'U\[\d+,\d+\]', formula2[:formula2.find('&')])
                 phi1 =  A[0]+'U[{0},{1}]'+ B[1]
                 phi2 = B[0]+'U[{0},{1}]'+ A[1]
             phi1 = GP_opt(phi1,op[0], x_reg,y_reg,x_anom,y_anom,rng_mod) 
             phi2 = GP_opt(phi2,op[1], x_reg,y_reg,x_anom,y_anom,rng_mod) 
         elif op2 =='|':
             phi1 = A[0]+'U[{0},{1}]'+formula2[formula2.find('|')+1:]
             phi2 = formula2[:formula2.find('|')]+'U[{0},{1}]'+A[1]
             if 'U' in formula2[formula2.find('|')+1:]:
                 phi1 = A[0]+'U[{0},{1}]'+ formula2[:formula2.find('|')]
                 phi2 = A[1]+'U[{0},{1}]'+ formula2[:formula2.find('|')]
             if 'U' in formula2[:formula2.find('|')]:
                 phi1 = A[0]+'U[{0},{1}]'+ formula2[formula2.find('|')+1:]
                 phi2 = A[1]+'U[{0},{1}]'+ formula2[formula2.find('|')+1:]
             phi1 = GP_opt(phi1,op[0], x_reg,y_reg,x_anom,y_anom,rng_mod) 
             phi2 = GP_opt(phi2,op[1], x_reg,y_reg,x_anom,y_anom,rng_mod) 
         elif ( op2 == 'G') or (op2 == 'F') or ((op2 =='<') and (size(formula2)==1)) or ((op2 =='>') and (size(formula2)==1))  or ((op2 == 'not' and size(formula2)==1)):
             phi1 =A[0]+ 'U[{},{}]' + '('+formula2+')'
             phi2 = '('+formula2+')' + 'U[{},{}]' + A[1]
             phi1 = GP_opt(phi1,op[0], x_reg,y_reg,x_anom,y_anom,rng_mod) 
             phi2 = GP_opt(phi2,op[1], x_reg,y_reg,x_anom,y_anom,rng_mod) 
         elif op2 == 'U':
             op = ['&','&']
             B = re.split(r'U\[\d+,\d+\]',formula2)
             phi1 = A[0] +op[0] + B[1]
             phi2 = A[1] +op[1] + B[0]
#             phi1 = A[0] + 'U[{},{}]' +B[1]
#             phi2 = B[0]+ 'U[{},{}]' +A[1]    
  
         return [phi1,phi2,op]
     elif ( op1 == 'G') or (op1 == 'F') or ((op1 =='<') and (size(formula2)==1)) or ((op1 =='>') and (size(formula2)==1))  or ((op1 == 'not' and size(formula2)==1)):
         op = ['&','|']
         if ( op2 == 'G') or (op2 == 'F') or ((op2 =='<') and (size(formula2)==1)) or ((op2 =='>') and (size(formula2)==1))  or ((op2 == 'not' and size(formula2)==1)):
             phi1 = '('+formula1+')' + op[0] + '('+formula2+')'
             phi2 = '('+formula1+')' + op[1] + '('+formula2+')'
         elif op2 == 'U':
             A = re.split(r'U\[\d+,\d+\]',formula2)
             phi1 = '('+formula1+')' + op[0] + A[1]
             phi2 = '('+formula1+')' + op[1] + A[0]
         elif op2 == '&':
             phi1 = '('+formula1+')' + op[0] +formula2[formula2.find('&')+1:]
             phi2 = '('+formula1+')' + op[1] + formula2[:formula2.find('&')]        
         elif op2 == '|':
             phi1 = '('+formula1+')' + op[0] + formula2[formula2.find('|')+1:]
             phi2 = '('+formula1+')' + op[1] + formula2[:formula2.find('|')]
         return [phi1,phi2,op]         
     else:
          return [formula1,formula2,[op1,op2]]
      
################################    Genetic Algorithm    #######################################

def GA(pop_df,x_reg, y_reg, x_anom, y_anom, rng):

      # pop_df = initialize(x_reg, y_reg, x_anom, y_anom, rng_mod)
     pop_df_new = top_half(rank(evaluate(pop_df,x_reg, y_reg, x_anom, y_anom)))
     pop_df_new = pop_df
     count = 1
     MCR = 100
     N = []
     while MCR>0.04:       
         print('Genetic Algorithm iteration:',count,'--------------') 
         pop_df_sechalf = pd.DataFrame(columns=['Name', 'Formula', 'Operation'])
         for i in range(26, 51):
             random.seed()
             n = random.random()
             num = random.sample(range(0,25), 2)
             while num in N:
                 num = random.sample(range(0,25), 2)
             N.append(num)
             if n > 0.5:
                 X = crossover(num[0],num[1],pop_df_new,x_reg, y_reg, x_anom, y_anom,rng)
                 while (pop_df['Formula'].eq(X[0])).any() or (pop_df['Formula'].eq(X[1])).any() :
                     print('OP were: ',X[0],(pop_df['Formula'].eq(X[0])).any(),' and ',X[1],(pop_df['Formula'].eq(X[1])).any())
                     num = random.sample(range(0,25), 2)
                     while num in N:
                         num = random.sample(range(0,25), 2)
                     N.append(num)
                     X= crossover(num[0],num[1],pop_df_new,x_reg, y_reg, x_anom, y_anom,rng)
                     
                 pop_df_sechalf = pop_df_sechalf.append({'Name': 'phi' + str(i),
                                         'Formula': X[0], 
                                         'Operation': X[2][0]}, ignore_index=True) 
                 pop_df_sechalf = pop_df_sechalf.append({'Name': 'phi' + str(i+1),
                                         'Formula': X[1], 
                                         'Operation': X[2][1]}, ignore_index=True)
                 print(X[0],X[1])
             else:
#                print('*Mutation*')
                phi,op = mutate(num[0], pop_df_new,x_reg, y_reg, x_anom, y_anom,rng)
                while (pop_df['Formula'].eq(phi)).any():
                    print('Found Repeat Formula')
                    # print('phi was: ',phi,(pop_df['Formula'].eq(phi)).any())
                    num = random.sample(range(0,25), 2)
                    while num in N:
                        num = random.sample(range(0,25), 2)
                    N.append(num)
                    phi,op = mutate(num[0], pop_df_new,x_reg, y_reg, x_anom, y_anom,rng)
                
                pop_df_sechalf = pop_df_sechalf.append({'Name': 'phi' + str(i),
                                         'Formula': phi, 
                                         'Operation': op}, ignore_index=True) 
                print(phi)
            
         pd_new = pd.concat([pop_df_new, pop_df_sechalf], ignore_index=True, axis=0)
         pop_df_new = top_half(rank(evaluate(pd_new,x_reg, y_reg, x_anom, y_anom)))
         R = pop_df_new.iloc[0]['Results'][0] + pop_df_new.iloc[0]['Results'][1]
         print('----------Population (Top 10)-----------\n',pop_df_new.head(5))
         MCR = 1 - (R/(len(x_reg.columns)+len(x_anom.columns)))
         print('MCR is: ', MCR)
         count+=1 
        
     pop_df_final = pop_df_new  
     print('----------Final Population (Top 10)-----------\n',pop_df_final.head(10))
     print('----------Best Individual-----------\n',pop_df_final.iloc[0])
     return pop_df_final.iloc[0]['Formula']
        
 
## Run the algorithm            
#Infered_STL = GA(x_reg, y_reg, x_anom, y_anom, rng_mod)