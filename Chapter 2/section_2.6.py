# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Section 2.6 (Optimistic Initial Values with stationary bandit)
"""
import numpy as np
import matplotlib.pyplot as plt

def TestProblem1(index):
    '''
    Stationary 10-armed Bandits Problem
    '''
    q_set = np.array([0.5,-0.5,2,0.8,1.1,-1.3,-0.2,-1,1,-0.9]) # the means of 10 guassian distributions
    R = np.random.normal(q_set[index], 1) # mean = q_set[index], var = 1
    
    return R

def BanditAlgo1(epsilon,steps = 1000):
    '''
    Stationary Bandit Algorithm without optimistic initial values
    '''
    Q_set = np.zeros(10) # the Q value for each armed (Initial values equal 0)
    N_set = np.zeros(10) # the array for recording times
    R_set = [] # recording rewards
    Results = [] # recording average rewards
    
    for i in range(steps): # Steps for experiments
        if np.random.rand() > epsilon:
            inds_tuple = np.where(Q_set == np.max(Q_set))
            inds = inds_tuple[0]
            ind = inds[int(np.floor(np.random.rand() * len(inds)))] # pick randomly an armed from max Q values
        else:
            ind = int(np.floor(np.random.rand() * 10)) # a random action
            
        R = TestProblem1(ind)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 1/N_set[ind] * (R - Q_set[ind])
        
        Results.append(np.mean(R_set))
        
    return Results

def BanditAlgo2(epsilon,steps = 1000):
    '''
    Stationary Bandit Algorithm with optimistic initial values
    '''
    Q_set = np.zeros(10) + 5 # the Q value for each armed (Initial values equal 0)
    N_set = np.zeros(10) # the array for recording times
    R_set = [] # recording rewards
    Results = [] # recording average rewards
    
    for i in range(steps): # Steps for experiments
        if np.random.rand() > epsilon:
            inds_tuple = np.where(Q_set == np.max(Q_set))
            inds = inds_tuple[0]
            ind = inds[int(np.floor(np.random.rand() * len(inds)))] # pick randomly an armed from max Q values
        else:
            ind = int(np.floor(np.random.rand() * 10)) # a random action
            
        R = TestProblem1(ind)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 1/N_set[ind] * (R - Q_set[ind])
        
        Results.append(np.mean(R_set))
        
    return Results

if __name__ == "__main__":
    '''
    The experiment only adopts a bandit problem, but the example in textbook adopts 2000 bandit problems.
    So if you want to get analogous results like Figure 2.3 in the textbook, please multiple running and pick a good result
    '''
    AR_without = BanditAlgo1(epsilon=0.1)
    AR_with = BanditAlgo2(epsilon=0.1)
    
plt.figure()
plt.plot(AR_without,label='realistic')
plt.plot(AR_with,label='optimistic')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.savefig('figure',dpi=600)
plt.show()