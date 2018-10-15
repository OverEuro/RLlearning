# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Exercise 2.5 (10-armed Bandits Problems with nonstationary rewards)

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

def TestProblem2(index,q_set):
    '''
    Unstationary 10-armed Bandits Problem
    '''
    R = np.random.normal(q_set[index], 1) # mean = q_set[index], var = 1
    
    return R

def BanditAlgo1(epsilon,steps = 10000):
    '''
    Stationary Bandit Algorithm
    '''
    Q_set = np.zeros(10) # the Q value for each armed (Initial values equal 0)
    q_set = np.zeros(10) # the initial means for each armed "q*"
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
        q_set = q_set + np.random.randn(10) * 0.05 # independent random walk for q*
        R = TestProblem2(ind,q_set)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 1/N_set[ind] * (R - Q_set[ind])
        
        Results.append(np.mean(R_set))
        
    return Results

def BanditAlgo2(epsilon,steps = 10000):
    '''
    Unstationary Bandit Algorithm with constant step-size
    '''
    Q_set = np.zeros(10) # the Q value for each armed (Initial values equal 0)
    q_set = np.zeros(10) # the initial means for each armed "q*"
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
        q_set = q_set + np.random.randn(10) * 0.05 # independent random walk for q*
        R = TestProblem2(ind,q_set)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 0.1 * (R - Q_set[ind])
        
        Results.append(np.mean(R_set))
        
    return Results
        
if __name__ == "__main__":
    '''
    The experiment only adopts a bandit problem, but the example in textbook adopts 2000 bandit problems.
    So if you want to get analogous results like Figure 2.2 in the textbook, please multiple running and pick a good result
    '''
    AR_on = BanditAlgo1(epsilon=0.1)
    AR_un = BanditAlgo2(epsilon=0.1)
    
plt.figure()
plt.plot(AR_on,label='stationary')
plt.plot(AR_un,label='unstationary')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.savefig('figure',dpi=600)
plt.show()