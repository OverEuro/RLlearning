# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Section 2.7 (Upper-Confidence-Bound Action Selection)
"""

import numpy as np
import matplotlib.pyplot as plt

def TestProblem1(index,q_set):
    '''
    Unstationary 10-armed Bandits Problem
    '''
    R = np.random.normal(q_set[index], 1) # mean = q_set[index], var = 1
    
    return R
def BanditAlgo1(epsilon,steps = 1000):
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
        R = TestProblem1(ind,q_set)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 0.1 * (R - Q_set[ind])
        
        Results.append(np.mean(R_set))
        
    return Results

def BanditAlgo2(steps = 1000):
    '''
    Unstationary Bandit Algorithm with constant step-size and UBC selection
    '''
    Q_set = np.zeros(10) # the Q value for each armed (Initial values equal 0)
    q_set = np.zeros(10) # the initial means for each armed "q*"
    N_set = np.zeros(10) # the array for recording times
    R_set = [] # recording rewards
    Results = [] # recording average rewards
    
    for i in range(steps): # Steps for experiments
        if np.prod(N_set) == 0: # judging whether existing the "N_set[a] == 0", if N_set[a] = 0, then a is a maximizing action
            inds_tuple = np.where(N_set == 0)
            inds = inds_tuple[0]
            ind = inds[int(np.floor(np.random.rand() * len(inds)))]
        else:
            Q_set_UCB = Q_set + 2 * np.sqrt(np.log(i) / N_set)
            inds_tuple = np.where(Q_set_UCB == np.max(Q_set_UCB))
            inds = inds_tuple[0]
            ind = inds[int(np.floor(np.random.rand() * len(inds)))] # pick randomly an armed from max Q values
        q_set = q_set + np.random.randn(10) * 0.05 # independent random walk for q*
        R = TestProblem1(ind,q_set)
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
    AR_un_epsilon = BanditAlgo1(epsilon=0.1)
    AR_un_ucb = BanditAlgo2()
    
plt.figure()
plt.plot(AR_un_epsilon,label='epsilon = 0.1')
plt.plot(AR_un_ucb,label='ucb algorithm')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.savefig('figure_02',dpi=600)
plt.show()