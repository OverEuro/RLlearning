# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Section 2.8 (Gradient Bandit Algorithm)
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

def BanditAlgo1(alpha,steps = 1000):
    '''
    Gradient Bandit Algorithm
    '''
    Q_set = np.zeros(10) # the Q value for each arm (Initial values equal 0)
    H_set = np.zeros(10) # the perference value for each arm
    P_set = np.exp(H_set) / np.sum(np.exp(H_set)) # the probability of taking action a at time t
    N_set = np.zeros(10) # the array for recording times
    R_set = [] # recording rewards
    index = 0 # the action index for perference updating
    reward = 0 # initial reward
    baseline = 0 # average reward as baseline and initial value equals zero
    Results = [] # recording average rewards
    
    for i in range(steps): # Steps for experiments
        if i == 0:
            ind = int(np.floor(np.random.rand() * 10)) # pick randomly action
        else:
            for j in range(10):
                if j == index:
                    H_set[j] = H_set[j] + alpha * (reward - baseline) * (1 - P_set[j])
                else:
                    H_set[j] = H_set[j] - alpha * (reward - baseline) * P_set[j]
            # updating the selection probability based on soft-max distribution
                P_set = np.exp(H_set) / np.sum(np.exp(H_set))
                inds_tuple = np.where(P_set == np.max(P_set))
                inds = inds_tuple[0]
                ind = inds[int(np.floor(np.random.rand() * len(inds)))] # pick randomly an arm from max Q values
            
        R = TestProblem1(ind)
        R_set.append(R)
        N_set[ind] += 1
        Q_set[ind] = Q_set[ind] + 1/N_set[ind] * (R - Q_set[ind])
        # updating current index, reward and baseline
        index = ind
        reward = R
        baseline = Q_set[ind]
        
        Results.append(np.mean(R_set))
        
    return Results

if __name__ == "__main__":
    '''
    The experiment only adopt a bandit problem, but the example in textbook adopts 2000 bandit problems.
    So if you want to get analogous results like Figure 2.5 in the textbook, please multiple running, 
    change the parameters such as alpha and baseline and pick a good result.
    '''
    AR_without = BanditAlgo1(alpha=0.4)
    AR_with = BanditAlgo1(alpha=0.1)
    
plt.figure()
plt.plot(AR_without,label='alpha = 0.4')
plt.plot(AR_with,label='alpha = 0.1')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.savefig('figure',dpi=600)
plt.show()