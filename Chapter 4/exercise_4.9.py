# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Exercise 4.9 (Gambler's Game)
"""

import numpy as np
import matplotlib.pyplot as plt

def bellman(s, a, V):
    '''
    after an action, there could be two dependent states: win or lose
    and the reward equals zero unless the state reaches 100
    '''
    s_win = s + a
    s_los = s - a
    p_h = 0.4
    
    v = p_h * V[s_win] + (1-p_h) * V[s_los]
    return v

def main():
    '''
    Part 1: Value Iteration
    '''
    # initialization
    n_states = 101
    states = np.arange(0,n_states,1)
    V = np.zeros(n_states)
    V[100] = 1
    # convergence parameters
    theta = 1e-8
    delta = 1e+6
    n_iter = 0
    plt.figure()
    while (delta > theta):
        n_iter += 1
        print(n_iter)
        delta = 0
        for i in range(n_states - 2):
            ind = i + 1
            # store the old value for computing delta
            vold = V[ind]
            # the set of all possible actions for certain state
            set_a = np.arange(1, min(states[ind], 100-states[ind])+1,1)
            n_set_a = len(set_a)
            Q = np.zeros(n_set_a)
            for j in range(n_set_a):
                Q[j] = bellman(states[ind], set_a[j], V)
            ind_best = np.argmax(Q)
            V[ind] = Q[ind_best]
            delta = max(delta,abs(vold - V[ind]))
#            print(delta)
        plt.plot(V)
    plt.show()
    '''
    Part 2: Greedy Policy "one sweep"
    '''
    pol = np.zeros(n_states)
    for i in range(n_states - 2):
        ind = i + 1
        # the set of all possible actions for certain state
        set_a = np.arange(1, min(states[ind], 100-states[ind])+1,1)
        n_set_a = len(set_a)
        Q = np.zeros(n_set_a)
        best_v = -1e+6
        best_a = 0
        for j in range(n_set_a):
            Q[j] = bellman(states[ind], set_a[j], V)
            if Q[j] - 1e-8 > best_v:
                best_v = Q[j]
                best_a = set_a[j]
#        ind_best = np.argmax(Q)
        pol[ind] = best_a
    # plot the optimistic polict
    plt.figure()
    plt.plot(pol[1:99])
    plt.show()
    
    return V, pol
    
if __name__ == "__main__":
    V, pol = main()