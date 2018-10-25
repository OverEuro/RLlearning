# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Example 4.2 (Jack's Car Rental)
"""

import numpy as np
import matplotlib.pyplot as plt

def possion(n,lambda_r):
    p = lambda_r**n / np.math.factorial(n) * np.exp(-lambda_r)
    return p

def get_p_r(rent,lambda_req,lambda_ret,max_n_car,max_n_tran):
    '''
    computing the transition probability matrix and reward array
    '''
    # the number of possible cars at any location in the morning (plus the transition cars)
    ncm = np.arange(0,max_n_car+max_n_tran+1,1)
    # the array of reward
    R = np.zeros(len(ncm))
    for n in ncm:
        t = 0
        '''
        the expectations when ncm = 0,1,2,...,25. Note that the total number of
        cars can be rented is ncm + nret.
        '''
        for nreq in range(max_n_car+max_n_tran+1): # <- the max number of requets
            t += rent * min(n, nreq) * possion(nreq, lambda_req)
        R[n] = t
    # the transition probability matrix
    P = np.zeros([len(ncm),max_n_car+1])
    for nreq in range(max_n_car+max_n_tran+1):
        # for all possible requests
        preq = possion(nreq, lambda_req)
        for nret in range(max_n_car+max_n_tran+1):
            # for all possible return
            pret = possion(nret, lambda_ret)
            for n in ncm:
                # for all possible numbers in the morning
                loc_req = min(n,nreq)
                n_even = max(0, min(max_n_car, n + nret - loc_req))
                P[n,n_even] += preq * pret
                '''
                state trans probability = sum(all same situation), so we use the
                "+=" for same position in the matrix.
                '''
    return R, P

def bellman(sa,sb,ntrans,V,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran):
    # restrict the action
    ntrans = max(-sb,min(ntrans,sa))
    ntrans = max(-max_n_tran,min(max_n_tran,ntrans))
    
    t = -2 * abs(ntrans) # pay $2 for each car
    sa_morning = int(sa - ntrans)
    sb_morning = int(sb + ntrans)
    for na in range(max_n_car+1):
        # all possible states for location A in the end of day (next state)
        for nb in range(max_n_car+1):
            # all possible states for location B in the end of day (next state)
            pa = pma[sa_morning,na]
            pb = pmb[sb_morning,nb]
            '''
            you need to think the transfer cost for all situation, and the cost
            cannot be involved into reward.
            '''
            t = t + pa*pb * (ra[sa_morning] + rb[sb_morning] + gamma * V[na,nb])
    return t

def IPE(V,pol,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran):
    # the total number of states
    n_states = int((max_n_car + 1) ** 2)
    # some parameters for IPE
    max_iters = 100
    n_iter = 0
    theta = 1e-6
    delta = 1e+6
    
    # the policy evaluation loop
    while (delta > theta and n_iter < max_iters):
        delta = 0
        '''
        each item in state value matrix represents a state, the total number is
        equal to (max_n_car + 1) ** 2
        '''
        for sa in range(max_n_car + 1):
            for sb in range(max_n_car + 1):
                # save the old state value for this location
                vold = V[sa,sb]
                # transfer certain cars from A to B according to the policy
                ntrans = pol[sa,sb]
                # update the V in-place
                V[sa,sb] = bellman(sa,sb,ntrans,V,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran)
                delta = max(delta, abs(vold - V[sa,sb]))
        n_iter += 1
    return V

def PI(V,pol,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran):
    # first assume the policy is stable
    policy_stable = 1
    # for each state
    for sa in range(max_n_car + 1):
        for sb in range(max_n_car + 1):
            # our policy says in this state
            be = pol[sa,sb]
            '''
            are there any better actions for this state?
            ---consider all possible actions in this state:
                we can transfer from A to B
                we can transfer from B to A (negtive number)
            '''
            ta = min(sa,max_n_tran)
            tb = min(sb,max_n_tran)
            # all possible actions in the state is 
            sas = np.arange(-tb,ta+1,1)
            n_as = len(sas)
            Q = -1e+6 * np.ones(n_as)
            for i in range(n_as):
                ntrans = sas[i]
                Q[i] = bellman(sa,sb,ntrans,V,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran)
            # check if this policy gives the best action
            ind_max = np.argmax(Q)
            best_policy = sas[ind_max]
            if (best_policy != be):
                policy_stable = 0
                pol[sa,sb] = best_policy
    return pol, policy_stable

def main(max_n_car,max_n_tran,lambda_a_ret,lambda_a_req,lambda_b_ret,lambda_b_req,gamma):
    '''
    max_n_car: the maximum number of cars we can park (overnight) at each location
    max_n_tran: the maximum number of cars we can transit (overnight)
    lambda_x_ret or req: enviroment parameters for location x
    gamma: discount rate
    '''
    rent = 10
    # pre-compute the reward array and transition probability matrix
    ra,pma = get_p_r(rent,lambda_a_req,lambda_a_ret,max_n_car,max_n_tran)
    rb,pmb = get_p_r(rent,lambda_b_req,lambda_b_ret,max_n_car,max_n_tran)
    
    # initial state value matrix
    V = np.zeros([max_n_car+1,max_n_car+1])
    # initial policy
    pol = np.zeros([max_n_car+1,max_n_car+1])
    
    # iterative policy improvement
    policy_stable = 0 
    n_iter = 0
    while (policy_stable == 0):
        # iterative policy evaluate
        V = IPE(V,pol,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran)
        # policy improvement
        pol, policy_stable = PI(V,pol,gamma,ra,pma,rb,pmb,max_n_car,max_n_tran)
        n_iter += 1
#        if n_iter > 2:
#            break
    return V, pol

if __name__ == "__main__":
    V, pol = main(max_n_car = 20,max_n_tran = 5,lambda_a_ret = 3, \
                         lambda_a_req = 3,lambda_b_ret = 2,lambda_b_req = 4,gamma = 0.9)
    plt.figure()
    plt.imshow(pol, interpolation='nearest')
    plt.colorbar()
    plt.savefig('opt_policy',dpi=600)
    plt.show()
    
    plt.figure()
    plt.imshow(V, interpolation='nearest')
    plt.colorbar()
    plt.savefig('opt_Value',dpi=600)
    plt.show()
    
    
        
        
        
        
        
        
        
        
        