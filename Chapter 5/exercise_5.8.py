# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Exercise 5.8 (Racetrack)
Off-policy Monte Carlo Control with epsilon-soft
"""

import numpy as np
import matplotlib.pyplot as plt

def Track():
    Tr = np.zeros((16,16))
    # set race track: the position of track equals 1
    Tr[15,3:12] = 1
    Tr[14,3:12] = 1
    Tr[9:14,4:12] = 1
    Tr[7:9,6:12] = 1
    Tr[6,6:13] = 1
    Tr[5,7:13] = 1
    Tr[4,7:16] = 1
    Tr[3,7:16] = 1
    Tr[2,8:16] = 1
    Tr[1,10:16] = 1
    # plot the race track
    plt.figure()
    plt.imshow(Tr, interpolation='nearest')
    plt.colorbar()
    plt.show()
    return Tr

def ini_policy(RT,npi,npj,dvi,dvj,dai,daj):
    '''
    Initialize a uniformly random policy
    '''
    pol = np.zeros((npi,npj,dvi,dvj,dai,daj))
    
    # check whether valid
    for pi in range(npi):
        for pj in range(npj):
            for vi in range(dvi):
                for vj in range(dvj):
                    # if not a valid position
                    if RT[pi,pj] != 1:
                        continue
                    possacts = np.ones((dai,daj))
                    if vi == 0:
                        possacts[0,:] = 0
                    if vj == 0:
                        possacts[:,0] = 0
                    if vi == 1 and vj == 0:
                        possacts[0,1] = 0
                    if vi == 0 and vj == 1:
                        possacts[1,0] == 0
                    if vi == 1 and vj == 1:
                        possacts[0,0] = 0
                    if vi == dvi - 1:
                        possacts[2,:] = 0
                    if vj == dvj - 1:
                        possacts[:,2] = 0
                    uniprob = possacts / np.sum(possacts)
                    pol[pi,pj,vi,vj,:,:] = uniprob
    return pol
                
        
def getepisode(RT,ei,pol,pst,npst,npi,npj,dvi,dvj,dai,daj):
    
    states_obs = np.ones((1,4),dtype='int')
    act_taken = np.zeros((npi,npj,dvi,dvj))
    rew_taken = np.zeros((npi,npj,dvi,dvj,dai,daj))
    # the initial position
    pii = npi - 1
    pjj = pst[int(np.mod(ei,npst))]
    # the initial velocity
    vii = 0
    vjj = 0
    # store the first state obversed
    states_obs[0,:] = np.array([pii,pjj,vii,vjj])
    actions = np.array([0,1,2,3,4,5,6,7,8])
    while (True):
        str_pol = np.reshape(pol[pii,pjj,vii,vjj,:,:],9)
#        print(str_pol)
        action = np.random.choice(actions,1,p=str_pol)
        act_taken[pii,pjj,vii,vjj] = action
        # get the index in pol matrix [3 * 3]
        djj = int(np.mod(action,3))
        dii = int((action - djj) / 3)
        # the specific actions selected in {-1,0,1}
        ai = dii - 1
        aj = djj - 1
        # update state according to this action and get a reward
        newvi = vii + ai
        newvj = vjj + aj
        # check boundary
        if (newvi < 0 or newvi > 4):
            newvi = vii
        if (newvj < 0 or newvj > 4):
            newvj = vjj
        vii = newvi
        vjj = newvj
        if np.random.rand() < 0.1:
            vii = 0
            vjj = 0
        pii -= vii # drive up (decrease row index)
        pjj += vjj # drive right (increase column index)
        # check the projected path
        if pii > 15 or pii < 0:
            # the initial position
            pii = npi - 1
            pjj = pst[int(np.mod(ei,npst))]
            # the initial velocity
            vii = 0
            vjj = 0
            continue
        if pjj > 15 or pjj < 0:
            # the initial position
            pii = npi - 1
            pjj = pst[int(np.mod(ei,npst))]
            # the initial velocity
            vii = 0
            vjj = 0
            continue
        ind = RT[pii,pjj]
        if ind == 1:
            # get the reward
            if pjj == npj - 1: # cross the finish line
                reward = 1
                rew_taken[pii,pjj,vii,vjj,dii,djj] += reward
                state = np.array([pii,pjj,vii,vjj])
                states_obs = np.vstack((states_obs,state))
                break
            else:
                reward = -1
                rew_taken[pii,pjj,vii,vjj,dii,djj] += reward
                state = np.array([pii,pjj,vii,vjj])
                states_obs = np.vstack((states_obs,state))
        else:
            # the initial position
            pii = npi - 1
            pjj = pst[int(np.mod(ei,npst))]
            # the initial velocity
            vii = 0
            vjj = 0
    
    return states_obs, act_taken, rew_taken
    
    
def main():
    '''
    This experiment's time cost is very huge, so please make sure your computer
    is fast enough before you try the full experiment with n_epsidos = 5e+6! Maybe
    it will take 40-45 minutes or longer...
    '''
    n_epsidos = int(1e+3)
    
    # generate race and some parameters
    RT = Track()
    npi,npj = np.shape(RT)
    # the dimension of velocity state
    dvi = 5
    dvj = 5
    # the dimension of actions
    dai = 3
    daj = 3
    '''
    a state consists of [npi,npj,dvi,dvj] with npi = 0:15, npj = 0:15, dvi = 0:4
    , dvj = 0:4, so the number of states is equal to 16**2*5**2 = 6400!
    '''
    Q = np.zeros((npi,npj,dvi,dvj,dai,daj)) # the initial action-value function
    C = np.zeros((npi,npj,dvi,dvj,dai,daj))
    opt_policy = np.zeros((npi,npj,dvi,dvj))

    # give the possible starting position
    t_pst = np.nonzero(RT[15,:])
    pst = t_pst[0]
    npst = len(pst)
    
    # initialize the policy
    for ei in range(n_epsidos):
        pol = ini_policy(RT,npi,npj,dvi,dvj,dai,daj)
        states_obs, act_taken, rew_taken = getepisode(RT,ei,pol,pst,npst,npi,npj,dvi,dvj,dai,daj)
        path_1 = states_obs[:,0:2]
        # initial parameters G=0 and W=1
        G = 0
        W = 1
        for j in range(np.size(states_obs,0)):
            ind = np.size(states_obs,0) - 1 - j
            # out of the reward
            action = act_taken[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3]]
            # get the index in pol matrix [3 * 3]
            djj = int(np.mod(action,3))
            dii = int((action - djj) / 3)
            G += rew_taken[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj]
            C[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj] += W
            Q[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj] += \
            W / C[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj] * \
            (G - Q[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj])
            new_action = np.argmax(Q[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],:,:])
            opt_policy[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3]] = new_action
            
            if new_action != action:
                break
            W = W * 1/(pol[states_obs[ind,0],states_obs[ind,1],states_obs[ind,2],states_obs[ind,3],dii,djj])
            
    return opt_policy, path_1

if __name__ == "__main__":
    opt_policy, path_1 = main()
    # plot all pathes according states obversed
    '''
    split the array into multiple parts. because the states obversed include restarting
    path when car intersects anywhere except finial line.
    '''
    set_path = []
    path = []
    for i in range(np.size(path_1,0)):
        if i > 0:
            if path_1[i,0] == 14 and path_1[i-1,0] < 14:
                set_path.append(path)
                path = []
            if path_1[i,0] == 15 and path_1[i-1,0] < 15:
                set_path.append(path)
                path = []
        path.append(path_1[i,:])
        if i == np.size(path_1,0) - 1:
            set_path.append(path)
    plt.figure()
    a = len(set_path)
    for i in range(a):
        b = len(set_path[i])
        subset = set_path[i]
        path = np.ones((b,2))
        for j in range(b):
            path[j,:] = np.array([subset[j]])
        # plot each path tried by car no matter fail(back to starting point) or success (end)
        plt.plot(path[:,1],15-path[:,0])
        plt.xlim((0,15))
        plt.ylim((0,15))
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            