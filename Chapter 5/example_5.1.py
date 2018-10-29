# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Example 5.1 (Blackjack)
Monte Carlo Prediction or Estimation
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt

def shufflecards():
    deck = np.random.permutation(52)
    return deck

def handValue(handcards):
    # compute 1:13 indexing for each card
    values = np.mod(handcards,13) + 1
    # map face cards (11,12,13) to 10
    values = np.minimum(values,10)
    # sum of handvards
    sv = np.sum(values)
    
    # select the value of ace
    if np.any(values == 1) and sv <= 11:
        sv += 10
        use_ace = 1
    else:
        use_ace = 0
    return sv, use_ace

def statefromHand(handcards, cardshow):
    '''
    returns the state (a three vector of numbers for a given hand of cards)
    [players current sum, dealer showing card, usable ace]
    Cards vectors are represented as 0:51, such as
    0:12 ---> A, 2, 3, ..., J, Q, K (heart)
    13:25---> A, 2, 3, ..., J, Q, K (spade)
                                    (club)
                                    (diamond)
    no king and queen
    '''
    hv,use_ace = handValue(handcards)
    cardshow = np.mod(cardshow,13) + 1
    st = np.array([hv, cardshow, use_ace])
    return st, hv

def get_rew(pv, dv):
    if (pv > 21): # player goes bust
        rew = -1
        return rew
    if (dv > 21): # dealer goes bust
        rew = 1
        return rew
    if (dv == pv): # draw
        rew = 0
        return rew
    if (pv > dv):
        rew = 1
    else:
        rew = -1
    return rew

def main(n_episodes):
    '''
    n_episodes = number of episodes
    Note: in case some states are not visited and result that reward=0, the average reward = 0 / 0,
    that is meaningless, so please set n_episodes >= 1e+5 to guarantee each state is visited.
    
    n_states = number of states
    t_rewSUM = sum of rewards for each states #tensor#
    v_states = number of visiting for each states #tensor#
    states_obs = states obversed
    '''
    # n_states = int((21 - 12 + 1) * 13 * 2)
    t_rewSUM = np.zeros([10,13,2])
    v_states = np.zeros([10,13,2])
    
    for i in range(n_episodes):
        states_obs = np.ones((1,3),dtype='int')
        # re-generate a deck
        deck = shufflecards()
        
        # the player gets the first two cards
        p = deck[0:2]
        # remove the two cards from the deck array
        deck = deck[2::]
        # the dealer gets the next two cards (and shows the first card)
        d = deck[0:2]
        # remove the two cards from the deck_rem array
        deck = deck[2::]
        dv, use_ace = handValue(d)
        cardshow = d[0]
    
        # accumulate the first state obversed
        state, pv = statefromHand(p,cardshow)
        states_obs[0,:] = state
        # implement the policy of the player (hit until we have a sum of handvards equals 20 or 21)
        while (pv < 20):
            p = np.hstack((p,deck[0]))
            deck = deck[1::] # Hit
            state, pv = statefromHand(p,cardshow)
            states_obs = np.vstack((states_obs,state))
        # implement the policy of the dealer (hit until we have a sum of handvards equals 17)
        while (dv < 17):
            d = np.hstack((d,deck[0]))
            deck = deck[1::] #Hit
            dv, use_ace = handValue(d)
        # get the reward according to pv and dv
        rew = get_rew(pv, dv)
        # accumulate these rewards and number of visiting
        for j in range(np.size(states_obs,0)): # the number of raw of states_obs
            if (states_obs[j,0] >= 12 and states_obs[j,0] <= 21):
                t_rewSUM[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2]] += rew
                v_states[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2]] += 1
    
    value_fun = t_rewSUM / v_states
    return value_fun

if __name__ == "__main__":
    value_fun = main(100000)
    # generate meshgrid
    xa = np.arange(12,22,1)
    ya = np.arange(1,14,1)
    X, Y = np.meshgrid(xa, ya)
    
    fig = plt.figure()
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(value_fun[:,:,1], cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X.T, Y.T, value_fun[:,:,1], rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
    plt.savefig('est_valuefun',dpi=600)
    plt.show()
    
    fig = plt.figure()
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(value_fun[:,:,0], cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X.T, Y.T, value_fun[:,:,0], rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
    plt.savefig('est_valuefun_no',dpi=600)
    plt.show()