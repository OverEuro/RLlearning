# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Section 5.4 (Blackjack)
Monte Carlo Control without Exploring Starts (epsilon-soft)
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

def main(n_episodes,epsilon):
    '''
    n_episodes = number of episodes
    Note: in case some states are not visited and result that reward=0, the average reward = 0 / 0,
    that is meaningless, so please set n_episodes >= 1e+5 to guarantee each state is visited.
    
    n_states = number of states
    pol      = initial random soft policy set. equal probability for stick or hit #tensor#
    pol_taken= policy taken
    v_action = state-action pair values #tensor#
    Q        = initial Q value or action value #4-dimensional array:
    [SUMhandcards, showingcard, ace_use, action]#
    t_rewSUM = sum of rewards for each state-action pair #4-dimensional array#
    v_states = number of visiting for each state-action pair #4-dimensional array#
    states_obs = states obversed
    '''
    #    n_states = int((21 - 12 + 1) * 13 * 2)
    #pol = np.random.randint(2,size=(10,13,2))
    pol = np.ones((10,13,2,2)) * 0.5
    pol_taken = np.ones((10,13,2))
    Q = np.zeros([10,13,2,2])
    t_rewSUM = np.zeros([10,13,2,2])
    v_states = np.zeros([10,13,2,2])
    
    for i in range(n_episodes):
        states_obs = np.ones((1,3),dtype='int')
        # re-generate a deck
        deck = shufflecards()
        
        # the player gets the first two cards
        p = deck[0:2]
        pv, use_ace = handValue(p)
        # remove the two cards from the deck array
        deck = deck[2::]
        # the dealer gets the next two cards (and shows the first card)
        d = deck[0:2]
        # remove the two cards from the deck_rem array
        deck = deck[2::]
        dv, use_ace = handValue(d)
        cardshow = d[0]
        
        while (pv < 12): # the policy is always to hit
            p = np.hstack((p,deck[0]))
            deck = deck[1::] # Hit
            pv, use_ace = handValue(p)
        
        # accumulate the first state obversed
        state, pv = statefromHand(p,cardshow)
        states_obs[0,:] = state
        # sample from (0,1) according to the P(0,1|A)
        action = np.random.choice(2,1, p = pol[state[0]-12, state[1]-1, state[2],:])
        pol_taken[state[0]-12, state[1]-1, state[2]] = action
        # implement the policy of the player (hit until we have a sum of handvards equals 22 or the action is stick)
        while (action == 1 and pv < 22):
            p = np.hstack((p,deck[0]))
            deck = deck[1::] # Hit
            state, pv = statefromHand(p,cardshow)
            states_obs = np.vstack((states_obs,state))
            if (pv <= 21):
                action = np.random.choice(2,1, p = pol[state[0]-12, state[1]-1, state[2],:])
                pol_taken[state[0]-12, state[1]-1, state[2]] = action
        # implement the fixed policy of the dealer (hit until we have a sum of handvards equals 17)
        while (dv < 17):
            d = np.hstack((d,deck[0]))
            deck = deck[1::] #Hit
            dv, use_ace = handValue(d)
        # get the reward according to pv and dv
        rew = get_rew(pv, dv)
        # accumulate these rewards and number of visiting
        for j in range(np.size(states_obs,0)): # the number of raw of states_obs
            if (states_obs[j,0] >= 12 and states_obs[j,0] <= 21):
                action = int(pol_taken[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2]])
                t_rewSUM[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action] += rew
                v_states[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action] += 1
                Q[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action] = t_rewSUM[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action]/ \
                v_states[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action]
                action_new = np.argmax(Q[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], :])
                for q in range(2):
                    if q == action_new:
                        pol[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], action_new] = 1 - epsilon + epsilon / 2
                    else:
                        pol[states_obs[j,0]-12, states_obs[j,1]-1, states_obs[j,2], q] = epsilon / 2

    return pol_taken

if __name__ == "__main__":
    pol_taken = main(5000000,0.1)
    
    # ploting
    
    plt.figure()
    plt.imshow(pol_taken[:,:,0], interpolation='nearest')
    plt.colorbar()
    plt.savefig('policy_no_use_soft',dpi=600)
    plt.show()
    
    plt.figure()
    plt.imshow(pol_taken[:,:,1], interpolation='nearest')
    plt.colorbar()
    plt.savefig('policy_use_soft',dpi=600)
    plt.show()