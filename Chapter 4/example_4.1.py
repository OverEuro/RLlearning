# -*- coding: utf-8 -*-
"""
Reinforcement Learning: An Introduction

Example 4.1 (Iterative Policy Evaluation, IPE)

Input policy pi, the policy to be evaluated
Initialize an array V(s) = 0, for all s belong to S_plus (episodic task)
"""

import numpy as np

def get_v(V_old,j,q,re):
    '''
    The order is "up, down, left and right", if it's out of the bound, the agent
    will leave the state unchanged
    '''
    # Up moving
    x_up_row = j - 1
    if x_up_row < 0:
        x_up_row = j
    x_up_col = q
    # down moving
    x_down_row = j + 1
    if x_down_row > 3:
        x_down_row = j
    x_down_col = q
    # left moving
    x_left_col = q - 1
    if x_left_col < 0:
        x_left_col = q
    x_left_row = j
    # right moving
    x_right_col = q + 1
    if x_right_col > 3:
        x_right_col = q
    x_right_row = j
    
    v = 1/4 * (4 * re + V_old[x_up_row,x_up_col]\
               + V_old[x_down_row,x_down_col]\
               + V_old[x_left_row,x_left_col]\
               + V_old[x_right_row,x_right_col])
    return v

def get_opt_policy(V_final,j,q):
    '''
    Input an arbitrary position (j,q), the algotirhm will give the optimal action
    sequences to get the terminal state
    '''
    p_x = j
    p_y = q
    act_seq = []
    neighbor_p = np.ones((4,2))
    neighbor_v = np.ones(4)
    if V_final[p_x,p_y] == 0:
        act_seq.append(np.array([p_x,p_y])) # do not moving !
    while V_final[int(p_x),int(p_y)] != 0:
        # Up moving
        x_up_row = p_x - 1
        if x_up_row < 0:
            x_up_row = p_x
        x_up_col = p_y
        neighbor_p[0,0] = x_up_row
        neighbor_p[0,1] = x_up_col
        neighbor_v[0] = V_final[int(x_up_row),int(x_up_col)]
        # down moving
        x_down_row = p_x + 1
        if x_down_row > 3:
            x_down_row = p_x
        x_down_col = p_y
        neighbor_p[1,0] = x_down_row
        neighbor_p[1,1] = x_down_col
        neighbor_v[1] = V_final[int(x_down_row),int(x_down_col)]
        # left moving
        x_left_col = p_y - 1
        if x_left_col < 0:
            x_left_col = p_y
        x_left_row = p_x
        neighbor_p[2,0] = x_left_row
        neighbor_p[2,1] = x_left_col
        neighbor_v[2] = V_final[int(x_left_row),int(x_left_col)]
        # right moving
        x_right_col = p_y + 1
        if x_right_col > 3:
            x_right_col = p_y
        x_right_row = p_x
        neighbor_p[3,0] = x_right_row
        neighbor_p[3,1] = x_right_col
        neighbor_v[3] = V_final[int(x_right_row),int(x_right_col)]
        
        # Updating position
        ind = np.argmax(neighbor_v)
        p_x = neighbor_p[ind,0].copy()
        p_y = neighbor_p[ind,1].copy()
        # recording action sequence
        act_seq.append(np.array([p_x,p_y]))
    
    return act_seq    

def IPE1(number):
    '''
    no in-place algorithm
    number: the number of iterations
    '''
    V_old = np.zeros((4,4)) # old array
    V_new = np.zeros((4,4)) # new array
    re = -1 # reward for each movement
    
    for i in range(number):
        for j in range(4):
            for q in range(4):
                V_new[j,q] = get_v(V_old,j,q,re)
                '''
                The terminal states
                '''
                if j == 0 and q == 0:
                    V_new[j,q] = 0
                if j == 3 and q == 3:
                    V_new[j,q] = 0
                V_old = V_new.copy()
    return V_new

if __name__ == "__main__":
    V_final = IPE1(1000)
    print(V_final)
    opt_seq = get_opt_policy(V_final, 2, 1) # the starting point is (1,2) in the matrix
    print(opt_seq)