"""
*************************************************************************************************************
Version
    Last revision: October 2021
    Author: Spilios Evmorfos

Purpose
    The purpose of this code file is to support the paper:
    S. Evmorfos, K. Diamantaras, A. Petropulu, Reinforcement Learning for Motion Policies in Mobile Relaying Networks,
    Journal of Transactions in Signal Processing (TSP), 2021

   Any part of this code used in your work should cite the above publication.

This code is provided "as is" to support the ideals of reproducible research. Any issues with this
code should be reported by email to se386@scarletmail.rutgers.edu.

The code is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
available at https://creativecommons.org/licenses/by-nc-sa/4.0/

*************************************************************************************************************
"""



# Import Python Libraries
import argparse
import numpy as np  
import torch 
import random  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ReplayMemory import ReplayMemory
import Environment
from Siren_network import DQNNet_siren
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # initialize a device


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_fill_eps',type=int, default=1, help = 'number of episodes to fill the Memory Replay')
    parser.add_argument('--mem_capacity', type=int, default=3000, help='the overall capacity of the Memory Replay')
    parser.add_argument('--initial_mem_capacity', type=int, default=300, help='the initial capacity of the Memory Replay')
    parser.add_argument('--num_train_eps', type=int, default=300, help='the number of the training episodes')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batch')
    


    



    arguments = parser.parse_args()
    
    # Square grid boundaries
    grid_min = Environment.grid_min
    grid_max = Environment.grid_max
    spacing = Environment.spacing

    # Number of cells the relay movement area is divided into (rows, columns)
    rMap_RowCells = Environment.rMap_RowCells
    rMap_ColCells = Environment.rMap_ColCells



    # Channel Hyperparameters
    ell= Environment.ell
    rho= Environment.rho

    P_S = Environment.P_S  
    P_R = Environment.P_R
    sigmaSQ = Environment.sigmaSQ
    sigma_xiSQ = Environment.sigma_xiSQ
    sigma_DSQ = Environment.sigma_DSQ
    etaSQ = Environment.etaSQ  
    c1 = Environment.c1
    c2 = Environment.c2
    c3 = Environment.c3

    mstep=Environment.mstep
    numEpisodes = Environment.numEpisodes
    numSlots = Environment.numSlots

    numRelays = Environment.numRelays

    
    env = Environment.GridWorld(rMap_RowCells, rMap_ColCells) # create the environment
    stateSpace = env.stateSpace # create the state space - list of numbers one number for every cell
    actionSpace= env.actionSpace # create the action space - dictionary
    candidateActions=env.candidateActions # candidate actions
    actionindex = env.actionindex
    T, C_SD, C_SD_chol, kappa, pathlossF, pathlossG, grid_X, grid_Y, S_cord, D_cord = env.setState() # useful variables for the state
    _ , validMoves = Environment.Qtable_initialization(stateSpace, actionSpace, candidateActions) # we have the valid moves for every state

    
    capacity = arguments.mem_capacity
    memory = ReplayMemory(capacity=capacity)
    initial_capacity = arguments.initial_mem_capacity
    f_maps = np.load(r"C:\Users\Spilios\OneDrive\Desktop\DQL_SIREN\f_maps_n_10_c1_10_c2_20_multipath_1_symmetric.npy")
    g_maps = np.load(r"C:\Users\Spilios\OneDrive\Desktop\DQL_SIREN\g_maps_n_10_c1_10_c2_20_multipath_1_symmetric.npy")
    
    """if one wants to create a new dataset of channel data, they need to comment the two commands above and uncomment the below commands"""
    
    #f_maps, g_maps = Environment.Perfect_CSI(pathlossF, pathlossG, C_SD, C_SD_chol, kappa)
    
    """  
    First we initialize the Replay Memory with experiences-tuples from some random trajectories
    """
    
    
    for k in range(arguments.memory_fill_eps):
        current_position = np.array([[19,10],[18,10],[15,17]]) # This is the positions for the relays at the beginning of every episode
        for i in range(initial_capacity):
            new_relay_pos = np.zeros([numRelays,2], dtype=np.int)

            action=[]

            currentState = np.ravel_multi_index( current_position.T, (rMap_RowCells, rMap_ColCells) )

            for r in range(numRelays):
                candidateActions = validMoves[currentState[r]]
                temp_action = random.choice(candidateActions)
                temp_relay_pos = current_position[r,:] + np.array(actionSpace[temp_action])
                while (temp_relay_pos == new_relay_pos[0:r]).all(1).any():
                    temp_action = random.choice(candidateActions)
                    temp_relay_pos = current_position[r,:] + np.array(actionSpace[temp_action])
                    
                new_relay_pos[r,:] = temp_relay_pos
                action.append(temp_action)
            """  
            Calculate the contribution of every relay to the SINR (reward for every relay-agent)
            """
            reward_relay1 = Environment.VI_local(f_maps[new_relay_pos[0][0],new_relay_pos[0][1],i],g_maps[new_relay_pos[0][0],new_relay_pos[0][1],i])
            reward_relay2 = Environment.VI_local(f_maps[new_relay_pos[1][0],new_relay_pos[1][1],i],g_maps[new_relay_pos[1][0],new_relay_pos[1][1], i])
            reward_relay3 = Environment.VI_local(f_maps[new_relay_pos[2][0],new_relay_pos[2][1],i],g_maps[new_relay_pos[2][0],new_relay_pos[2][1],i])
            reward = []
            reward.append(reward_relay1)
            reward.append(reward_relay2)
            reward.append(reward_relay3)
            memory.store (current_position,action,new_relay_pos,reward)
            current_position = new_relay_pos
            
            
    reward_history = []
    accumulated_reward = []
    reward_per_episode_list = []
    action_size = 9
    lr = arguments.lr
    Rinit_pos = np.array([[9,3],[9,2],[9,4]])
    
    policy_net = DQNNet_siren(Rinit_pos[0].size, action_size, lr).to(device)
    target_net = DQNNet_siren(Rinit_pos[0].size, action_size, lr).to(device)
    target_net.eval()
    policy_net.train()
    batch_size = arguments.batch_size
    num_train_eps = arguments.num_train_eps
    discount = 0.99
    
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_frequency = 40
    update_epsilon_frequency = 40
    
    for ep_cnt in range(num_train_eps):
        state = np.array([[9,3],[9,2],[9,4]]) # we begin from the same position after every episode
        reward_per_episode = 0
        for j in range(400): # 400 time slots in every episode
            currentState = np.ravel_multi_index( state.T, (rMap_RowCells, rMap_ColCells) )
            action = []
            new_relay_pos = np.zeros([numRelays,2], dtype=np.int) 
            for r in range(numRelays):
                drawNum = np.random.rand()
                if drawNum <= epsilon: # for a small percentage of the time we pick an action randomly for maintaining exploration and to avoid local minima in the policy
                    valid_actions = validMoves[currentState[r]]
                    temp_action = random.choice(valid_actions)
                    temp_relay_pos = state[r,:] + np.array(actionSpace[temp_action])
                    
                    """ 
                    We check if the action chosen is valid for the relay position - check if the next position is within the grid boundaries
                    """
                    while (temp_relay_pos == new_relay_pos[0:r]).all(1).any():
                        temp_action = random.choice(valid_actions)
                        temp_relay_pos = state[r,:] + np.array(actionSpace[temp_action])
                    
                    new_relay_pos[r,:] = temp_relay_pos
                    action.append(temp_action)
                else:
                    input_relay = state[r]
                    input_relay = torch.from_numpy(input_relay)
                    input_relay = input_relay.type(torch.float)
                    
                    input_relay = input_relay.to(device)
                    
                    if r == 0:
                        with torch.no_grad():
                            action_choice = policy_net.forward(input_relay)
                            
                    elif r == 1:
                        with torch.no_grad():
                            action_choice = policy_net.forward(input_relay)
                    elif r == 2:
                        with torch.no_grad():
                            action_choice = policy_net.forward(input_relay)
                    action_choice = action_choice*torch.tensor([0.999,0.999,0.999,0.999,1,0.999,0.999,0.999,0.999]).to(device) #subtract a small percentage for all the actions
                    # except for the action "stay" to encourage energy preservation
                    action_index = torch.argmax(action_choice).item()
                    action_chosen = candidateActions[action_index]
                    valid_actions = validMoves[currentState[r]] 
                    while action_chosen not in valid_actions:
                        action_choice[action_index] = -10000.
                        action_index = torch.argmax(action_choice).item()
                        action_chosen = candidateActions[action_index]
                    
                    temp_relay_pos = state[r,:] + np.array(actionSpace[action_chosen])
                    
                    while (temp_relay_pos == new_relay_pos[0:r]).all(1).any() or action_chosen not in valid_actions :
                        action_choice[action_index] = -10000.
                        action_index = torch.argmax(action_choice).item()
                        action_chosen = candidateActions[action_index]
                        
                        temp_relay_pos = state[r,:] + np.array(actionSpace[action_chosen])
                        
                    new_relay_pos[r,:] = temp_relay_pos 
                    action.append(action_chosen)
                    print(action_chosen)
            # at this point we have calculated the action and new state for every relay
            print(new_relay_pos) # we print the new state (position)
            reward_relay1 = Environment.VI_local(f_maps[new_relay_pos[0][0],new_relay_pos[0][1],j],g_maps[new_relay_pos[0][0],new_relay_pos[0][1],j])
            reward_relay2 = Environment.VI_local(f_maps[new_relay_pos[1][0],new_relay_pos[1][1],j],g_maps[new_relay_pos[1][0],new_relay_pos[1][1],j])
            reward_relay3 = Environment.VI_local(f_maps[new_relay_pos[2][0],new_relay_pos[2][1],j],g_maps[new_relay_pos[2][0],new_relay_pos[2][1],j])
            """
            At this point, for all relays we have calculated the action, the new state and the reward (contribution to the SINR)
            """
            reward = []
            reward.append(reward_relay1)
            reward.append(reward_relay2)
            reward.append(reward_relay3)
            reward_slot = reward_relay1 + reward_relay2 + reward_relay3
            print(reward_slot)
            reward_per_episode = reward_per_episode + reward_slot
            accumulated_reward.append(reward_slot)
            memory.store(state,action,new_relay_pos,reward) # store the experience- tuple to the Replay Memory
            state = new_relay_pos # set the new state to be the current state
            
            """ 
            We sample a batch of experiences for the Deep Q Network weight update (gradient descent)
            """
            indices_to_sample = random.sample(range(len(memory.buffer_state)), batch_size)
            states = [memory.buffer_state[i] for i in indices_to_sample]
            action_sample = [memory.buffer_action[i] for i in indices_to_sample]
            next_states = [memory.buffer_next_state[i] for i in indices_to_sample]
            rewards = [memory.buffer_reward[i] for i in indices_to_sample]

            states_relay1 = np.zeros([len(states),2], dtype = np.int)
            
            for i in range(len(states)):
                states_relay1[i] = states[i][0]
            
            actions = np.zeros([len(states),1], dtype = np.int)

            for i in range(len(states)):
                actions[i] = np.int(actionindex[action_sample[i][0]])
                
            actions = torch.from_numpy(actions)
            actions.to(device)
            
            next_states_relay1 = np.zeros([len(states),2], dtype = np.int)
                
            for i in range(len(states)):
                next_states_relay1[i] = next_states[i][0]
                
            
            rewards_relay1 = np.zeros([len(states),1], dtype = np.float)

            for i in range(len(states)):
                rewards_relay1[i] = rewards[i][0]
                
            rewards_relay1 = np.array(rewards_relay1)
            rewards_relay1 = torch.from_numpy(rewards_relay1)
            rewards_relay1 = torch.tensor(rewards_relay1, dtype = torch.float32)
            rewards_relay1 = rewards_relay1.to(device)
            
            states_relay1 = torch.from_numpy(states_relay1)
            states_relay1 = torch.tensor(states_relay1, dtype = torch.float32)
            states_relay1 = states_relay1.to(device)
            actions = actions.to(device)
            actions = torch.tensor(actions, dtype = torch.int64)
            q_pred = policy_net.forward(states_relay1).gather(1, actions)
            
            next_states_relay1 = torch.from_numpy(next_states_relay1)
            next_states_relay1 = torch.tensor(next_states_relay1, dtype = torch.float32)
            next_states_relay1 = next_states_relay1.to(device)
            q_target = target_net.forward(next_states_relay1).max(dim=1).values
            q_target = q_target.view(-1,1)
            y_j = rewards_relay1 + (discount * q_target)
            y_j = y_j.view(-1, 1)

            y_j = torch.tensor(y_j, dtype=torch.float32)

            """ Learning with gradient descent 
            """
            policy_net.optimizer.zero_grad()
            loss = F.mse_loss(y_j, q_pred).mean()
            loss.backward()
            policy_net.optimizer.step()
            
            if j % update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict()) # copy the weights of the Q Network to the Target Network 


            if j % update_epsilon_frequency == 0:
                epsilon = max(epsilon_min, epsilon*epsilon_decay)
        
        print("this is the end of episode", ep_cnt)
        reward_per_episode_list.append(reward_per_episode)   
        
        
    """ 
    Plotting the average SINR at the destination for every episode
    """  
    reward_per_episode_list1 = np.array(reward_per_episode_list)
    k = reward_per_episode_list1.size

    a = [i for i in range(k)]

    results = np.zeros(k)
    for i in range(k):
        results[i] = reward_per_episode_list[i]/400
        results[i] = 10*np.log10(results[i])

    plt.plot(a,results)
    plt.show()
    

    
         



                    


    
    
    
    
        
    
    
    
            
            
