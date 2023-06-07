import numpy as np
import random
from collections import defaultdict

class Player:
    def __init__(self, 
                 position = [0,0],
                 movements = ['left','right','up','down','stay']
                ):
        self.movements = movements
        self.position = position
        self.barriers = [[0, 1], [2,1]]    # Cells above the two barriers
        self.starting_positions = [[0, 0], [2, 1]]

        
    def move(self, movement):
        '''
        Compute the new position of a player after performing a movement.
        movement (string) : the movement to perform. Invalid string values are interpreted as the 'stay' movement.     
        '''
        if movement == 'left'  and 'left' in self.movements:
            new_position = [self.position[0] - 1, self.position[1]]
        elif movement == 'right' and 'right' in self.movements:
            new_position = [self.position[0] + 1, self.position[1]]
        elif movement == 'up' and 'up' in self.movements:
            new_position = [self.position[0], self.position[1] + 1]
            if new_position in self.barriers and self.position in self.starting_positions:
                # 50% chance to go through the barrier
                if random.uniform(0, 1) < 0.5:     
                    new_position = self.position
                else:
                    None
        elif movement == 'down' and 'down' in self.movements:
            new_position = [self.position[0], self.position[1] - 1]
        else:  
            new_position = self.position

        return new_position
   

class Grid:
    def __init__(self,
                 length = 2, 
                 width = 2, 
                 players = [Player(),Player()],
                 reward_coordinates = [1,1],
                 reward_value = 20,
                 obstacle_coordinates = [],
                 collision_allowed = False,
                 collision_penalty = 0):
        self.length = length
        self.width = width
        self.players =  players
        self.reward_coordinates = reward_coordinates
        self.reward_value = reward_value
        self.obstacle_coordinates = obstacle_coordinates
        self.collision_allowed = collision_allowed
        self.collision_penalty =  collision_penalty
        self.joint_player_coordinates = [players[0].position, players[1].position]
        self.states = self.joint_states()


    def get_player_0(self):
        return self.players[0]


    def get_player_1(self):
        return self.players[1]
    

    def joint_states(self):
        '''
        Returns a list of all possible joint states in the game.
        '''
        if not self.collision_allowed:
            #Agents are only allowed to collide on the reward cell, whether they arrive there at the same time or not
            joint_states = [[[i,j],
                             [k,l]] for i in range(self.length) for j in range(self.width) 
                             for k in range(self.length) for l in range(self.width)
                             if [i,j] != [k,l]  and [i,j] not in self.obstacle_coordinates and [k,l] not in self.obstacle_coordinates
            ]
            joint_states.append([self.reward_coordinates,self.reward_coordinates]) #Add the reward state as joint state

        else:  #Agents can collide on any cell, but they can't move to an obstacle
            joint_states = [[[i,j],
                             [k,l]] for i in range(self.length) for j in range(self.width) 
                             for k in range(self.length) for l in range(self.width)
                             if  [i,j] not in self.obstacle_coordinates and [k,l] not in self.obstacle_coordinates
            ]

        return joint_states
 

    def identify_walls(self):
        '''
        Identify all impossible transitions due to the grid walls and the obstacles
        '''
        walls = []
        for i in range(self.length):
            for j in range(self.width):
                if [i,j] not in self.obstacle_coordinates:
                    fictious_player = Player(position = [i,j]) #Used to explore the grid in search of walls
                    if fictious_player.move('left')[0] not in range(self.length) or fictious_player.move('left')in self.obstacle_coordinates:
                        walls.append(['left',fictious_player.position])
                    if fictious_player.move('right')[0] not in range(self.length) or fictious_player.move('right') in self.obstacle_coordinates:
                        walls.append(['right',fictious_player.position])
                    if fictious_player.move('up')[1] not in range(self.width) or fictious_player.move('up') in self.obstacle_coordinates:
                        walls.append(['up',fictious_player.position])
                    if fictious_player.move('down')[1] not in range(self.width) or fictious_player.move('down') in self.obstacle_coordinates:
                        walls.append(['down',fictious_player.position])

        return walls
    

    def compute_reward(self,
                       old_state,
                       new_state,
                       movement,
                       collision_detected = False):
        '''
        Compute the reward obtained by a player for transitioning from its old state to its new state
        '''
        if  old_state == self.reward_coordinates:  #Stop receiving rewards once the goal is reached
            reward = 0
        elif new_state == self.reward_coordinates:  #The goal state is reached for the first time
            reward = self.reward_value
        elif new_state == old_state and movement != 'stay': #The player moved and bumped in a player or an obstacle
            reward = self.collision_penalty
        elif movement == 'stay' and collision_detected:  #The player stayed but was percuted by another player
            reward = self.collision_penalty       
        else: # The player made a regular valid movement
            reward = -1
        return reward

    def create_transition_table(self):
        '''
        Creates a dictionary where each pair of joint state and joint movement is mapped to a new resulting joint state
        '''
        recursivedict = lambda : defaultdict(recursivedict)
        transitions = recursivedict()
        joint_states = self.joint_states()
        walls = self.identify_walls()
        player0_movements =  self.players[0].movements
        player1_movements =  self.players[1].movements

        for state in joint_states:
            for m0 in player0_movements:
                for m1 in player1_movements:
                    if [m1,state[1]] in walls or state[1] == self.reward_coordinates:
                        if [m0,state[0]]  in walls or state[0] == self.reward_coordinates:
                            new_state = state
                        else : 
                            new_state = [Player(state[0]).move(m0),state[1]]                            
                    else:
                        if [m0,state[0]] in walls or state[0] == self.reward_coordinates:
                            new_state = [state[0],Player(state[1]).move(m1)]
                        else:
                            new_state = [Player(state[0]).move(m0),Player(state[1]).move(m1)]                   
                    if (new_state[0] == state[1]  and new_state[1] == state[0] ) or new_state not in joint_states: 
                        # There is a collision or a swap of positions
                         new_state = state #Return to previous state  
                    transitions[joint_states.index(state)][m0][m1] = joint_states.index(new_state)
                    
        return transitions

    def create_q_tables(self): 
        '''
        Creates the q tables which contains the Q-values used by the the nash Q Learning algorithm.
        The q tables are represented as 3-dimensional tensors and are initialized with null values. 
        '''
        player0_movements =  self.players[0].movements
        player1_movements =  self.players[1].movements
        joint_states = self.joint_states()
        q_tables0 = np.zeros((len(joint_states),
                              len(player0_movements)
                                    ))    
        q_tables1 = np.zeros((len(joint_states),
                              len(player1_movements)
                                    ))
        return q_tables0, q_tables1

  
class  QLearning:

    def __init__(self,
                 grid = Grid(),
                 learning_rate = 0.5,
                 max_iter = 100,
                 discount_factor = 0.7,
                 epsilon = 0.5,
                 random_state = 42):       

        self.grid = grid
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        random.seed(random_state)
        

    def fit(self,
            number_of_episodes,
            return_history = False): 
        """
        Fit the Nash Q Learning algorithm on the grid and return one Q table per player. 
        return_history (bool) : if True, print all the changing positions of the players on the grid during the learning cycle.
        """
        current_state = [self.grid.players[0].position, self.grid.players[1].position]
        player0_movements =  self.grid.players[0].movements
        player1_movements =  self.grid.players[1].movements
        joint_states = self.grid.joint_states()
        Q0, Q1 = self.grid.create_q_tables()
        transition_table = self.grid.create_transition_table()  
        state_tracker = [current_state]
        game_scores0 = []
        game_scores1 = []
        episode_counter = 0
        episode_rewards0 = []
        episode_rewards1 = []
        while episode_counter != number_of_episodes:
            if (current_state[0] == self.grid.reward_coordinates) and (current_state[1] == self.grid.reward_coordinates):  #Both players reached the reward, return to original position
                game_scores0.append(sum(episode_rewards0))
                game_scores1.append(sum(episode_rewards1))
                current_state = [self.grid.players[0].position, self.grid.players[1].position]
                episode_counter += 1
                episode_rewards0 = []
                episode_rewards1 = []

            # Agent 0
            if random.uniform(0,1) >= self.epsilon: #greedy
                q_values_0 = [Q0[self.grid.states.index(current_state)][player0_movements.index(action)] for action in player0_movements]
                max_q_values_0 = max(q_values_0)
                count_0 = q_values_0.count(max_q_values_0)
                if count_0 > 1:
                    best_actions_0 = [a for a in player0_movements if Q0[self.grid.states.index(current_state), player0_movements.index(a)] == max_q_values_0]
                    m0 = random.choice(best_actions_0)
                else:
                    best_actions_0 = [a for a in player0_movements if Q0[self.grid.states.index(current_state), player0_movements.index(a)] == max_q_values_0]
                    m0 = best_actions_0[0]
            else: #random
                m0 = random.choice(player0_movements)
            
            # Agent 1
            if random.uniform(0,1) >= self.epsilon: #greedy
                q_values_1 = [Q1[self.grid.states.index(current_state)][player1_movements.index(action)] for action in player1_movements]
                max_q_values_1 = max(q_values_1)
                count_1 = q_values_1.count(max_q_values_1)
                if count_1 > 1:
                    best_actions_1 = [a for a in player1_movements if Q1[self.grid.states.index(current_state), player1_movements.index(a)] == max_q_values_1]
                    m1 = random.choice(best_actions_1)
                else:
                    best_actions_1 = [a for a in player1_movements if Q1[self.grid.states.index(current_state), player1_movements.index(a)] == max_q_values_1]
                    m1 = best_actions_1[0]
            else: #random
                m1 = random.choice(player1_movements)

            new_state = joint_states[transition_table[joint_states.index(current_state)][m0][m1]]
            reward_0 = self.grid.compute_reward(current_state[0], new_state[0], m0)
            reward_1 = self.grid.compute_reward(current_state[1], new_state[1], m1)

            episode_rewards0.append(reward_0)
            episode_rewards1.append(reward_1)

            #Update state
            old_value_0 = Q0[(self.grid.states.index(current_state), player0_movements.index(m0))]                                          
            next_max_0 = max([Q0[(self.grid.states.index(new_state), player0_movements.index(a))] for a in player0_movements])                                  
            Q0[(self.grid.states.index(current_state), player0_movements.index(m0))] = (1 - self.learning_rate) * old_value_0 + self.learning_rate * (reward_0 + self.discount_factor * next_max_0)

            old_value_1 = Q1[(self.grid.states.index(current_state), player1_movements.index(m1))]                                          
            next_max_1 = max([Q1[(self.grid.states.index(new_state), player1_movements.index(a))] for a in player1_movements])                                  
            Q1[(self.grid.states.index(current_state), player0_movements.index(m1))] = (1 - self.learning_rate) * old_value_1 + self.learning_rate * (reward_1 + self.discount_factor * next_max_1)
        
            current_state = new_state
            state_tracker.append(current_state)   

        if return_history:
            print(state_tracker)
        return Q0, Q1, (sum(game_scores0)/number_of_episodes), (sum(game_scores1)/number_of_episodes)

    
    def get_best_policy(self, Q0, Q1):
        """
        Given two Q tables, one for each agent, return their best available path on the grid.
        """
        current_state = [self.grid.players[0].position,self.grid.players[1].position]
        joint_states = self.grid.joint_states()
        transition_table = self.grid.create_transition_table()
        player0_movements =  self.grid.players[0].movements
        player1_movements =  self.grid.players[1].movements
        policy0 = []
        policy1 = []
        while current_state != [self.grid.reward_coordinates, self.grid.reward_coordinates]: # while the reward state is not reached for both agents
            m0 = 'stay'
            m1 = 'stay'

            q_values_0 = [Q0[self.grid.states.index(current_state)][player0_movements.index(action)] for action in player0_movements]
            max_q_values_0 = max(q_values_0)
            count_0 = q_values_0.count(max_q_values_0)
            if count_0 > 1:
                best_actions_0 = [a for a in player0_movements if Q0[self.grid.states.index(current_state), player0_movements.index(a)] == max_q_values_0]
                m0 = random.choice(best_actions_0)
            else:
                best_actions_0 = [a for a in player0_movements if Q0[self.grid.states.index(current_state), player0_movements.index(a)] == max_q_values_0]
                m0 = best_actions_0[0]

            q_values_1 = [Q1[self.grid.states.index(current_state)][player1_movements.index(action)] for action in player1_movements]
            max_q_values_1 = max(q_values_1)
            count_1 = q_values_1.count(max_q_values_1)
            if count_1 > 1:
                best_actions_1 = [a for a in player1_movements if Q1[self.grid.states.index(current_state), player1_movements.index(a)] == max_q_values_1]
                m1 = random.choice(best_actions_1)
            else:
                best_actions_1 = [a for a in player1_movements if Q1[self.grid.states.index(current_state), player1_movements.index(a)] == max_q_values_1]
                m1 = best_actions_1[0]

            if current_state[0] != self.grid.reward_coordinates:
                policy0.append(m0)
            else : #target reached for player 0
                policy0.append('stay')
            if current_state[1] != self.grid.reward_coordinates:
                policy1.append(m1)
            else: #target reached for player 1
                policy1.append('stay')
            if current_state != joint_states[transition_table[joint_states.index(current_state)][m0][m1]]: #there was a movement
                current_state = joint_states[transition_table[joint_states.index(current_state)][m0][m1]]
            else :  #No movement, the model did not converge
                policy0 = 'model failed to converge to a policy'
                policy1 = 'model failed to converge to a policy'
                break            

        return policy0, policy1