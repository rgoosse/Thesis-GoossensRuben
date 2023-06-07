import random
import matplotlib.pyplot as plt 
import numpy as np
import math
import AdaptedBeerDistributionGame as BG

num_of_weeks = BG.weeks

# n can only be integers in the range [-4, 4]
n_range = [i for i in range(-4, 5)]        

# inventory range is within [-12, 12]
inv_range = [i for i in range (-BG.inventory_limit, BG.inventory_limit+1)]

# price setting range is constrained to 1 to start_price €
price_range = [i for i in range(2, BG.price_limit+1)]

# Q-learning parameters
alpha = 0.9                     # Learning rate
gamma = 0.75                    # Discount factor
epsilon = 0.10                  # Epsilon-greedy parameter

# Form all action-tuples
action_pairs = [(order, price) for order in n_range for price in price_range]

# Function that initializes the Q-table
def q_table_maker(states):
    q_table = {}
    for state in states:
        for pair in action_pairs:
            q_table[(state, pair)] = -100000.0
    return q_table

class QAgent:
    def __init__(self, q_table):
        self.action_pairs = action_pairs
        self.q_table = q_table
        self.player = BG.QPlayer()
        self.player_state = self.player.inventory_offset[0] + self.player.supply_orders[0]

    def select_action(self):
        if random.uniform(0, 1) < epsilon:
            self.action = random.choice(self.action_pairs)
        else:
            q_values = [self.q_table[(self.player_state, pair)] for pair 
                        in self.action_pairs]
            max_q_value = max(q_values)
            count = q_values.count(max_q_value)
            if count > 1:
                best_actions = [pair for pair in self.action_pairs 
                                if self.q_table[(self.player_state, pair)] == max_q_value]
                self.action = random.choice(best_actions)
            else:
                best_action = [pair for pair in self.action_pairs 
                               if self.q_table[(self.player_state, pair)] == max_q_value]
                self.action = best_action[0]

    def execute_action(self, week):
        self.player.play_round(week, BG.demand[week])
        self.player.order_supply(week, BG.demand[week], self.action)
        self.next_state = (self.player.inventory_offset[week+1] 
                           + self.player.supply_orders[week+1])
        self.player.cost(week, self.action[1])
        self.reward = self.player.costs[week+1]
    
    def execute_action_no_order(self, week):
        self.player.play_round(week, 0)
        self.player.order_supply(week, 0, (0,0))
        self.next_state = (self.player.inventory_offset[week+1] 
                           + self.player.supply_orders[week+1])
        self.player.cost(week, self.action[1])
        self.reward = self.player.costs[week+1]   
    
    def execute_half_order(self, week):
        self.player.play_round(week, BG.demand[week]/2)
        self.player.order_supply(week, BG.demand[week]/2, self.action)
        self.next_state = (self.player.inventory_offset[week+1] 
                           + self.player.supply_orders[week+1])
        self.player.cost(week, self.action[1])
        self.reward = self.player.costs[week+1]

    def update_q_table(self):
        if self.next_state  >= -BG.inventory_limit and self.next_state <= BG.inventory_limit:
            old_value = self.q_table[(self.player_state, self.action)]                                          
            next_max = max([self.q_table[(self.next_state, pair)] for pair in 
                            self.action_pairs])                           
            new_value = (1 - alpha) * old_value + alpha * (self.reward + gamma * next_max)           
            self.q_table[(self.player_state, self.action)] = new_value
        else:
            old_value = self.q_table[(self.player_state, self.action)]       
            # We give a large penalty for not respecting the inventory constraints                                   
            next_max = -10**9                      
            new_value = (1 - alpha) * old_value + alpha * (self.reward + gamma * next_max)           
            self.q_table[(self.player_state, self.action)] = new_value


# Define the Q-learning algorithm
def q_learning(states, num_episodes):
    q_table_1 = q_table_maker(states)
    total_cost_per_episode_1 = []

    q_table_2 = q_table_maker(states)
    total_cost_per_episode_2 = []

    # Multiple episodes per simulation
    for episode in range(num_episodes):
        p1_total_rewards = [-BG.start_inv]
        p2_total_rewards = [-BG.start_inv]
        # Loop over weeks
        for i in range(num_of_weeks-1):
            # Starting states of the two players
            if i == 0:
                # Define the players
                player1 = QAgent(q_table_1)
                player2 = QAgent(q_table_2)
                player1.player_state = (player1.player.inventory_offset[i] 
                                        + player1.player.supply_orders[i])
                player2.player_state = (player2.player.inventory_offset[i] 
                                        + player2.player.supply_orders[i])

            if i != 0:
                player1.player_state = (player1.player.inventory_offset[i] 
                                        + player1.player.supply_orders[i])
                player2.player_state = (player2.player.inventory_offset[i] 
                                        + player2.player.supply_orders[i])

            player1.select_action()
            player1_action = player1.action 

            player2.select_action()
            player2_action = player2.action

            # Select the winner
            # Buyers opts for supplier with lowest price, regardless if the supplier is unable to deliver the full order
            if player1_action[1] > player2_action[1]:
                # Player 1 has a demand of 0
                player1.execute_action_no_order(i)
                player1.update_q_table()
                # Player 2 is selected and excutes the order
                player2.execute_action(i)
                player2.update_q_table()

            if player1_action[1] < player2_action[1]:
                # Player 1 is selected and excutes the order
                player1.execute_action(i)
                player1.update_q_table()
                # Player 2 has a demand of 0
                player2.execute_action_no_order(i)
                player2.update_q_table()

            # If same price, we split demand
            if player1_action[1] == player2_action[1]:
                player1.execute_half_order(i)
                player1.update_q_table()

                player2.execute_half_order(i)
                player2.update_q_table()

            # Add the total rewards of this episode to the global total rewards
            p1_total_rewards.append(player1.reward)
            p2_total_rewards.append(player2.reward)

        total_cost_per_episode_1.append(sum(p1_total_rewards))
        total_cost_per_episode_2.append(sum(p2_total_rewards))

    avg_cost_per_episode_1 = sum(total_cost_per_episode_1)/(episode+1)
    avg_cost_per_episode_2 = sum(total_cost_per_episode_2)/(episode+1)

    # Return the avg cost over the number of iterations
    return avg_cost_per_episode_1, q_table_1, avg_cost_per_episode_2, q_table_2

# Simulation of multiple episodes
agent_1_rewards = []
agent_2_rewards = []
num_of_episodes = []
num_of_unexploredstates1 = []
num_of_unexploredstates2 = []
final_q_table1 = None
final_q_table2 = None

# Simulation parameters
start_episode = 10**6
end_episode = 1+2*10**6
step = 10**6

# Simulate and get the final q-table
for i in range(start_episode, end_episode+1, step):
    cost1, q_table1, cost2, q_table2  = q_learning(inv_range, i)
    agent_1_rewards.append(cost1)
    agent_2_rewards.append(cost2)
    num_of_unexploredstates1.append(sum(value == -100000.0 for value in q_table1.values())/len(q_table1))
    num_of_unexploredstates2.append(sum(value == -100000.0 for value in q_table2.values())/len(q_table2))
    num_of_episodes.append(i)
    if i == math.trunc((end_episode-start_episode)/step)*step+start_episode:
        final_q_table1 = q_table1
        final_q_table2 = q_table2

# Evaluate the final Q-table 
def evaluate(q_table1, q_table2):
    for i in range(num_of_weeks-1):
        if i == 0:
            player1 = BG.QPlayer()  
            player2 = BG.QPlayer()  
            player1_state = player1.inventory_offset[i] + player1.supply_orders[i]
            player2_state = player2.inventory_offset[i] + player2.supply_orders[i]
        if i != 0:
            player1_state = player1.inventory_offset[i] + player1.supply_orders[i]
            player2_state = player2.inventory_offset[i] + player2.supply_orders[i]
        
        q_values_1 = [q_table1[(player1_state, pair)] for pair in action_pairs]
        q_values_2 = [q_table2[(player2_state, pair)] for pair in action_pairs]
        max_q_value_1 = max(q_values_1)
        max_q_value_2 = max(q_values_2)
        count_1 = q_values_1.count(max_q_value_1)
        count_2 = q_values_2.count(max_q_value_2)
        if count_1 > 1:
            best_actions_1 = [pair for pair in action_pairs if q_table1[(player1_state, pair)] == max_q_value_1]
            action_1 = random.choice(best_actions_1)
        else:
            best_action1 = [pair for pair in action_pairs if q_table1[(player1_state, pair)] == max_q_value_1]
            action_1 = best_action1[0]
        if count_2 > 1:
            best_actions_2 = [pair for pair in action_pairs if q_table2[(player2_state, pair)] == max_q_value_2]
            action_2 = random.choice(best_actions_2)
        else:
            best_action2 = [pair for pair in action_pairs if q_table2[(player2_state, pair)] == max_q_value_2]
            action_2 = best_action2[0]

        player1_action = action_1
        player2_action = action_2

        if player1_action[1] > player2_action[1]:
            player1.play_round(i, 0)
            player1.order_supply(i, 0, (0,0))
            player1.cost(i, player1_action[1])

            player2.play_round(i, BG.demand[i])
            player2.order_supply(i, BG.demand[i], player2_action)
            player2.cost(i, player2_action[1])

        if player1_action[1] < player2_action[1]:
            player1.play_round(i, BG.demand[i])
            player1.order_supply(i, BG.demand[i], player1_action)
            player1.cost(i, player1_action[1])

            player2.play_round(i, 0)
            player2.order_supply(i, 0, (0,0))
            player2.cost(i, player2_action[1])

        if player1_action[1] == player2_action[1]:
            player1.play_round(i, BG.demand[i]/2)
            player1.order_supply(i, BG.demand[i]/2, player1_action)
            player1.cost(i, player1_action[1])

            player2.play_round(i, BG.demand[i]/2)
            player2.order_supply(i, BG.demand[i]/2, player2_action)
            player2.cost(i, player2_action[1])            

    return player1.inventory_offset, player1.supply_orders, player1.costs, player1.price_set, player1.sold_goods, player2.inventory_offset, player2.supply_orders, player2.costs, player2.price_set, player2.sold_goods

io1, so1, c1, p1, sg1, io2, so2, c2, p2, sg2 = evaluate(final_q_table1, final_q_table2)
cc1 = [c1[0]]
cc2 = [c2[0]]
for i in range(len(c1)-1):
    cc1.append(cc1[i]+c1[i+1])
    cc2.append(cc2[i]+c2[i+1])

# Print extra information
print_info = 0
if print_info == 1:
    print("The product price of Player 1 is:", p1)
    print("The product price of Player 2 is:", p2)
    print("The amount of sold goods by Player 1 is:", sg1)
    print("The amount of sold goods by Player 2 is:", sg2)
    print("The inventory offset of Player 1 is:", io1)
    print("The inventory offset of Player 2 is:", io2)
    print("The profit of the optimal policy strategy for player 1 is:", sum(c1))
    print("The profit of the optimal policy strategy for player 2 is:", sum(c2))
    print("Percentage of unexplored state-action pairs of player 1:", round(100*num_of_unexploredstates1[-1], 2), "%")
    print("Percentage of unexplored state-action pairs of player 2:", round(100*num_of_unexploredstates2[-1], 2), "%")


# Global overiew of demand, order quantity, inventory and cost of the last iteration
plot_global = 1
if plot_global == 1:
    figure, axis = plt.subplots(2, 2)
    plt.rc('font', size=14) 

    axis[0, 0].plot([i for i in range(BG.weeks)], p1, color='black')
    axis[0, 0].set_title("Product Price")
    axis[0, 0].set(xlabel="Week", ylabel="Euro")

    axis[0, 1].plot([i for i in range(BG.weeks)], so1, color='black')
    axis[0, 1].set_title("Order Amount")
    axis[0, 1].set(xlabel="Week", ylabel="Units")

    axis[1, 0].plot([i for i in range(BG.weeks)], io1, color='black')
    axis[1, 0].set_title("Inventory")
    axis[1, 0].set(xlabel="Week", ylabel="Units")

    axis[1, 1].plot([i for i in range(BG.weeks)], cc1, color='black')
    axis[1, 1].set_title("Profit")
    axis[1, 1].set(xlabel="Week", ylabel="Units")

    plt.show()

plot_average_profit = 0
if plot_average_profit == 1:
    fig, ax = plt.subplots(1, 1)
    plt.rc('font', size=14) 

    ax.plot(num_of_episodes, agent_1_rewards, color='black', linestyle='solid', label="Agent 1")
    ax.plot(num_of_episodes, agent_2_rewards, color='black', linestyle='dashed', label="Agent 2")
    ax.legend()
    ax.set_title("Average Profit")
    ax.set(xlabel="Number of Episodes", ylabel="Euro")

    plt.show()

plot_strategy = 0
if plot_strategy == 1:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.rc('font', size=14)

    ax1.plot([i for i in range(BG.weeks)], p1, color='black', linestyle='solid', label="Agent 1")
    ax1.legend()
    ax1.set_title("Product Price")
    ax1.set(xlabel="Week", ylabel="Euro")


    ax2.plot([i for i in range(BG.weeks)], p2, color='black', linestyle='dashed', label="Agent 2")
    ax2.legend()
    ax2.set_title("Product Price")
    ax2.set(xlabel="Week", ylabel="Euro")

    plt.show()


# Plot the percentage of unexplored states
plot_unexploredstates = 0
if plot_unexploredstates == 1:
    plt.rc('font', size=14) 
    plt.plot(num_of_episodes, [100*i for i in num_of_unexploredstates1], label="Player 1", color='black', linestyle='solid')
    plt.plot(num_of_episodes, [100*i for i in num_of_unexploredstates2], label="Player 2", color='black', linestyle='dashed')
    plt.legend()
    plt.title("Percentage of Unexplored State-Action Pairs")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Unexplored State-Action Pairs [%]")
    plt.show()

plot_price_inv = 0
if plot_price_inv == 1:
    plt.rc('font', size=14) 
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot([i for i in range(BG.weeks)], p1, color='black', linestyle='solid')
    ax1.set_title("Product Price")
    ax1.set(xlabel="Week", ylabel="Euro")

    ax2.plot([i for i in range(BG.weeks)], io1, color='black', linestyle='solid')
    ax2.set_title("Inventory")
    ax2.set(xlabel="Week", ylabel="Units")

    plt.show()

plot_demand_vs_order = 0
if plot_demand_vs_order == 1:
    plt.rc('font', size=14) 
    plt.plot([i for i in range(BG.weeks)], [io1[i]+so1[i] for i in range(len(so1))], label="On-hand Units", color='black', linestyle='solid')
    plt.legend()
    plt.title("Weekly amount of on-hand units")
    plt.xlabel("Weeks")
    plt.ylabel("Units")
    plt.show()