import matplotlib.pyplot as plt
import Q_Learning3x3 as q
import NashQLearning3x3 as nashq
import warnings
warnings.filterwarnings('ignore')
import time

# ? Grid parameters
grid_length = 3
grid_width = 3
obstacles_coord = []
reward_coord = [1, 2]
reward_val = 100
collision_val = -10
A_start_coord = [0, 0]
B_start_coord = [2, 0]

# ? Which algorithms to run
run_q = 1
run_nashq = 0

# ? Wheter or not to print the best policy
print_policy = 1

# ? Number or iterations
start = 1
stop = 7000
step = 100

'''
Basic Q-learning
'''

if run_q == 1:
    start_q = time.perf_counter()
    num_of_episodes = []
    rewards0 = []
    rewards1 = []
    final_qtable_0 = None
    final_qtable_1 = None
    for ep in range(start, stop, step):
        q_player1 = q.Player(A_start_coord)
        q_player2 = q.Player(B_start_coord)

        q_grid = q.Grid(length = grid_length,
                    width = grid_width,
                    players = [q_player1, q_player2],
                    obstacle_coordinates = obstacles_coord, 
                    reward_coordinates = reward_coord,
                    reward_value = reward_val,
                    collision_penalty = collision_val)

        q_alg = q.QLearning(q_grid, max_iter = 200, discount_factor = 0.7, learning_rate = 0.7, epsilon = 0.01)
        Q0, Q1, Q0_rewards, Q1_rewards = q_alg.fit(ep, return_history = False)
        rewards0.append(Q0_rewards)
        rewards1.append(Q1_rewards)
        num_of_episodes.append(ep)
        final_qtable_0 = Q0
        final_qtable_1 = Q1
    if print_policy == 1:     
        p0, p1 = q_alg.get_best_policy(final_qtable_0, final_qtable_1)
        stop_q = time.perf_counter()


'''
Nash Q-learning
'''

if run_nashq == 1:
    start_nashq = time.perf_counter()
    num_of_episodes_nash = []
    rewardsNashq0 = []
    rewardsNashq1 = []
    final_nashqtable_0 = []
    final_nashqtable_1 = []
    for ep in range(start, stop, step):
        nashq_player1 = nashq.Player(A_start_coord)
        nashq_player2 = nashq.Player(B_start_coord)

        nashq_grid = nashq.Grid(length = grid_length,
                    width = grid_width,
                    players = [nashq_player1, nashq_player2],
                    obstacle_coordinates = obstacles_coord, #A single obstacle in the middle of the grid
                    reward_coordinates = reward_coord,
                    reward_value = reward_val,
                    collision_penalty = collision_val)

        nashq_alg = nashq.NashQLearning(nashq_grid, max_iter = 3000, discount_factor = 0.7, learning_rate = 0.7, epsilon = 0.10, decision_strategy = 'epsilon-greedy')
        NashQ0, NashQ1, NashQ0_rewards, NashQ1_rewards = nashq_alg.fit(ep, return_history = False)
        rewardsNashq0.append(NashQ0_rewards)
        rewardsNashq1.append(NashQ1_rewards)
        num_of_episodes_nash.append(ep)
        final_nashqtable_0 = NashQ0
        final_nashqtable_1 = NashQ1

    if print_policy == 1:
        Nashp0, Nashp1 = nashq_alg.get_best_policy(final_nashqtable_0, final_nashqtable_1)
        stop_nashq = time.perf_counter()


'''
Visualisation section
'''

if print_policy == 1:
    if run_q == 1:
        print('--- Optimal policy for the Q-learning algorithm ---')
        print('Player 0 follows the  policy : %s of length %s' %('-'.join(p0),len(p0)))
        print('Player 1 follows the  policy : %s of length %s' %('-'.join(p1),len(p1)))
        print(f"Total running time is {stop_q-start_q:0.4f}")
    if run_nashq == 1:
        print('--- Optimal policy for the Nash Q-learning algorithm ---')
        print('Player 0 follows the  policy : %s of length %s' %('-'.join(Nashp0),len(Nashp0)))
        print('Player 1 follows the  policy : %s of length %s' %('-'.join(Nashp1),len(Nashp1)))
        print(f"Total running time is {stop_nashq-start_nashq:0.4f}")

if run_q == 1:
    fig, ax1 = plt.subplots(1, 1)
    plt.rc('font', size=14) 
    ax1.plot(num_of_episodes, rewards0, color='black', linestyle='solid', label="Agent 1")
    ax1.plot(num_of_episodes, rewards1, color='black', linestyle='dashed', label="Agent 2")
    ax1.legend()
    ax1.set_title("Q-learning")
    ax1.set(xlabel="Iterations", ylabel="Average Reward")

    plt.show()

if run_nashq == 1:
    fig, ax1 = plt.subplots(1, 1)
    plt.rc('font', size=14) 
    ax1.plot(num_of_episodes, NashQ0_rewards, color='black', linestyle='solid', label="Agent 1")
    ax1.plot(num_of_episodes, NashQ1_rewards, color='black', linestyle='dashed', label="Agent 2")
    ax1.legend()
    ax1.set_title("Nash Q-learning")
    ax1.set(xlabel="Iterations", ylabel="Average Reward")

    plt.show()