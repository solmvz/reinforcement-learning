import copy
import math
import random

class Struct:
    def __init__(self):
        # for cell obstacles
        self.N = -1
        self.E = -1
        self.S = -1
        self.W = -1
        # Reward - Value - Policy
        self.R = 1
        self.V = 0
        self.P = 0  # (1:North, 2:East, 3:South, 4:West)
        # for cell number
        self.number = -1
        # for cell value
        # (value can be anything to show the chosen path)
        self.value = "."
        # policy iteration variables
        self.successor_direction = 0
        self.successor_value = 0

    def set_N(self, value):
        self.N = value
    def set_E(self, value):
        self.E = value
    def set_S(self, value):
        self.S = value
    def set_W(self, value):
        self.W = value
    def set_R(self, value):
        self.R = value
    def set_V(self, value):
        self.V = value
    def set_P(self, value):
        self.P = value
    def set_number(self, value):
        self.number = value
    def set_value(self, value):
        self.value = value
    def set_successor_direction(self, value):
        self.successor_direction = value
    def set_successor_value(self, value):
        self.successor_value = value

    def get_N(self):
        return self.N
    def get_E(self):
        return self.E
    def get_S(self):
        return self.S
    def get_W(self):
        return self.W
    def get_R(self):
        return self.R
    def get_V(self):
        return self.V
    def get_P(self):
        return self.P
    def get_number(self):
        return self.number
    def get_value(self):
        return self.value
    def get_successor_direction(self):
        return self.successor_direction
    def get_successor_value(self):
        return self.successor_value

    def initialize_p(self):
        # function is used for policy iteration
        # at first we choose a random action based on the possible moves of each cell
        # which makes the first initialization better
        random_p = 0
        # N
        if self.N == 0 and self.E == 1 and self.S == 1 and self.W == 1:
            random_p = 3
        # E
        if self.N == 1 and self.E == 0 and self.S == 1 and self.W == 1:
            random_p = 4
        # S
        if self.N == 1 and self.E == 1 and self.S == 0 and self.W == 1:
            random_p = 2
        # W
        if self.N == 1 and self.E == 1 and self.S == 1 and self.W == 0:
            random_p = 3
        # N,E
        if self.N == 0 and self.E == 0 and self.S == 1 and self.W == 1:
            random_p = 3
        # N,S
        if self.N == 0 and self.E == 1 and self.S == 0 and self.W == 1:
            random_p = 2
        # N,W
        if self.N == 0 and self.E == 1 and self.S == 1 and self.W == 0:
            random_p = 2
        # E,W
        if self.N == 1 and self.E == 0 and self.S == 1 and self.W == 0:
            random_p = 3
        # E,S
        if self.N == 1 and self.E == 0 and self.S == 0 and self.W == 1:
            random_p = 1
        # S,W
        if self.N == 1 and self.E == 1 and self.S == 0 and self.W == 0:
            random_p = 1
        # N,E,S
        if self.N == 0 and self.E == 0 and self.S == 0 and self.W == 1:
            random_p = 4
        # N,W,S
        if self.N == 0 and self.E == 1 and self.S == 0 and self.W == 0:
            random_p = 2
        # E,S,W
        if self.N == 1 and self.E == 0 and self.S == 0 and self.W == 0:
            random_p = 1
        # E,N,W
        if self.N == 0 and self.E == 0 and self.S == 1 and self.W == 0:
            random_p = 3
        # N,E,W,S
        if self.N == 1 and self.E == 1 and self.S == 1 and self.W == 1:
            random_p = 4
        if self.N == 0 and self.E == 0 and self.S == 0 and self.W == 0:
            random_p = 0
        self.set_P(random_p)
        return
class State:
    def __init__(self):
        self.board = [[]]
        self.dimension = -1
        self.start_i = -1
        self.goal_i = -1
        self.start_j = -1
        self.goal_j = -1

        iterator = 0  # not important
        with open("sample_25.txt") as file:  # opening file to read
            lines = file.readlines()  # an array of all lines separeted from each other
            for i in lines:  # iterate lines one by one
                # -----------------------------
                if i == lines[0]:  # first line
                    array = []
                    for string in i.split():  # adds each specific word in a line into the array
                        array.append(string)
                    self.dimension = int(array[0])  # setting the dimension
                    # building a 2D array of structs
                    self.board = [[Struct() for j in range(self.dimension)] for i in range(self.dimension)]
                    # initializing the cells' numbers
                    counter = 1
                    for k in range(self.dimension):
                        for j in range(self.dimension):
                            self.board[j][k].set_number(counter)
                            counter += 1
                    iterator += 1
                # -----------------------------
                elif i == lines[(self.dimension * self.dimension) + 1]:  # last line = lines[26]
                    array = []
                    for string in i.split():  # same as above part
                        array.append(string)
                    self.start = int(array[0])
                    self.goal = int(array[1])
                    # changing the values of start and goal from default "." to "S" and "G" to show them on board
                    for k in range(self.dimension):
                        for j in range(self.dimension):
                            if self.board[k][j].get_number() == self.start:
                                self.board[k][j].set_value('S')
                                self.start_i = k
                                self.start_j = j
                            if self.board[k][j].get_number() == self.goal:
                                self.board[k][j].set_value('G')
                                self.board[k][j].set_R(100)
                                self.goal_i = k
                                self.goal_j = j
                    iterator += 1
                # -----------------------------
                else:  # other lines = lines[1] to lines[25]
                    # setting the obstacles of each cell
                    for k in range(self.dimension):
                        for j in range(self.dimension):
                            if self.board[k][j].get_number() == iterator:
                                self.board[k][j].set_N(int(lines[iterator][0]))
                                self.board[k][j].set_E(int(lines[iterator][2]))
                                self.board[k][j].set_S(int(lines[iterator][4]))
                                self.board[k][j].set_W(int(lines[iterator][6]))
                    iterator += 1

    def get_dimension(self):
        return self.dimension
    def get_start(self):
        return self.start_i, self.start_j
    def get_goal(self):
        return self.goal_i, self.goal_j

    def print_board(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                print(self.board[i][j].get_value(), end=" ")
            print()
    def print_board_val(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                print(self.board[i][j].get_V(), end=" ")
            print()
    def print_obstacles(self):
        for j in range(self.dimension):
            for i in range(self.dimension):
                print("lines[", format(self.board[i][j].get_number(), "2d"), "]: N =",
                      format(self.board[i][j].get_N(), "2d"),
                      " E =", format(self.board[i][j].get_E(), "2d"),
                      " S =", format(self.board[i][j].get_S(), "2d"),
                      " W =", format(self.board[i][j].get_W(), "2d"))

    def switch(self, x1, y1, x2, y2):
        if self.board[x2][y2].get_value() == 'G':
            self.board[x1][y1].set_value('#')
            self.board[x2][y2].set_value('F')
            return
        temp = self.board[x1][y1].get_value()
        self.board[x1][y1].set_value("#")
        self.board[x2][y2].set_value(temp)
        return

# value iteration
def is_goal(current, i, j):
    if current.board[i][j].get_value() == 'G':
        return True
    return False
def set_actions(current, i, j):
    actions = []
    if current.board[i][j].get_E() != 0 and j + 1 < current.dimension:
        actions.append('E')
    if current.board[i][j].get_W() != 0 and j - 1 >= 0:
        actions.append('W')
    if current.board[i][j].get_N() != 0 and i - 1 >= 0:
        actions.append('N')
    if current.board[i][j].get_S() != 0 and i + 1 < current.dimension:
        actions.append('S')
    return actions
def do_action(current, action, i, j):
    if is_goal(current, i, j):
        return i, j
    if action == 'E':
        return i, j + 1
    if action == 'W':
        return i, j - 1
    if action == 'N':
        return i - 1, j
    if action == 'S':
        return i + 1, j
def find_agent(current):
    for i in range(current.dimension):
        for j in range(current.dimension):
            if current.board[i][j].get_value() == 'S':
                return i, j
def end_game(current):
    for i in range(current.dimension):
        for j in range(current.dimension):
            if current.board[i][j].get_value() == 'F':
                return True
    return False
def max_v(current, i, j, actions, step):
    gamma = 0.7
    reward = 1
    action_values = []
    if is_goal(current, i, j):
        return 1000
    if len(actions) == 0:
        return 0
    for a in actions:
        i2, j2 = do_action(current, a, i, j)
        if is_goal(current, i2, j2):
            reward = 100
        action_values.append(reward + gamma * float(current.board[i2][j2].get_V()))
    v = max(action_values)
    return v
def value_iteration(current):
    epsilon = 0.001
    k = 0
    while True:
        # for stopping condition
        delta = 0
        # loop over state space
        for i in range(current.dimension):
            for j in range(current.dimension):
                actions = set_actions(current, i, j)
                best_action_value = max_v(current, i, j, actions, 0)
                if best_action_value > current.board[i][j].get_V():
                    delta = max(delta, abs(best_action_value - current.board[i][j].get_V()))
                    current.board[i][j].set_V(best_action_value)
        if delta < epsilon:
            break
    return

# policy iteration
def successor(current, i, j):
    agent = current.board[i][j]
    next_array = []
    x = i
    y = j
    if agent.get_N() == 1 and x - 1 >= 0:
        temp = (1, x, y, x - 1, y)  # 1: direction, (x, y): first position, (x, y - 1): position after move
        next_array.append(temp)
    if agent.get_E() == 1 and y + 1 < current.dimension:
        temp = (2, x, y, x, y + 1)
        next_array.append(temp)
    if agent.get_S() == 1 and x + 1 < current.dimension:
        temp = (3, x, y, x + 1, y)
        next_array.append(temp)
    if agent.get_W() == 1 and y - 1 >= 0:
        temp = (4, x, y, x, y - 1)
        next_array.append(temp)
    return next_array
def initialize_v_to_zero(current):
    for i in range(current.get_dimension()):
        for j in range(current.get_dimension()):
            current.board[i][j].set_V(0)
    return
def initialize_p_randomly(current):
    for i in range(current.get_dimension()):
        for j in range(current.get_dimension()):
            current.board[i][j].initialize_p()
    return
def convergence_v(state1, state2):
    flag = True
    for i in range(state1.get_dimension()):
        for j in range(state1.get_dimension()):
            if abs(state1.board[i][j].get_V() - state2.board[i][j].get_V()) >= 0.0000000000000001:
                flag = False
    if flag:
        return True
    else:
        return False
def argmax(actions):
    maximum = -99
    direction = 0
    for a in actions:
        if a.get_successor_value() >= maximum:
            maximum = a.get_successor_value()
            direction = a.get_successor_direction()
    return direction
old_v = State()
initialize_v_to_zero(old_v)
def evaluate_the_current_policy(current, gamma):
    global old_v
    max_iteration = 700000
    x_g, y_g = current.get_goal()
    for k in range(max_iteration):
        new_v = copy.deepcopy(current)  # saving the new Vs
        # in each iteration we have to clean everything and again initialize the new Vs to zero and then continue iteration
        initialize_v_to_zero(new_v)
        # for each cell
        for i in range(current.get_dimension()):
            for j in range(current.get_dimension()):
                if i == x_g and j == y_g:  # absurd state
                    v = 0
                    actions = successor(current, i, j)
                    # for each possible action of the cell
                    for a in actions:
                        if a[0] == current.board[i][j].get_P():
                            # k[1]k[2]: coordinates of the destination cell with all actions from goal lead to goal itself
                            v += (100 + (gamma * old_v.board[a[1]][a[2]].get_V()))
                    # updating the new V
                    new_v.board[i][j].set_V(v)
                else:
                    v = 0
                    actions = successor(current, i, j)
                    # for each possible action of the cell
                    for a in actions:
                        if a[0] == current.board[i][j].get_P():
                            v += (1 + (gamma * old_v.board[a[3]][a[4]].get_V()))
                    # updating the new V
                    new_v.board[i][j].set_V(v)
        if convergence_v(old_v, new_v):
            break
        else:
            # after finishing the first iteration we have to save the new Vs as old ones, to prepare for second iteration and so on..
            old_v = copy.deepcopy(new_v)
    return old_v
def one_step_lookahead(current, i, j, gamma):
    valued_actions = []
    actions = successor(current, i, j)
    x_g, y_g = current.get_goal()
    for a in actions:
        if i == x_g and j == y_g:  # absurd state
            v = (100 + (gamma * current.board[a[1]][a[2]].get_V()))
        else:
            v = (1 + (gamma * current.board[a[3]][a[4]].get_V()))
        # setting the value of action
        current.board[a[3]][a[4]].set_successor_value(v)
        current.board[a[3]][a[4]].set_successor_direction(a[0])
        valued_actions.append(current.board[a[3]][a[4]])
    return valued_actions
def policy_iteration(current, gamma=0.5):
    global old_v
    max_iteration = 700000
    initialize_p_randomly(current)
    for k in range(max_iteration):
        # evaluation
        current = evaluate_the_current_policy(current, gamma)
        policy_stable = True
        # for each cell
        for i in range(current.get_dimension()):
            for j in range(current.get_dimension()):
                actions_values = one_step_lookahead(current, i, j, gamma)
                best_action = argmax(actions_values)
                if best_action != current.board[i][j].get_P():
                    policy_stable = False

                current.board[i][j].set_P(best_action)

                if best_action != current.board[i][j].get_P():
                    policy_stable = False

        if policy_stable:
            return current

# Q learning
def walk_Q(current, action, x, y):
    if is_goal(current, x, y):
        return current, x, y
    if action == 'N':
        new_x = x - 1
        new_y = y
        return current, new_x, new_y
    if action == 'E':
        new_x = x
        new_y = y + 1
        return current, new_x, new_y
    if action == 'S':
        new_x = x + 1
        new_y = y
        return current, new_x, new_y
    if action == 'W':
        new_x = x
        new_y = y - 1
        return current, new_x, new_y
def schedule(T, step):
    T_0 = 1.0
    alpha = 0.3
    T = math.exp(-alpha * step) * T_0
    step += 1
    return T, step
# gamma 0.3 for 5*5 and 0.5 for 25*25
def choose_action(current, actions, i, j, T, gamma=0.5, reward=1, max_A='NULL', max_V=0):
    for a in actions:
        i2, j2 = do_action(current, a, i, j)
        if is_goal(current, i2, j2):
            reward = 100
        new_v = reward + gamma * float(current.board[i2][j2].get_V())
        if new_v > current.board[i2][j2].get_V():
            current.board[i2][j2].set_V(new_v)
        if new_v > max_V:
            max_A = a
            max_V = new_v
    r = random.uniform(0, 1)
    rand_A = random.choice(actions)
    i_rand, j_rand = do_action(current, rand_A, i, j)
    if r < T:
        return rand_A
    else:
        return max_A
def q_iteration(current, i, j, T):
    trained = False
    action_values = []
    while True:
        actions = set_actions(current, i, j)
        if is_goal(current, i, j):
            trained = True
            break
        if len(actions) == 0:
            break
        max_A = choose_action(current, actions, i, j, T)
        action_values.append(max_A)
        current, i, j = walk_Q(current, max_A, i, j)
    return trained, current, max(action_values)
def q_learning(current):
    x = 0
    i, j = find_agent(current)
    trained = False
    while x < 100:
        T = 1.0
        k = 0
        while not trained:
            T, k = schedule(T, k)
            trained, current, best_action_value = q_iteration(current, i, j, T)
        x += 1
    return current

#testing
def walk_2(current, action, x, y):
    if is_goal(current, x, y):
        return current, x, y
    if action == 'N':
        new_x = x - 1
        new_y = y
        current.switch(x, y, new_x, new_y)
        return current, new_x, new_y
    if action == 'E':
        new_x = x
        new_y = y + 1
        current.switch(x, y, new_x, new_y)
        return current, new_x, new_y
    if action == 'S':
        new_x = x + 1
        new_y = y
        current.switch(x, y, new_x, new_y)
        return current, new_x, new_y
    if action == 'W':
        new_x = x
        new_y = y - 1
        current.switch(x, y, new_x, new_y)
        return current, new_x, new_y
def play_game(current):
    i, j = find_agent(current)
    max_V = 0
    while True:
        max_A = 'NULL'
        actions = set_actions(current, i, j)
        if end_game(current) or len(actions) == 0:
            break
        for a in actions:
            i2, j2 = do_action(current, a, i, j)
            new_v = current.board[i2][j2].get_V()
            if new_v > max_V:
                max_A = a
                max_V = new_v
            if max_A == 'NULL':
                max_A = random.choice(actions)
        current, i, j = walk_2(current, max_A, i, j)
    return current
def walk(current, direction, old_x, old_y):
    if direction == 1:  # north
        new_x = old_x - 1
        new_y = old_y
        current.board[new_x][new_y].set_value('#')
        return new_x, new_y, current
    elif direction == 2:  # east
        new_x = old_x
        new_y = old_y + 1
        current.board[new_x][new_y].set_value('#')
        return new_x, new_y, current
    elif direction == 3:  # south
        new_x = old_x + 1
        new_y = old_y
        current.board[new_x][new_y].set_value('#')
        return new_x, new_y, current
    else:  # west
        new_x = old_x
        new_y = old_y - 1
        current.board[new_x][new_y].set_value('#')
        return new_x, new_y, current
def play(current):
    x_s, y_s = current.get_start()
    x_g, y_g = current.get_goal()
    x_n, y_n = x_s, y_s
    while x_n != x_g or y_n != y_g:
        policy = current.board[x_n][y_n].get_P()
        # print("x: ", x_n, " and y: ", y_n, " and policy: ", policy)
        x_n, y_n, current = walk(current, policy, x_n, y_n)
        if x_n == x_g and y_n == y_g:
            current.print_board()
    return

def main():
    game = State()
    game.print_board()

    choice = eval(input("1. Value Iteration\n2. Policy Iteration\n3. Q Learning\nYour Choice? "))

    if choice == 1:
        # value iteraion
        value_iteration(game)
        #game.print_board_val()
        print("Training Finished")
        play_game(game)
        game.print_board()
        print("Walking Finished")

    if choice == 2:
        # policy iteration
        result = policy_iteration(game)
        print("Training Finished")
        play(result)
        print("Walking Finished")

    if choice == 3:
        # Q learning
        game = q_learning(game)
        print("Training Finished")
        #game.print_board_val()
        game = play_game(game)
        game.print_board()
        print("Walking Finished")

main()
