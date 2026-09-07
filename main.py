import numpy as np
from heapq import heappush, heappop

class PuzzleNode:
    def __init__(self, state):
        # store as tuple for hashing; accept list/tuple/np array
        self.current_state = tuple(state)
        self.previous_node = None
        self.heuristic_cost = 0
        self.path_cost = 0  # g(n)

    # ----- getters / setters (your original style) -----
    def get_previous_node(self):
        return self.previous_node

    def get_state(self):
        return self.current_state

    def update_path_cost(self, cost):
        self.path_cost = cost

    def update_heuristic_cost(self, cost):
        self.heuristic_cost = cost

    def set_previous_node(self, node):
        self.previous_node = node

    def get_heuristic_cost(self):
        return self.heuristic_cost

    def get_path_cost(self):
        return self.path_cost

    def get_total_cost(self):
        return self.path_cost + self.heuristic_cost

    # allow nodes to be compared in heap if needed (tie-break)
    def __lt__(self, other):
        return self.get_total_cost() < other.get_total_cost()

    # -------- COST / HEURISTICS (fixed) --------
    @staticmethod
    def calculate_cost(state, goal, heuristic_type):
        """
        heuristic_type:
          1 -> Misplaced tiles (ignoring 0)
          2 -> Manhattan distance (ignoring 0)
        """
        # ensure tuples for consistency
        state = tuple(state)
        goal = tuple(goal)

        if heuristic_type == 1:
            # Misplaced tiles (ignore blank 0)
            return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])

        elif heuristic_type == 2:
            # Manhattan distance (ignore blank 0)
            cost = 0
            # map goal positions
            goal_pos = {tile: (idx // 3, idx % 3) for idx, tile in enumerate(goal)}
            for idx, tile in enumerate(state):
                if tile == 0:
                    continue
                r, c = divmod(idx, 3)
                gr, gc = goal_pos[tile]
                cost += abs(r - gr) + abs(c - gc)
            return cost

        else:
            return 0

    # -------- SUCCESSOR GENERATION (clean) --------
    @staticmethod
    def expand_neighbors(current_node):
        """
        Generate neighboring states by sliding a tile into the blank.
        Returns a list[tuple] of neighbor states (tuples).
        """
        s = list(current_node.get_state())
        zi = s.index(0)
        neighbors = []

        # valid zero moves for 3x3
        possible = []
        if zi >= 3: possible.append(zi - 3)   # up
        if zi % 3 != 0: possible.append(zi - 1)  # left
        if zi % 3 != 2: possible.append(zi + 1)  # right
        if zi < 6: possible.append(zi + 3)   # down

        for swap in possible:
            ns = s.copy()
            ns[zi], ns[swap] = ns[swap], ns[zi]
            neighbors.append(tuple(ns))

        return neighbors

    # -------- TRACE (kept your printing style) --------
    @staticmethod
    def trace_solution_path(goal_node, generated_nodes, expanded_nodes):
        # reconstruct chain from goal_node back to start via previous_node
        path_nodes = []
        cur = goal_node
        while cur is not None:
            path_nodes.append(cur)
            cur = cur.get_previous_node()
        path_nodes.reverse()

        # pretty print boards with h/g/f
        for node in path_nodes:
            st = node.get_state()
            print(st[0:3])
            print(st[3:6])
            print(st[6:9])
            print("h(n):", node.get_heuristic_cost())
            print("g(n):", node.get_path_cost())
            print("f(n) = h(n) + g(n):", node.get_total_cost())
            print("\n")

        print("Goal Reached\n")
        print("Path Cost:", len(path_nodes) - 1)
        print("Expanded Nodes:", expanded_nodes)
        print("Nodes Generated:", generated_nodes)

# ---------- utils: input, solvability ----------
def read_board_lines(prompt):
    """
    Keep your '9 separate inputs' feel but accept either:
      - 9 numbers on one line (spaces/commas)
      - OR 9 numbers one per line
    """
    print(prompt)
    nums = []
    while len(nums) < 9:
        raw = input().strip()
        if not raw:
            continue
        # allow a single line with multiple numbers
        raw = raw.replace(",", " ")
        parts = [p for p in raw.split() if p]
        for p in parts:
            try:
                nums.append(int(p))
            except:
                pass
    if set(nums) != set(range(0, 9)):
        raise ValueError("State must be a permutation of 0..8 (0=blank).")
    return tuple(nums[:9])

def inversions(state):
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def is_solvable(start, goal):
    # 3x3 solvability: inversion parity must match
    return inversions(start) % 2 == inversions(goal) % 2

# ---------- A* (correct counters & logic) ----------
def a_star_solve(start_state, goal_state, heuristic_type):
    """
    Returns (goal_node, nodes_generated, nodes_expanded)
      - nodes_generated: times we actually PUSHED a node to frontier
      - nodes_expanded: times we POPPED a node from frontier (first time)
    """
    if start_state == goal_state:
        node = PuzzleNode(start_state)
        node.update_path_cost(0)
        node.update_heuristic_cost(0)
        return node, 0, 0

    # g(n) best known cost and parent mapping
    g_score = {start_state: 0}
    parent = {start_state: None}

    # priority queue entries: (f, g, state)
    frontier = []
    h0 = PuzzleNode.calculate_cost(start_state, goal_state, heuristic_type)
    heappush(frontier, (h0, 0, start_state))
    nodes_generated = 0   # only count when we push into the heap (excluding start if you prefer)
    nodes_expanded = 0

    closed = set()

    while frontier:
        f, g, state = heappop(frontier)
        if state in closed:
            # already expanded this best version
            continue
        closed.add(state)
        nodes_expanded += 1

        if state == goal_state:
            # build PuzzleNode chain from parent/g_score
            return build_solution_node_chain(state, parent, g_score, goal_state, heuristic_type), nodes_generated, nodes_expanded

        # expand neighbors
        dummy_node = PuzzleNode(state)
        for nbr in PuzzleNode.expand_neighbors(dummy_node):
            tentative_g = g + 1
            # if never seen or found a better path
            if nbr not in g_score or tentative_g < g_score[nbr]:
                g_score[nbr] = tentative_g
                parent[nbr] = state
                hn = PuzzleNode.calculate_cost(nbr, goal_state, heuristic_type)
                fn = tentative_g + hn
                heappush(frontier, (fn, tentative_g, nbr))
                nodes_generated += 1

    return None, nodes_generated, nodes_expanded  # no solution

def build_solution_node_chain(goal_state, parent, g_score, goal, heuristic_type):
    # reconstruct state path first
    path_states = []
    s = goal_state
    while s is not None:
        path_states.append(s)
        s = parent[s]
    path_states.reverse()

    # now build linked PuzzleNode chain with correct g/h and previous_node
    prev_node = None
    first_node = None
    for st in path_states:
        node = PuzzleNode(st)
        g = g_score[st]
        h = PuzzleNode.calculate_cost(st, goal, heuristic_type)
        node.update_path_cost(g)
        node.update_heuristic_cost(h)
        node.set_previous_node(prev_node)
        if first_node is None:
            first_node = node
        prev_node = node
    # prev_node is the goal node at the end
    return prev_node

# ---------- main (keeps your prompts/flow) ----------
if __name__ == "__main__":
    print("Enter the initial puzzle state (9 numbers; 0 is blank).")
    initial = read_board_lines("You can enter 9 numbers one-per-line or all on one line:")

    print("\nInitial State:")
    print(initial[0:3]); print(initial[3:6]); print(initial[6:9])

    print("\nEnter the goal puzzle state (9 numbers; 0 is blank).")
    goal = read_board_lines("You can enter 9 numbers one-per-line or all on one line:")

    print("\nGoal State:")
    print(goal[0:3]); print(goal[3:6]); print(goal[6:9])

    # heuristic selection
    while True:
        try:
            heuristic_choice = int(input("\nSelect the heuristic function:\n1. Misplaced Tiles\n2. Manhattan Distance\n> ").strip())
            if heuristic_choice not in (1, 2):
                raise ValueError
            break
        except ValueError:
            print("Please enter 1 or 2.")

    print("\nSelected heuristic:", "Misplaced Tiles" if heuristic_choice == 1 else "Manhattan Distance")

    # solvability check (important!)
    if not is_solvable(initial, goal):
        print("\nThis initial state is NOT solvable with respect to the chosen goal. No search performed.")
    else:
        print("\nSolving Puzzle:\n")
        goal_node, nodes_generated, nodes_expanded = a_star_solve(initial, goal, heuristic_choice)
        if goal_node is None:
            print("No solution exists.")
        else:
            PuzzleNode.trace_solution_path(goal_node, nodes_generated, nodes_expanded)
