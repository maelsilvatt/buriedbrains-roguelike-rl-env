import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

class ProgressiveDAG:
    def __init__(self, max_layers=500, total_steps=500, 
                 bifurcation_probs=[0.5, 0.3, 0.2], 
                 max_no_bifurcation=3, seed=None, visualize=True):
        """
        Initializes the ProgressiveDAG class.

        Parameters:
        - max_layers: Maximum depth (number of layers) of the DAG.
        - total_steps: Total number of agent steps (used to define bifurcation phases).
        - bifurcation_probs: List of bifurcation probabilities for each phase.
        - max_no_bifurcation: Max corridor length before forced branching.
        - seed: Random seed for reproducibility.
        - visualize: Whether to enable live visualization.
        """
        self.G = nx.DiGraph()
        self.max_layers = max_layers
        self.bifurcation_probs = bifurcation_probs
        self.max_no_bifurcation = max_no_bifurcation
        self.total_steps = total_steps
        self.visualize_enabled = visualize
        
        self.layers = {0: 0}
        self.next_node = 1
        self.visited = set()
        self.frontier = deque([0])
        self.corridor_length = {0: 0}
        
        random.seed(seed)
        
        num_phases = len(bifurcation_probs)
        self.phase_thresholds = [(i + 1) * total_steps // num_phases for i in range(num_phases)]
        
        self.G.add_node(0, layer=0, visited=False)
        
        if self.visualize_enabled:
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.pos = {0: (0, 0)}

    def get_current_bifurcation_prob(self, current_step):
        """Returns the current bifurcation probability based on simulation step."""
        for i, threshold in enumerate(self.phase_thresholds):
            if current_step <= threshold:
                return self.bifurcation_probs[i]
        return self.bifurcation_probs[-1]

    def expand_node(self, node, current_step):
        """
        Expands a node by adding one or two children depending on bifurcation rules.
        """
        if node not in self.frontier:
            return

        self.frontier.remove(node)
        self.visited.add(node)
        self.G.nodes[node]['visited'] = True
        
        current_layer = self.layers[node]
        if current_layer >= self.max_layers - 1:
            return
        
        # Decide whether to create 1 or 2 children
        children = 1
        if self.corridor_length.get(node, 0) >= self.max_no_bifurcation and current_layer < self.max_layers - 2:
            children = 2
        else:
            prob = self.get_current_bifurcation_prob(current_step)
            if random.random() < prob and current_layer < self.max_layers - 2:
                children = 2

        for i in range(children):
            child = self.next_node
            self.G.add_edge(node, child)
            self.layers[child] = current_layer + 1
            self.G.add_node(child, layer=current_layer + 1, visited=False)
            self.frontier.append(child)
            
            # Update corridor length: reset on bifurcation, increase on linear path
            if children == 2:
                self.corridor_length[child] = 0
            else:
                self.corridor_length[child] = self.corridor_length.get(node, 0) + 1

            # Positioning logic for visualization
            if self.visualize_enabled:
                x_pos = self.layers[child]
                y_spacing = 0.8
                y_base = self.pos[node][1]

                if children == 1:
                    y_pos = y_base
                else:
                    y_pos = y_base + (y_spacing if i == 0 else -y_spacing)
                
                self.pos[child] = (x_pos, y_pos)

            self.next_node += 1

    def agent_move(self, node, current_step):
        """
        Simulates an agent visiting a node, expanding it if needed.

        Returns a list of the node's children (successors).
        """
        if node not in self.G:
            raise ValueError("Node does not exist in the graph.")
        
        self.expand_node(node, current_step)
        return list(self.G.successors(node))

    def visualize(self, current_node=None):
        """
        Renders the current state of the DAG with node colors indicating state:
        - gold: current agent position
        - lightgreen: visited
        - salmon: frontier (available to expand)
        - lightblue: unexplored
        """
        if not self.visualize_enabled:
            return
        
        self.ax.clear()
        
        node_colors = []
        for node in self.G.nodes():
            if node == current_node:
                node_colors.append('gold')
            elif self.G.nodes[node]['visited']:
                node_colors.append('lightgreen')
            elif node in self.frontier:
                node_colors.append('salmon')
            else:
                node_colors.append('lightblue')
        
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=8, edge_color='gray', arrows=True)
        
        title_step = path.index(current_node) if current_node in path else 0
        self.ax.set_title(f"DAG Exploration (Step: {title_step}/{TOTAL_STEPS})")
        plt.pause(0.01)
        self.fig.canvas.draw()


# --- Simulation Parameters ---
TOTAL_STEPS = 50
VISUALIZE = True
BIFURCATION_PROBS = [0.6, 0.4, 0.3]
MAX_NO_BIFURCATION = 4

# --- Start Simulation ---
if VISUALIZE:
    plt.ion()

dag = ProgressiveDAG(
    total_steps=TOTAL_STEPS,
    bifurcation_probs=BIFURCATION_PROBS,
    max_no_bifurcation=MAX_NO_BIFURCATION,
    seed=random.randint(1, 1000),
    visualize=VISUALIZE
)

path = [0]
if VISUALIZE:
    dag.visualize(current_node=0)

# Agent movement simulation
for i in range(TOTAL_STEPS):
    current = path[-1]
    neighbors = dag.agent_move(current, i)
    
    if neighbors:
        next_node = random.choice(neighbors)
        path.append(next_node)
        if VISUALIZE:
            dag.visualize(current_node=next_node)
    else:
        print(f"\nAgent reached a dead end at step {i+1}.")
        break

# Final visualization
if VISUALIZE:
    plt.ioff()
    dag.visualize(current_node=path[-1])
    plt.show()

# --- Final Report ---
print(f"Planned steps: {TOTAL_STEPS}")
print(f"Executed steps: {len(path)-1}")
print(f"Total nodes generated: {dag.next_node}")
print(f"Bifurcation probabilities: {BIFURCATION_PROBS}")
print(f"Corridor limit (max_no_bifurcation): {MAX_NO_BIFURCATION}")
