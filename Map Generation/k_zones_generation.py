import networkx as nx
import matplotlib.pyplot as plt
import random

class KFloorGenerator:
    def __init__(self, num_nodes=20, edge_probability=0.2, 
                 alpha=1.0, beta=1.0, gamma=1.0, 
                 prune_threshold_percent=0.15, seed=None, visualize=True):
        
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.alpha = alpha      # Weight for degree centrality
        self.beta = beta        # Weight for betweenness centrality
        self.gamma = gamma      # Weight for closeness centrality
        self.prune_threshold_percent = prune_threshold_percent  # % of nodes to prune based on lowest score
        self.seed = seed
        self.visualize_enabled = visualize
        
        random.seed(self.seed)
        
        self.G = None
        self.pruned_nodes = set()
        self.node_scores = {}
        
        if self.visualize_enabled:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.pos = None  # Node positions for visualization layout

    def generate_erdos_renyi_graph(self):
        """Generates a random graph using the Erdős-Rényi G(n, p) model."""
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        
        # Add edges based on edge probability
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < self.edge_probability:
                    self.G.add_edge(i, j)
        
        # Ensure the graph is fully connected
        if not nx.is_connected(self.G) and self.num_nodes > 1:
            components = list(nx.connected_components(self.G))
            main_component = max(components, key=len)
            for comp in components:
                if comp != main_component:
                    self.G.add_edge(random.choice(list(comp)), random.choice(list(main_component)))
        
        print(f"Generated Erdős-Rényi graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        print(f"Number of connected components: {nx.number_connected_components(self.G)}")

    def calculate_centrality_metrics(self):
        """Calculates degree, betweenness, and closeness centralities for all nodes."""
        dc = nx.degree_centrality(self.G)
        bc = nx.betweenness_centrality(self.G)
        cc = nx.closeness_centrality(self.G)
        
        self.node_scores = {}
        for node in self.G.nodes():
            score = (self.alpha * dc.get(node, 0) + 
                     self.beta * bc.get(node, 0) + 
                     self.gamma * cc.get(node, 0))
            self.node_scores[node] = score

    def prune_graph(self):
        """Marks the lowest-score nodes as 'pruned' based on centrality score."""
        if not self.node_scores:
            self.calculate_centrality_metrics()
        
        sorted_nodes = sorted(self.node_scores.items(), key=lambda item: item[1])
        num_to_prune = int(self.num_nodes * self.prune_threshold_percent)
        
        for i in range(min(num_to_prune, len(sorted_nodes))):
            node_to_prune = sorted_nodes[i][0]
            self.pruned_nodes.add(node_to_prune)
            self.G.nodes[node_to_prune]['pruned'] = True
        
        print(f"{len(self.pruned_nodes)} nodes were pruned (marked in gray).")

    def generate_k_floor(self):
        """
        Generates the 'K Floor' – a connected random graph with heuristic pruning.

        This method performs the full generation pipeline:
        1. Generates an Erdős-Rényi graph.
        2. Computes node centrality scores.
        3. Prunes a percentage of the lowest-scoring nodes.
        4. (Optional) Visualizes the graph with scores and pruned nodes.

        This is the main entry point for generating and rendering the K Floor layout.
        """
        self.generate_erdos_renyi_graph()
        self.calculate_centrality_metrics()
        self.prune_graph()
        
        if self.visualize_enabled:
            self.visualize()
            plt.show()

    def visualize(self):
        """Renders the K Floor graph, highlighting pruned and active nodes."""
        if not self.visualize_enabled:
            return
            
        self.ax.clear()
        
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=self.seed)
        
        node_colors = []
        node_labels = {}
        for node in self.G.nodes():
            color = 'gray' if node in self.pruned_nodes else 'lightblue'
            node_colors.append(color)
            node_labels[node] = f"{node}\nScore: {self.node_scores.get(node, 0):.2f}"
            
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                labels=node_labels,
                node_color=node_colors,
                node_size=1000,
                font_size=8,
                font_weight='bold',
                edge_color='gray')
        
        self.ax.scatter([], [], c='lightblue', label='Active Node', s=120)
        self.ax.scatter([], [], c='gray', label='Pruned Node', s=120)
        self.ax.legend(loc='upper right')
        
        self.ax.set_title(f"K Floor with Heuristic Pruning (N={self.num_nodes}, p={self.edge_probability})")


if __name__ == "__main__":
    # Generation parameters
    num_nodes_k = 25            # Total number of nodes (rooms)
    edge_prob_k = 0.15          # Edge probability (affects graph density)
    alpha_weight = 1.0          # Weight for degree centrality
    beta_weight = 1.0           # Weight for betweenness centrality
    gamma_weight = 1.0          # Weight for closeness centrality
    prune_percentage = 0.20     # % of lowest-scoring nodes to prune
    
    # Instantiate and generate the K Floor
    k_floor_gen = KFloorGenerator(
        num_nodes=num_nodes_k,
        edge_probability=edge_prob_k,
        alpha=alpha_weight,
        beta=beta_weight,
        gamma=gamma_weight,
        prune_threshold_percent=prune_percentage,
        seed=42,
        visualize=True
    )
    
    k_floor_gen.generate_k_floor()
