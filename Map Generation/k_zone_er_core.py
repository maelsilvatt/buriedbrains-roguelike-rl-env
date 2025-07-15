import networkx as nx
import matplotlib.pyplot as plt
import random

class KFloorGenerator:
    def __init__(self, num_nodes=20, edge_probability=0.2, 
                 alpha=1.0, beta=1.0, gamma=1.0, 
                 prune_threshold_percent=0.15, seed=None, visualize=True):
        
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.alpha = alpha  # Peso para Centralidade de Grau
        self.beta = beta    # Peso para Centralidade de Intermediação
        self.gamma = gamma  # Peso para Centralidade de Proximidade
        self.prune_threshold_percent = prune_threshold_percent # Porcentagem dos nós com menor score a serem podados
        self.seed = seed
        self.visualize_enabled = visualize
        
        random.seed(self.seed)
        
        self.G = None
        self.pruned_nodes = set()
        self.node_scores = {}
        
        if self.visualize_enabled:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.pos = None # Posições serão geradas com o grafo

    def generate_erdos_renyi_graph(self):
        """Gera um grafo aleatório usando o modelo Erdős-Rényi G(n,p)."""
        # Garante que o grafo seja não-direcionado para o Andar K
        self.G = nx.Graph() 
        self.G.add_nodes_from(range(self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < self.edge_probability:
                    self.G.add_edge(i, j)
        
        # Opcional: Garante que o grafo seja conectado, se necessário para o benchmark
        # Se não for conectado, pode gerar múltiplos "pedaços" de mapa.
        if not nx.is_connected(self.G) and self.num_nodes > 1:
            components = list(nx.connected_components(self.G))
            # Conecta componentes menores ao maior
            main_component = max(components, key=len)
            for comp in components:
                if comp != main_component:
                    # Conecta um nó do componente menor a um nó do componente principal
                    self.G.add_edge(random.choice(list(comp)), random.choice(list(main_component)))
        
        print(f"Grafo inicial Erdős-Rényi gerado com {self.G.number_of_nodes()} nós e {self.G.number_of_edges()} arestas.")
        print(f"Número de componentes conectados: {nx.number_connected_components(self.G)}")


    def calculate_centrality_metrics(self):
        """Calcula as métricas de centralidade para cada nó."""
        
        # Centralidade de Grau (Degree Centrality)
        # cd[v] = deg(v) / (n-1)
        dc = nx.degree_centrality(self.G)
        
        # Centralidade de Intermediação (Betweenness Centrality)
        # C_B(v) = sum(s!=v!=t) [sigma_st(v) / sigma_st]
        bc = nx.betweenness_centrality(self.G)
        
        # Centralidade de Proximidade (Closeness Centrality)
        # C_C(v) = (n-1) / sum(u!=v) d(u,v)
        # Nota: Closeness Centrality é calculada por padrão apenas para componentes conectados
        # Se o grafo não for totalmente conectado, ele calcula para cada componente.
        cc = nx.closeness_centrality(self.G)
        
        self.node_scores = {}
        for node in self.G.nodes():
            # Função Score(v) = α*C_D(v) + β*C_B(v) + γ*C_C(v) 
            score = (self.alpha * dc.get(node, 0) + 
                     self.beta * bc.get(node, 0) + 
                     self.gamma * cc.get(node, 0))
            self.node_scores[node] = score

    def prune_graph(self):
        """Aplica a poda heurística baseada nos scores de centralidade."""
        if not self.node_scores:
            self.calculate_centrality_metrics()
            
        # Ordena os nós pelos scores de centralidade (menores primeiro)
        sorted_nodes = sorted(self.node_scores.items(), key=lambda item: item[1])
        
        # Calcula quantos nós devem ser podados
        num_to_prune = int(self.num_nodes * self.prune_threshold_percent)
        
        # Seleciona os nós com os menores scores para podar
        for i in range(min(num_to_prune, len(sorted_nodes))):
            node_to_prune = sorted_nodes[i][0]
            self.pruned_nodes.add(node_to_prune)
            # O nó não é removido, mas será marcado como 'pruned' e visualizado em cinza
            self.G.nodes[node_to_prune]['pruned'] = True
        
        print(f"Total de {len(self.pruned_nodes)} nós podados (cinzas).")

    def generate_k_floor(self):
        """Executa o processo completo de geração do Andar K."""
        self.generate_erdos_renyi_graph()
        self.calculate_centrality_metrics()
        self.prune_graph()
        
        if self.visualize_enabled:
            self.visualize()
            plt.show() 

    def visualize(self):
        """Visualiza o grafo do Andar K."""
        if not self.visualize_enabled:
            return
            
        self.ax.clear()
        
        # nx.spring_layout é bom para grafos não direcionados
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=self.seed) 
        
        node_colors = []
        node_labels = {}
        for node in self.G.nodes():
            if node in self.pruned_nodes:
                node_colors.append('gray') # Nós podados em cinza
            else:
                node_colors.append('lightblue') # Nós ativos
            node_labels[node] = f"{node}\nScore: {self.node_scores.get(node, 0):.2f}"
            
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                labels=node_labels, # Adiciona labels com o score
                node_color=node_colors, 
                node_size=1000, # Tamanho do nó maior para o score
                font_size=8,
                font_weight='bold',
                edge_color='gray')
        
        # Legenda dos nós podados
        self.ax.scatter([], [], c='lightblue', label='Nó Ativo', s=120)
        self.ax.scatter([], [], c='gray', label='Nó Podado (Inativo)', s=120)
        self.ax.legend(loc='upper right')
        
        self.ax.set_title(f"Andar K Gerado com Poda Heurística (Nós: {self.num_nodes}, Prob. Aresta: {self.edge_probability})")

if __name__ == "__main__":
    # Parametros para o Andar K
    num_nodes_k = 25 # Número de salas no andar K
    edge_prob_k = 0.15 # Probabilidade de conexão entre as salas (ajuste para densidade)
    
    # Pesos para as métricas de centralidade (alpha, beta, gamma)
    # Reflete a ênfase em Exploração, Controle Territorial, Mobilidade
    alpha_weight = 1.0 
    beta_weight = 1.0
    gamma_weight = 1.0
    
    # Porcentagem dos nós com os menores scores de centralidade a serem podados
    prune_percentage = 0.20 # Poda 20% dos nós com os scores mais baixos
    
    # Gerar e visualizar o Andar K
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