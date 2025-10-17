# buriedbrains/map_generation.py
import networkx as nx
import random
from typing import Dict, Any

# --- Geração da Zona de Progressão (DAG) ---
def generate_p_zone_topology(
    num_floors: int = 5,
    branching_factor: int = 2,
    seed: int = None
) -> nx.DiGraph:
    """
    Gera a topologia para uma Zona de Progressão como um Grafo Acíclico Dirigido (DAG).
   
    """
    if seed is not None:
        random.seed(seed)
        
    G = nx.DiGraph()
    G.add_node("start", floor=0)
    
    current_level_nodes = ["start"]
    
    for i in range(num_floors):
        next_level_nodes = []
        for node in current_level_nodes:
            for j in range(branching_factor):
                new_node = f"p_{i+1}_{len(next_level_nodes)}"
                G.add_node(new_node, floor=i + 1)
                G.add_edge(node, new_node)
                next_level_nodes.append(new_node)
        current_level_nodes = next_level_nodes
        
    return G

# --- Geração da Zona do Karma (Erdos-Renyi) ---
def _prune_graph_by_centrality(G: nx.Graph, alpha=0.5, beta=0.3, gamma=0.2) -> nx.Graph:
    """
    Poda arestas de um grafo com base em uma pontuação heurística de centralidade dos nós.
   
    """
    scores = {node: alpha * nx.degree_centrality(G)[node] +
                      beta * nx.betweenness_centrality(G)[node] +
                      gamma * nx.closeness_centrality(G)[node]
              for node in G.nodes()}

    edges_to_remove = []
    # A heurística de remoção pode ser ajustada
    for u, v in G.edges():
        # Arestas conectadas a nós de baixa pontuação têm maior chance de serem removidas
        if random.random() > (scores[u] + scores[v]):
            if G.degree[u] > 1 and G.degree[v] > 1: # Garante que não desconecta o grafo
                edges_to_remove.append((u, v))
            
    G.remove_edges_from(edges_to_remove)
    return G

def generate_k_zone_topology(
    num_nodes: int = 15,
    connectivity_prob: float = 0.25,
    pruning_params: Dict[str, float] = None,
    seed: int = None
) -> nx.Graph:
    """
    Gera a topologia para uma Zona do Karma usando um modelo Erdős-Rényi podado.
   
    """
    if seed is not None:
        random.seed(seed)
        
    # 1. Gera o grafo aleatório base
    G = nx.erdos_renyi_graph(num_nodes, connectivity_prob, seed=seed)
    
    # 2. Garante que o grafo seja conectado
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i+1]))
            G.add_edge(node1, node2)
            
    # 3. Poda o grafo para complexidade estratégica, se parâmetros forem fornecidos
    if pruning_params:
        G = _prune_graph_by_centrality(G, **pruning_params)
        # Garante a conectividade novamente após a poda
        if not nx.is_connected(G):
             # Retorna o grafo não podado se a poda o desconectou
             G = nx.erdos_renyi_graph(num_nodes, connectivity_prob, seed=seed)

    # 4. Renomeia os nós para clareza
    mapping = {old_node: f"k_{i}" for i, old_node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G