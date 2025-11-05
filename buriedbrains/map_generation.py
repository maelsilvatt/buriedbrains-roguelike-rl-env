# buriedbrains/map_generation.py
import networkx as nx
import random
from typing import Dict, List, Tuple

# --- Geração da Zona de Progressão (Dinâmica, por Sucessores) ---
def generate_progression_successors(
    graph: nx.DiGraph, 
    parent_node: str, 
    nodes_per_floor_counters: Dict[int, int]
) -> Tuple[List[str], Dict[int, int]]:
    """
    Gera a topologia dos NÓS SUCESSORES para um nó de progressão (DAG).
    
    Esta função é chamada dinamicamente a cada passo de movimento, alinhando-se
    com a natureza POMDP e a poda de ramos do ambiente.
    
    :param graph: O objeto nx.DiGraph do agente (para ser modificado).
    :param parent_node: O nó a partir do qual gerar os sucessores (ex: "Sala 1_0").
    :param nodes_per_floor_counters: O dicionário de contagem de nós do agente.
    :return: Uma tupla (lista_de_novos_nos, dicionario_de_contagem_atualizado).
    """
    
    parent_floor = graph.nodes[parent_node].get('floor', 0)
    next_floor = parent_floor + 1
    branching_factor = 2 # Fator de ramificação (conforme Figura 2 do memorial) [cite: 322]
    
    new_node_names = []

    # Garante que o contador para o próximo andar exista
    if next_floor not in nodes_per_floor_counters:
        nodes_per_floor_counters[next_floor] = 0

    for i in range(branching_factor):
        current_index = nodes_per_floor_counters[next_floor]
        new_node_name = f"p_{next_floor}_{current_index}" # 'p' para progressão
        
        # Atualiza o contador para o próximo nó
        nodes_per_floor_counters[next_floor] += 1
        
        # Adiciona o nó e a aresta ao grafo
        graph.add_node(new_node_name, floor=next_floor)
        graph.add_edge(parent_node, new_node_name)
        
        new_node_names.append(new_node_name)
    
    # Retorna os nomes dos nós criados e o contador atualizado
    return new_node_names, nodes_per_floor_counters

# --- Geração da Zona do Karma (Estática, por Arena) ---
# Esta é a sua lógica, que está perfeita.

def _prune_graph_by_centrality(G: nx.Graph, alpha=0.5, beta=0.3, gamma=0.2) -> nx.Graph:
    """
    Poda arestas de um grafo com base em uma pontuação heurística de centralidade.
    Implementa a Equação 3.8 do memorial. [cite: 228-240, 233]
    """
    scores = {}
    
    # Normaliza centralidades para que a soma seja ~1.0
    # (Evita que (scores[u] + scores[v]) seja > 1.0)
    try:
        deg_cen = nx.degree_centrality(G)
        bet_cen = nx.betweenness_centrality(G)
        clo_cen = nx.closeness_centrality(G)

        # Normaliza (min-max scaling) cada centralidade para 0-1
        def normalize(d):
            min_v = min(d.values())
            max_v = max(d.values())
            range_v = max_v - min_v
            if range_v == 0:
                return {k: 0 for k in d}
            return {k: (v - min_v) / range_v for k, v in d.items()}

        deg_cen_n = normalize(deg_cen)
        bet_cen_n = normalize(bet_cen)
        clo_cen_n = normalize(clo_cen)

        scores = {node: alpha * deg_cen_n.get(node, 0) +
                        beta * bet_cen_n.get(node, 0) +
                        gamma * clo_cen_n.get(node, 0)
                  for node in G.nodes()}
                  
    except Exception as e:
        # Fallback se a centralidade falhar (ex: grafo muito pequeno)
        print(f"[Map Generation WARN] Falha ao calcular centralidade: {e}")
        scores = {node: 0.5 for node in G.nodes()} # Valor neutro

    edges_to_remove = []
    
    for u, v in G.edges():
        # Arestas conectadas a nós de baixa pontuação têm maior chance de serem removidas
        score_sum = scores.get(u, 0) + scores.get(v, 0)
        # Ajusta a probabilidade de remoção (ex: random() > (0.5 + score_sum/2))
        prob_keep = 0.5 + (score_sum / 4) # Ajuste esta heurística
        
        if random.random() > prob_keep:
            # Garante que não desconecta o grafo
            if G.degree[u] > 1 and G.degree[v] > 1:
                edges_to_remove.append((u, v))
                
    G.remove_edges_from(edges_to_remove)
    return G

def generate_k_zone_topology(
    floor_level: int,
    num_nodes: int = 15,
    connectivity_prob: float = 0.25,
    pruning_params: Dict[str, float] = None,
    seed: int = None
) -> nx.Graph:
    """
    Gera a topologia para uma Zona do Karma usando um modelo Erdős-Rényi podado.
    Como descrito no memorial. 
    
    :param floor_level: O andar atual (para nomear as salas).
    """
    if seed is not None:
        random.seed(seed)
        
    G = nx.Graph() # Começa vazio
    
    # Tenta gerar um grafo conectado
    for _ in range(10): # Tenta 10 vezes garantir conectividade
        G = nx.erdos_renyi_graph(num_nodes, connectivity_prob)
        if nx.is_connected(G):
            break
    else:
        # Se falhar 10x, força a conectividade (sua lógica original)
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i+1]))
            G.add_edge(node1, node2)
            
    # 3. Poda o grafo para complexidade estratégica 
    if pruning_params:
        G_pruned = G.copy()
        G_pruned = _prune_graph_by_centrality(G_pruned, **pruning_params)
        # Garante a conectividade novamente após a poda
        if nx.is_connected(G_pruned):
            G = G_pruned
        # Se a poda desconectou, silenciosamente usa o grafo original (não podado)

    # 4. Renomeia os nós e adiciona atributos de andar
    mapping = {old_node: f"k_{floor_level}_{old_node}" for old_node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(G, floor_level, 'floor')

    return G