import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

class ProgressiveDAG:
    def __init__(self, max_layers=6, bifurcation_prob=0.7, seed=None, visualize=True):
        self.G = nx.DiGraph()
        self.layers = {0: 0}  # {node: layer}
        self.max_layers = max_layers
        self.bifurcation_prob = bifurcation_prob
        self.next_node = 1
        self.visited = set()
        self.frontier = deque([0])  
        self.frontier_expansion_count = 0  # Contador de expansões da fronteira      
        self.visualize_enabled = visualize  # Flag para visualização
        random.seed(seed)
        
        # Adiciona o nó inicial
        self.G.add_node(0, layer=0, visited=False)
        
        if self.visualize_enabled:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.pos = {0: (0, 0)}  # Inicializa com posição do nó 0
    
    def expand_node(self, node):
        """Expande um nó (gera seus filhos) se estiver na fronteira"""
        if node not in self.frontier:
            return
            
        self.frontier.remove(node)
        self.frontier_expansion_count += 1
        self.visited.add(node)
        self.G.nodes[node]['visited'] = True
        
        current_layer = self.layers[node]
        if current_layer >= self.max_layers - 1:
            return
            
        # Decide se terá 1 ou 2 filhos
        children = 2 if (random.random() < self.bifurcation_prob and 
                        current_layer < self.max_layers - 2) else 1                    
        
        for _ in range(children):
            child = self.next_node
            self.G.add_edge(node, child)
            self.layers[child] = current_layer + 1
            self.G.add_node(child, layer=current_layer + 1, visited=False)
            self.frontier.append(child)                        
            
            if self.visualize_enabled:
                # Calcula posição para o novo nó
                x_pos = current_layer + 1
                y_offset = 0.5 if children == 2 else 0
                y_pos = self.pos[node][1] + (y_offset if _ == 0 else -y_offset)
                self.pos[child] = (x_pos, y_pos)
                  # Atualiza contador de expansões da fronteira

            self.next_node += 1
    
    def agent_move(self, node):
        """Simula o movimento do agente para um nó"""
        if node not in self.G:
            raise ValueError("Nó não existe no grafo")
            
        self.expand_node(node)
        return list(self.G.successors(node))
    
    def visualize(self, current_node=None):
        """Visualiza o DAG com atualização em tempo real"""
        if not self.visualize_enabled:
            return
            
        self.ax.clear()
        
        node_colors = []
        node_sizes = []
        for node in self.G.nodes():
            if node == current_node:
                node_colors.append('gold')  # Nó atual (destaque)
                node_sizes.append(1200)
            elif self.G.nodes[node]['visited']:
                node_colors.append('lightgreen')  # Visitado
                node_sizes.append(800)
            elif node in self.frontier:
                node_colors.append('salmon')      # Fronteira
                node_sizes.append(800)
            else:
                node_colors.append('lightblue')   # Não gerado
                node_sizes.append(800)
        
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
               node_color=node_colors, 
               node_size=node_sizes,
               font_weight='bold',
               edge_color='gray')
        
        # Legenda
        self.ax.scatter([], [], c='gold', label='Atual', s=120)
        self.ax.scatter([], [], c='lightgreen', label='Visitado', s=120)
        self.ax.scatter([], [], c='salmon', label='Fronteira', s=120)
        self.ax.scatter([], [], c='lightblue', label='Não gerado', s=120)
        self.ax.legend(loc='upper right')
        
        title = f"Exploração Progressiva do DAG (Nó atual: {current_node})"
        self.ax.set_title(title)

        plt.pause(0.5)  # Pausa por 0.5 segundos
        self.fig.canvas.draw()

# --- Simulação com visualização em tempo real --- #
VISUALIZE = False

if VISUALIZE:
    plt.ion()  # Modo interativo

dag = ProgressiveDAG(max_layers=50000, bifurcation_prob=0.6, 
                    seed=random.randint(1, 100), visualize=VISUALIZE)

# Visualização inicial
dag.visualize(current_node=0)

# Simulação do agente explorando
path = [0]  # Raiz
for _ in range(50000):  # Faz 50000 movimentos
    current = path[-1]
    neighbors = dag.agent_move(current)
    
    if neighbors:
        # Escolhe um vizinho aleatório para "visitar" na próxima iteração
        next_node = random.choice(neighbors)
        path.append(next_node)
        dag.visualize(current_node=next_node)
    else:
        print("O agente chegou a um nó sem saída!")
        break

if VISUALIZE:
    plt.ioff()  # Desativa o modo interativo
    dag.visualize()  # Mostra o resultado final

print("\nResumo da exploração:")
# print("Caminho do agente:", path)
# print("Nós visitados:", sorted(dag.visited))
# print("Fronteira final:", list(dag.frontier))
print("Total de nós gerados:", dag.next_node)
print("Expansões da fronteira:", dag.frontier_expansion_count)

if VISUALIZE:
    plt.show()  # Mantém a janela aberta