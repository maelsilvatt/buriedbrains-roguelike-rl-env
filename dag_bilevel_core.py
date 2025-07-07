import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

class ProgressiveDAG:
    def __init__(self, max_layers=500, total_steps=500, 
                 bifurcation_probs=[0.5, 0.3, 0.2], 
                 max_no_bifurcation=3, seed=None, visualize=True):
        """
        Inicializador do DAG Progressivo.
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
        """Retorna a probabilidade de bifurcação com base no passo atual da simulação."""
        for i, threshold in enumerate(self.phase_thresholds):
            if current_step <= threshold:
                return self.bifurcation_probs[i]
        return self.bifurcation_probs[-1]

    def expand_node(self, node, current_step):
        """Expande um nó, gerando seus filhos com base nas novas regras."""
        if node not in self.frontier:
            return

        self.frontier.remove(node)
        self.visited.add(node)
        self.G.nodes[node]['visited'] = True
        
        current_layer = self.layers[node]
        if current_layer >= self.max_layers - 1:
            return
            
        children = 1
        if (self.corridor_length.get(node, 0) >= self.max_no_bifurcation and 
            current_layer < self.max_layers - 2):
            children = 2
        else:
            prob = self.get_current_bifurcation_prob(current_step)
            if (random.random() < prob and current_layer < self.max_layers - 2):
                children = 2
        
        for i in range(children):
            child = self.next_node
            self.G.add_edge(node, child)
            self.layers[child] = current_layer + 1
            self.G.add_node(child, layer=current_layer + 1, visited=False)
            self.frontier.append(child)
            
            if children == 2:
                self.corridor_length[child] = 0
            else:
                self.corridor_length[child] = self.corridor_length.get(node, 0) + 1

            if self.visualize_enabled:
                # --- LÓGICA DE POSICIONAMENTO UNIFORME (MODIFICADA) ---
                x_pos = self.layers[child]  # Usa a camada do filho para a posição X

                # Define um espaçamento vertical fixo para as bifurcações
                y_spacing = 0.8 
                
                y_base = self.pos[node][1] # Posição Y do pai

                if children == 1:
                    # Se for filho único, mantém a mesma altura do pai para criar corredores retos
                    y_pos = y_base
                else:
                    # Se forem dois filhos, espalha-os uniformemente acima e abaixo do pai
                    # Para o primeiro filho (i=0), o resultado é +y_spacing
                    # Para o segundo filho (i=1), o resultado é -y_spacing
                    y_pos = y_base + (y_spacing if i == 0 else -y_spacing)
                
                self.pos[child] = (x_pos, y_pos)
                
            self.next_node += 1
    
    def agent_move(self, node, current_step):
        """Simula o movimento do agente, passando o passo atual para a expansão."""
        if node not in self.G:
            raise ValueError("Nó não existe no grafo")
        
        self.expand_node(node, current_step)
        return list(self.G.successors(node))
    
    def visualize(self, current_node=None):
        """Visualiza o DAG."""
        if not self.visualize_enabled:
            return
            
        self.ax.clear()
        
        node_colors = []
        for node in self.G.nodes():
            if node == current_node: node_colors.append('gold')
            elif self.G.nodes[node]['visited']: node_colors.append('lightgreen')
            elif node in self.frontier: node_colors.append('salmon')
            else: node_colors.append('lightblue')
        
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=8, edge_color='gray', arrows=True)
        
        title_step = path.index(current_node) if current_node in path else 0
        self.ax.set_title(f"Exploração do DAG (Passo: {title_step}/{TOTAL_STEPS})")
        plt.pause(0.01)
        self.fig.canvas.draw()

# --- Parâmetros da Simulação ---
TOTAL_STEPS = 50
VISUALIZE = True
BIFURCATION_PROBS = [0.6, 0.4, 0.3]
MAX_NO_BIFURCATION = 4

# --- Início da Simulação ---
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

# Simulação do agente
for i in range(TOTAL_STEPS):
    current = path[-1]
    neighbors = dag.agent_move(current, i)
    
    if neighbors:
        next_node = random.choice(neighbors)
        path.append(next_node)
        if VISUALIZE:
            dag.visualize(current_node=next_node)
    else:
        print(f"\nO agente chegou a um nó sem saída no passo {i+1}!")
        break

if VISUALIZE:
    plt.ioff()
    dag.visualize(current_node=path[-1])
    plt.show()

# --- Relatório Final ---
print(f"Total de passos planejados: {TOTAL_STEPS}")
print(f"Total de passos executados: {len(path)-1}")
print(f"Total de nós gerados: {dag.next_node}")
print(f"Probabilidades de Bifurcação usadas: {BIFURCATION_PROBS}")
print(f"Limite de corredor (max_no_bifurcation): {MAX_NO_BIFURCATION}")