# buriedbrains/reputation.py
import numpy as np
from typing import Dict, Any, Callable

# --- Funções de Potencial ---
# Estas funções definem o "campo de força" moral do ambiente.

def saint_villain_potential(z: complex, action_type: str, params: Dict[str, Any]) -> complex:
    """
    Define o campo de potencial que atrai/repele agentes dos polos 'santo' e 'vilão'.
   
    """
    z_saint = params.get('z_saint', 0.95 + 0j)  # Polo do "santo"
    z_villain = params.get('z_villain', -0.95 + 0j) # Polo do "vilão"
    attraction_strength = params.get('attraction', 1.0)
    
    if action_type == 'good':
        # Gradiente que "puxa" o agente em direção ao polo santo
        return -attraction_strength * (z - z_saint)
    elif action_type == 'bad':
        # Gradiente que "puxa" o agente em direção ao polo vilão
        return -attraction_strength * (z - z_villain)
    else: # Ação neutra ou desconhecida
        # Gradiente que faz a reputação decair de volta para o centro (neutralidade)
        return attraction_strength * z

# --- Classe Principal do Sistema de Reputação ---

class HyperbolicReputationSystem:
    """
    Gerencia os estados de reputação (karma) dos agentes no disco de Poincaré.
   
    """
    def __init__(self, potential_func: Callable, potential_params: Dict, dt: float = 0.1, noise_scale: float = 0.01):
        """
        Inicializa o sistema de reputação.
        """
        self.agent_karma: Dict[Any, complex] = {}
        self.potential_func = potential_func
        self.potential_params = potential_params
        self.dt = dt
        self.noise_scale = noise_scale

    def add_agent(self, agent_id: Any, initial_z: complex = 0j):
        """Adiciona um novo agente ao sistema, começando no centro (neutro)."""
        self.agent_karma[agent_id] = initial_z

    def get_karma_state(self, agent_id: Any) -> complex:
        """Retorna a posição complexa (z) do agente no disco."""
        return self.agent_karma.get(agent_id, 0j)

    def hyperbolic_distance(self, agent1_id: Any, agent2_id: Any) -> float:
        """
        Calcula a "distância moral" (distância hiperbólica) entre as reputações de dois agentes.
        """
        z1 = self.agent_karma.get(agent1_id, 0j)
        z2 = self.agent_karma.get(agent2_id, 0j)
        
        numerator = 2 * np.abs(z1 - z2)**2
        denominator = (1 - np.abs(z1)**2) * (1 - np.abs(z2)**2)
        
        # Adiciona um pequeno epsilon para evitar erros de ponto flutuante ou divisão por zero
        if denominator <= 1e-9:
            return np.inf
            
        return np.arccosh(1 + numerator / denominator)

    def update_karma(self, agent_id: Any, action_type: str):
        """
        Atualiza o karma de um agente com base em uma ação ('good', 'bad'),
        usando o método de Euler-Maruyama para a Equação Diferencial Estocástica.
        """
        z = self.agent_karma.get(agent_id)
        if z is None:
            return

        # 1. Termo de deriva: gradiente do potencial no ponto z
        grad = self.potential_func(z, action_type, self.potential_params)
        
        # 2. Termo de difusão: ruído estocástico (movimento Browniano)
        noise = self.noise_scale * (np.random.randn() + 1j * np.random.randn())
        
        # 3. Passo de Euler-Maruyama para a SDE
        z_new = z - grad * self.dt + noise * np.sqrt(self.dt)
        
        # 4. Projeção: garante que o agente permaneça dentro do disco de Poincaré
        if np.abs(z_new) >= 1:
            z_new = z_new / np.abs(z_new) * 0.9999 # Projeta de volta para a borda
            
        self.agent_karma[agent_id] = z_new