# buriedbrains/reputation.py
import numpy as np
from typing import Dict, Any, Callable

# --- Funções de Drift (Física do Movimento) ---

def saint_villain_drift(z: complex, action_type: str, params: Dict[str, Any]) -> complex:
    """
    Calcula a direção e velocidade da mudança de reputação.
    CORRIGIDO: 'good' atrai para z_saint, 'bad' atrai para z_villain.
    """
    # Configuração dos Polos (Padrão: Santo na Direita/Cima, Vilão na Esquerda/Baixo)
    # Mas respeita o que vier do env.py (0.95 e -0.95)
    z_saint = params.get('z_saint', 0.95 + 0j) 
    z_villain = params.get('z_villain', -0.95 + 0j)
    
    speed = params.get('speed', 8.0)
    
    target = 0j # Padrão: Centro (Decaimento)

    if action_type == 'good':
        # Ação Boa: O alvo é o Polo Santo
        target = z_saint
        
    elif action_type == 'bad' or action_type == 'evil':
        # Ação Ruim: O alvo é o Polo Vilão
        target = z_villain
        
    else: 
        # Ação Neutra ou Tempo passando: Decaimento para o centro (esquecimento)
        decay = params.get('decay', 0.1)
        # O vetor é contrário à posição atual (puxa para 0,0)
        return -decay * z

    # Calcula o vetor de direção: (Alvo - Posição Atual)
    direction = target - z
    return speed * direction

# --- Classe Principal do Sistema de Reputação ---

class HyperbolicReputationSystem:
    """
    Gerencia o karma no disco de Poincaré usando Equações Diferenciais Estocásticas.
    """
    def __init__(self, potential_func: Callable, potential_params: Dict, dt: float = 0.1, noise_scale: float = 0.01):
        self.agent_karma: Dict[Any, complex] = {}
        self.drift_func = potential_func
        self.params = potential_params
        self.dt = dt
        self.noise_scale = noise_scale

    def add_agent(self, agent_id: Any, initial_z: complex = 0j):
        """Adiciona agente no sistema."""
        self.agent_karma[agent_id] = initial_z

    def get_karma_state(self, agent_id: Any) -> complex:
        """Retorna a coordenada complexa atual."""
        return self.agent_karma.get(agent_id, 0j)

    def update_karma(self, agent_id: Any, action_type: str):
        """
        Aplica a física de movimento hiperbólico.
        """
        z = self.agent_karma.get(agent_id)
        if z is None:
            # Se o agente não existe (ex: bug de init), adiciona no centro
            z = 0j
            self.agent_karma[agent_id] = z

        # 1. Calcula vetor de intenção (Drift)
        drift = self.drift_func(z, action_type, self.params)
        
        # 2. Fator de Escala Riemanniano (Métrica de Poincaré)
        # Faz com que movimento perto da borda seja mais "custoso/lento" em coordenadas euclidianas
        # impedindo que saia do disco unitário facilmente.
        poincare_scale = (1 - np.abs(z)**2)
        
        # 3. Ruído Estocástico
        noise = self.noise_scale * (np.random.randn() + 1j * np.random.randn())
        
        # 4. Atualização (Euler-Maruyama)
        step = (drift * poincare_scale * self.dt) + (noise * poincare_scale * np.sqrt(self.dt))
        z_new = z + step

        # 5. Boundary Check (Segurança Final)
        # Garante que |z| < 1 estrito
        abs_z = np.abs(z_new)
        if abs_z >= 0.995:
            z_new = z_new / abs_z * 0.995
            
        self.agent_karma[agent_id] = z_new