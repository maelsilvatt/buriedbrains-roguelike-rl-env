# buriedbrains/reputation.py
import numpy as np
from typing import Dict, Any, Callable

# --- Funções de Drift (Movimento) ---

def saint_villain_drift(z: complex, action_type: str, params: Dict[str, Any]) -> complex:
    """
    Retorna o vetor de VELOCIDADE (Drift) baseado na ação.
    (Renomeado para bater com a sua chamada na instanciação)
    """
    # Configuração dos Polos
    # Santo = Esquerda (-0.95), Vilão = Direita (+0.95)
    z_saint = params.get('z_saint', -0.95 + 0j) 
    z_villain = params.get('z_villain', 0.95 + 0j)
    
    speed = params.get('speed', 1.0) # Intensidade do movimento
    
    if action_type == 'good':
        # Vetor: do ponto atual PARA o santo
        direction = z_saint - z 
        return speed * direction
        
    elif action_type == 'bad':
        # Vetor: do ponto atual PARA o vilão
        direction = z_villain - z
        return speed * direction
        
    else: 
        # Neutro: Decai suavemente para o centro (0,0)
        decay = params.get('decay', 0.5)
        return -decay * z

# --- Classe Principal do Sistema de Reputação ---

class HyperbolicReputationSystem:
    """
    Gerencia o karma no disco de Poincaré usando Equações Diferenciais Estocásticas.
    """
    def __init__(self, potential_func: Callable, potential_params: Dict, dt: float = 0.1, noise_scale: float = 0.01):
        """
        Inicializa o sistema com os parâmetros de física do disco.
        """
        self.agent_karma: Dict[Any, complex] = {}
                
        self.drift_func = potential_func  # A função saint_villain_drift
        self.params = potential_params    # Os parâmetros (z_saint, speed, etc)
        self.dt = dt                      # Passo de tempo (0.1)
        self.noise_scale = noise_scale    # Intensidade do ruído (0.01)

    def add_agent(self, agent_id: Any, initial_z: complex = 0j):
        """Adiciona agente no centro (ou posição inicial)."""
        self.agent_karma[agent_id] = initial_z

    def get_karma_state(self, agent_id: Any) -> complex:
        """Retorna a coordenada complexa atual."""
        return self.agent_karma.get(agent_id, 0j)

    def hyperbolic_distance(self, agent1_id: Any, agent2_id: Any) -> float:
        """
        Calcula a distância 'moral' na métrica de Poincaré.
        """
        z1 = self.agent_karma.get(agent1_id, 0j)
        z2 = self.agent_karma.get(agent2_id, 0j)
        
        numerator = 2 * (np.abs(z1 - z2)**2)
        denominator = (1 - np.abs(z1)**2) * (1 - np.abs(z2)**2)
        
        if denominator <= 1e-9: 
            return 50.0 # Valor alto de fallback
            
        val = 1 + numerator / denominator
        return np.arccosh(max(1.0, val))

    def update_karma(self, agent_id: Any, action_type: str):
        """
        Aplica a física de movimento hiperbólico.
        """
        z = self.agent_karma.get(agent_id)
        if z is None:
            return

        # 1. Calcula vetor de intenção (Drift)
        drift = self.drift_func(z, action_type, self.params)
        
        # 2. Fator de Escala Riemanniano (CRUCIAL)
        # Impede que atravesse a borda e simula a expansão do espaço.
        poincare_scale = (1 - np.abs(z)**2)
        
        # 3. Ruído Estocástico (Aleatoriedade moral)
        noise = self.noise_scale * (np.random.randn() + 1j * np.random.randn())
        
        # 4. Atualização (Euler-Maruyama)
        step_drift = drift * poincare_scale * self.dt
        step_noise = noise * poincare_scale * np.sqrt(self.dt)
        
        z_new = z + step_drift + step_noise

        # 5. Boundary Check (Segurança final)
        abs_z = np.abs(z_new)
        if abs_z >= 0.999:
            z_new = z_new / abs_z * 0.999
            
        self.agent_karma[agent_id] = z_new