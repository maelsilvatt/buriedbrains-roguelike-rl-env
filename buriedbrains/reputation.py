import numpy as np
from typing import Dict, Any, Callable

# --- Funções de Drift (Deriva) ---

def saint_villain_drift(z: complex, action_type: str, params: Dict[str, Any]) -> complex:
    """
    Retorna o vetor de VELOCIDADE (Drift) desejado.
    """
    # ALINHAMENTO COM O PLOT:
    # Santo = Esquerda (-0.95), Vilão = Direita (+0.95)
    z_saint = params.get('z_saint', -0.95 + 0j) 
    z_villain = params.get('z_villain', 0.95 + 0j)
    
    speed = params.get('speed', 1.0) # Velocidade base
    
    if action_type == 'good':
        # Vetor do ponto atual (z) em direção ao santo
        # Direção = Destino - Origem
        direction = z_saint - z 
        return speed * direction
        
    elif action_type == 'bad':
        direction = z_villain - z
        return speed * direction
        
    else: # Neutro: decai para o centro (0,0)
        # Direção = 0 - z = -z
        decay = params.get('decay', 0.5)
        return -decay * z

# --- Classe Principal ---

class HyperbolicReputationSystem:
    def __init__(self, drift_func: Callable, params: Dict, dt: float = 0.1, noise_scale: float = 0.05):
        self.agent_karma: Dict[Any, complex] = {}
        self.drift_func = drift_func
        self.params = params
        self.dt = dt
        self.noise_scale = noise_scale

    def add_agent(self, agent_id: Any, initial_z: complex = 0j):
        self.agent_karma[agent_id] = initial_z

    def get_karma_state(self, agent_id: Any) -> complex:
        return self.agent_karma.get(agent_id, 0j)

    def hyperbolic_distance(self, agent1_id: Any, agent2_id: Any) -> float:
        z1 = self.agent_karma.get(agent1_id, 0j)
        z2 = self.agent_karma.get(agent2_id, 0j)
        
        # Fórmula da métrica de Poincaré
        # d(u, v) = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2)))
        numerator = 2 * (np.abs(z1 - z2)**2)
        denominator = (1 - np.abs(z1)**2) * (1 - np.abs(z2)**2)
        
        if denominator <= 1e-9: return 100.0 # Valor alto de fallback (infinito)
            
        val = 1 + numerator / denominator
        # Segurança numérica para arccosh (domínio >= 1)
        return np.arccosh(max(1.0, val))

    def update_karma(self, agent_id: Any, action_type: str):
        z = self.agent_karma.get(agent_id)
        if z is None: return

        # 1. Calcula o vetor de intenção (Drift)
        drift = self.drift_func(z, action_type, self.params)
        
        # 2. Fator de Escala Riemanniano (CRUCIAL PARA POINCARÉ)
        # Perto do centro (z=0), scale=1. Perto da borda (z=0.9), scale fica minúsculo.
        # Isso impede que o agente "atravesse" a borda ou ande rápido demais nos extremos.
        poincare_scale = (1 - np.abs(z)**2)
        
        # 3. Ruído
        # O ruído também deve ser escalado, senão o ruído joga ele pra fora do disco na borda
        noise = self.noise_scale * (np.random.randn() + 1j * np.random.randn())
        
        # 4. Equação Diferencial Estocástica (Euler-Maruyama adaptado)
        # z_new = z + (drift * scale * dt) + (noise * scale * sqrt(dt))
        
        step_drift = drift * poincare_scale * self.dt
        step_noise = noise * poincare_scale * np.sqrt(self.dt)
        
        z_new = z + step_drift + step_noise

        # 5. Boundary Check (Projeção)
        # Mesmo com o scale, erros numéricos podem ocorrer.
        abs_z = np.abs(z_new)
        if abs_z >= 0.999:
            z_new = z_new / abs_z * 0.999
            
        self.agent_karma[agent_id] = z_new