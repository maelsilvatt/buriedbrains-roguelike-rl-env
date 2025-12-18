# buriedbrains/reputation.py
import numpy as np
from typing import Dict, Any, Callable

# ==========================================
# CONSTANTES DE REPUTAÇÃO
# ==========================================

# Polos de Reputação:
# Define para onde o agente é puxado. 
# Nota: Devem estar dentro do disco unitário (|z| < 1).
POLE_SAINT = 0.95 + 0j       # Ponto de máxima bondade
POLE_VILLAIN = -0.95 + 0j    # Ponto de máxima maldade
POLE_NEUTRAL = 0j            # Centro (Neutro/Esquecimento)

# Dinâmica de Mudança:
BASE_SPEED = 1.0             # Quão rápido a reputação muda ao agir
DECAY_RATE = 0.1             # Quão rápido o mundo "esquece" (retorno ao centro)

# Física Estocástica:
PHYSICS_DT = 0.1             # Passo de tempo (Time Step) da equação diferencial
PHYSICS_NOISE_SCALE = 0.01   # "Tremor" na reputação (incerteza social)

# Segurança:
# O limite absoluto do disco de Poincaré é 1.0. 
# Chegar em 1.0 ou passar gera divisão por zero ou infinitos na métrica.
# Usamos 0.995 como "muro invisível" de segurança.
MAX_DISK_RADIUS = 0.995      

# ==========================================
# LÓGICA DO SISTEMA
# ==========================================
def saint_villain_drift(z: complex, action_type: str, params: Dict[str, Any]) -> complex:
    """
    Calcula a direção e velocidade da mudança de reputação.    
    """
    # Configuração dos Polos (permite override via params, senão usa constantes)
    z_saint = params.get('z_saint', POLE_SAINT) 
    z_villain = params.get('z_villain', POLE_VILLAIN)
    
    speed = params.get('speed', BASE_SPEED)
    
    target = POLE_NEUTRAL # Padrão: Centro (Decaimento)

    if action_type == 'good':
        # Ação Boa: O alvo é o Polo Santo
        target = z_saint
        
    elif action_type == 'bad' or action_type == 'evil':
        # Ação Ruim: O alvo é o Polo Vilão
        target = z_villain
        
    else: 
        # Ação Neutra ou Tempo passando: Decaimento para o centro (esquecimento)
        decay = params.get('decay', DECAY_RATE)
        # O vetor é contrário à posição atual (puxa para 0,0)
        return -decay * z

    # Calcula o vetor de direção: (Alvo - Posição Atual)
    direction = target - z
    return speed * direction

class HyperbolicReputationSystem:
    """
    Gerencia o karma no disco de Poincaré usando Equações Diferenciais Estocásticas.
    """
    def __init__(
        self, 
        potential_func: Callable, 
        potential_params: Dict, 
        dt: float = PHYSICS_DT, 
        noise_scale: float = PHYSICS_NOISE_SCALE
    ):
        self.agent_karma: Dict[Any, complex] = {}
        self.drift_func = potential_func
        self.params = potential_params
        self.dt = dt
        self.noise_scale = noise_scale

    def add_agent(self, agent_id: Any, initial_z: complex = POLE_NEUTRAL):
        """Adiciona agente no sistema."""
        self.agent_karma[agent_id] = initial_z

    def get_karma_state(self, agent_id: Any) -> complex:
        """Retorna a coordenada complexa atual."""
        return self.agent_karma.get(agent_id, POLE_NEUTRAL)

    def update_karma(self, agent_id: Any, action_type: str):
        """
        Aplica a física de movimento hiperbólico.
        """
        z = self.agent_karma.get(agent_id)
        if z is None:
            # Se o agente não existe (ex: bug de init), adiciona no centro
            z = POLE_NEUTRAL
            self.agent_karma[agent_id] = z

        # Calcula vetor de intenção (Drift)
        drift = self.drift_func(z, action_type, self.params)
        
        # Fator de Escala Riemanniano (Métrica de Poincaré)
        # Faz com que movimento perto da borda seja mais "custoso/lento" em coordenadas euclidianas
        # impedindo que saia do disco unitário facilmente.
        poincare_scale = (1 - np.abs(z)**2)
        
        # Ruído Estocástico
        noise = self.noise_scale * (np.random.randn() + 1j * np.random.randn())
        
        # Atualização (Euler-Maruyama)
        step = (drift * poincare_scale * self.dt) + (noise * poincare_scale * np.sqrt(self.dt))
        z_new = z + step
        
        # Garante que |z| < 1 estrito
        abs_z = np.abs(z_new)
        if abs_z >= MAX_DISK_RADIUS:
            z_new = z_new / abs_z * MAX_DISK_RADIUS
            
        self.agent_karma[agent_id] = z_new