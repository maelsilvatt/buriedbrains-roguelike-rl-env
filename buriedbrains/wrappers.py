# buriedbrains/wrappers.py
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

class SymmetricSelfPlayWrapper(gym.Wrapper):
    """
    Transforma o ambiente MAE (Multi-Agent) em SAE (Single-Agent) para o SB3.
    Controla o Agente 1 (aprendiz) e usa a própria política para controlar o Agente 2 (oponente).
    """
    def __init__(self, env):
        super().__init__(env)
        # O SB3 verá apenas o espaço do Agente 1
        self.action_space = env.action_space['a1']
        self.observation_space = env.observation_space['a1']
        
        self.model = None # Referência para a política 
        self.last_obs_a2 = None # Guarda a observação do oponente
        self.lstm_states_a2 = None # Guarda a memória LSTM do oponente

    def set_model(self, model):
        """Recebe o modelo que está sendo treinado para usar como oponente."""
        self.model = model

    def reset(self, seed=None, options=None):
        # Reseta o ambiente real (MAE)
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        # Guarda o estado inicial do oponente (Agente 2)
        self.last_obs_a2 = obs_dict['a2']
        self.lstm_states_a2 = None # Reseta a memória do oponente
        
        # Retorna apenas o estado do aprendiz (Agente 1)
        return obs_dict['a1'], info_dict['a1']

    def step(self, action_a1):
        # 1. Decidir a ação do Oponente (Agente 2)
        action_a2 = 0 # Default (Wait) se não tiver modelo ainda
        
        if self.model:
            # Usa o modelo para prever a ação do A2 baseada na obs do A2
            # deterministic=False para manter a variabilidade/exploração
            action_a2, self.lstm_states_a2 = self.model.predict(
                self.last_obs_a2, 
                state=self.lstm_states_a2, 
                deterministic=False
            )
            # Em caso de ação ser array (ex: ação discreta), converte para escalar
            if isinstance(action_a2, np.ndarray):
                action_a2 = action_a2.item()

        # 2. Executar o passo no ambiente real com AMBAS as ações
        actions = {'a1': action_a1, 'a2': action_a2}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(actions)

        # 3. Atualizar o estado do oponente para o próximo passo
        self.last_obs_a2 = obs_dict['a2']
        
        # Se o A2 morreu (respawnou), precisamos resetar sua memória LSTM    
        if rew_dict['a2'] <= -200: # Penalidade de morte
             self.lstm_states_a2 = None

        # 4. Retornar os dados do Aprendiz (A1) para o SB3
        return (
            obs_dict['a1'], 
            rew_dict['a1'], 
            term_dict['a1'], # Note: term_dict['a1'] é False na morte (respawn), True só no fim do jogo
            trunc_dict['a1'], 
            info_dict['a1']
        )