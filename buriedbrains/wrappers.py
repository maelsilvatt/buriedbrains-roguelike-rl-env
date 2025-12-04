import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

class SharedPolicyVecEnv(VecEnv):
    """
    Transforma um ambiente Multi-Agent (MAE) em um VecEnv do SB3.
    
    O SB3 enxergará cada AGENTE como um "Ambiente" separado.
    Isso permite que a mesma política controle todos os agentes e aprenda
    com a experiência de TODOS eles simultaneamente (Parameter Sharing).
    """
    def __init__(self, env):
        self.env = env
        self.agent_ids = env.agent_ids
        self.num_agents = len(self.agent_ids)
        
        # Define o espaço de observação e ação
        # Pega o espaço do primeiro agente como modelo
        observation_space = env.observation_space[self.agent_ids[0]]
        action_space = env.action_space[self.agent_ids[0]]
        
        # Inicializa a classe pai (VecEnv) dizendo que temos 'num_agents' ambientes
        super().__init__(self.num_agents, observation_space, action_space)
        
        # Buffers internos
        self.last_obs = None
        self.actions = None

    def seed(self, seed=None):
        """
        Define a semente para o ambiente global subjacente.
        """
                
        return self.env.reset(seed=seed)

    def reset(self):
        """
        Reseta o ambiente real e retorna as observações de TODOS os agentes empilhadas.
        """
        obs_dict, info_dict = self.env.reset()
        
        # Converte Dict de Obs -> Array Numpy [Agente1_Obs, Agente2_Obs, ...]
        obs_list = [obs_dict[aid] for aid in self.agent_ids]
        self.last_obs = np.stack(obs_list)
        
        return self.last_obs

    def step_async(self, actions):
        # SB3 manda as ações, guardamos para usar no step_wait
        self.actions = actions

    def step_wait(self):
        """
        Executa o passo real no ambiente.
        """
        # Converte Array de Ações (SB3) -> Dicionário (Env)
        action_dict = {
            agent_id: self.actions[i] 
            for i, agent_id in enumerate(self.agent_ids)
        }
        
        # Passo no Ambiente Real
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        # Converte Dicionários -> Arrays para o SB3
        obs_list = []
        rew_list = []
        done_list = []
        infos_list = []
        
        for i, agent_id in enumerate(self.agent_ids):
            obs_list.append(obs_dict[agent_id])
            rew_list.append(rew_dict[agent_id])
            
            # Combina terminated e truncated
            is_done = term_dict[agent_id] or trunc_dict[agent_id]
            done_list.append(is_done)
            
            # Info extra necessária para o SB3
            info = info_dict[agent_id].copy()
            if is_done:
                info['terminal_observation'] = obs_dict[agent_id]
            infos_list.append(info)

        self.last_obs = np.stack(obs_list)
        rewards = np.array(rew_list, dtype=np.float32)
        dones = np.array(done_list, dtype=bool)
        
        # Auto-reset global se necessário
        if term_dict.get('__all__', False) or trunc_dict.get('__all__', False):
            new_obs_dict, _ = self.env.reset()
            for i, agent_id in enumerate(self.agent_ids):
                self.last_obs[i] = new_obs_dict[agent_id]
                infos_list[i]['terminal_observation'] = obs_dict[agent_id]

        return self.last_obs, rewards, dones, infos_list

    def close(self):
        pass        

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Método obrigatório do VecEnv. Retorna False pois não estamos usando wrappers do SB3 internamente."""
        return [False] * self.num_agents

    def set_attr(self, attr_name, value, indices=None):
        """Método obrigatório do VecEnv. Seta atributo no ambiente base."""
        # Como todos os agentes compartilham o MESMO ambiente, setamos no self.env
        setattr(self.env, attr_name, value)

    def get_attr(self, attr_name, indices=None):
        """Retorna o valor de um atributo para cada 'ambiente' (agente)."""
        val = getattr(self.env, attr_name)
        return [val] * self.num_agents

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Chama um método do ambiente base."""
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_agents)]
    
    def seed(self, seed=None):
        self.env.reset(seed=seed)