import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

class IslandWrapper(gym.Wrapper):
    """
    Transforma um ambiente Multi-Agent (MAE) em um 'Ambiente de Ilha'.
    Retorna observações empilhadas (Agentes, Features).
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agent_ids = env.agent_ids
        self.num_agents = len(self.agent_ids)
        
        # Pega o espaço do primeiro agente
        single_obs_space = env.observation_space[self.agent_ids[0]]
        single_action_space = env.action_space[self.agent_ids[0]]
        
        # Define o espaço de observação da Ilha inteira (16, 198)
        self.observation_space = spaces.Box(
            low=single_obs_space.low[0], 
            high=single_obs_space.high[0],
            shape=(self.num_agents, *single_obs_space.shape),
            dtype=single_obs_space.dtype
        )
        
        # Ação múltipla
        if isinstance(single_action_space, spaces.Discrete):
            # Se for Discreto, vira MultiDiscrete([4, 4, 4...])
            self.action_space = spaces.MultiDiscrete([single_action_space.n] * self.num_agents)
        else:
            # Se for Box, vira Box(16, Acoes)
            self.action_space = spaces.Box(
                low=single_action_space.low[0],
                high=single_action_space.high[0],
                shape=(self.num_agents, *single_action_space.shape),
                dtype=single_action_space.dtype
            )

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        # Garante a ordem correta dos IDs ao empilhar
        obs_list = [obs_dict[aid] for aid in self.agent_ids]
        stacked_obs = np.stack(obs_list)
        
        return stacked_obs, {} 

    def step(self, action):
        # action entra como array (16,). Convertemos para dict.
        action_dict = {
            agent_id: action[i] 
            for i, agent_id in enumerate(self.agent_ids)
        }
        
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        global_done = term_dict.get('__all__', False) or trunc_dict.get('__all__', False)
        
        # Empilha retornos
        obs_list = [obs_dict[aid] for aid in self.agent_ids]
        rew_list = [rew_dict[aid] for aid in self.agent_ids]
        
        stacked_obs = np.stack(obs_list)
        stacked_rew = np.array(rew_list, dtype=np.float32)
        
        infos = [info_dict[aid] for aid in self.agent_ids]
        
        if global_done:
            new_obs_dict, _ = self.env.reset()
            stacked_obs = np.stack([new_obs_dict[aid] for aid in self.agent_ids])
            for i, aid in enumerate(self.agent_ids):
                infos[i]['terminal_observation'] = obs_dict[aid]

        return stacked_obs, stacked_rew, global_done, False, {'sub_infos': infos}

class FlattenParallelWrapper(VecEnvWrapper):
    """
    Pega um SubprocVecEnv que retorna dados de 'Ilhas' (N_Envs, N_Agents, ...)
    e achata tudo para (N_Total_Agents, ...).
    """
    def __init__(self, venv):
        self.num_islands = venv.num_envs
        
        # Hack para descobrir o tamanho real
        island_obs_shape = venv.observation_space.shape
        self.agents_per_island = island_obs_shape[0] 
        single_agent_shape = island_obs_shape[1:] 
        
        total_agents = self.num_islands * self.agents_per_island
        
        super().__init__(venv)
        
        self.num_envs = total_agents
        
        # Configura espaços achatados
        self.observation_space = spaces.Box(
            low=venv.observation_space.low[0], 
            high=venv.observation_space.high[0],
            shape=single_agent_shape,          
            dtype=venv.observation_space.dtype
        )
        
        base_act = venv.action_space
        if isinstance(base_act, spaces.MultiDiscrete):
             self.action_space = spaces.Discrete(base_act.nvec[0])
        elif isinstance(base_act, spaces.Box):
             self.action_space = spaces.Box(
                 low=base_act.low[0],
                 high=base_act.high[0],
                 shape=base_act.shape[1:],
                 dtype=base_act.dtype
             )

    def reset(self):
        obs = self.venv.reset() 
        return obs.reshape(-1, *obs.shape[2:]) 
    
    def step_async(self, actions):
        # O SB3 manda 'actions' com shape (Total_Agents, ...) ex: (64,)
        # O SubprocVecEnv espera (Num_Ilhas, Agents_Por_Ilha, ...) ex: (4, 16)
        
        # Remodela o array linear para o formato de ilhas
        reshaped_actions = actions.reshape(self.num_islands, self.agents_per_island, *actions.shape[1:])
        
        # Manda os pacotes corretos para o Subproc
        self.venv.step_async(reshaped_actions)    

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        
        obs = obs.reshape(-1, *obs.shape[2:])
        rews = rews.flatten()
        dones_expanded = np.repeat(dones, self.agents_per_island)
        
        flat_infos = []
        for i in range(self.num_islands):
            island_infos = infos[i].get('sub_infos', [])
            flat_infos.extend(island_infos)
            
        return obs, rews, dones_expanded, flat_infos