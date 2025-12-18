# buriedbrains/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
import networkx as nx
from collections import deque

# Importando módulos internos
from . import progression
from . import combat
from . import content_generation
from . import map_generation
from . import reputation
from . import skill_encoder
from . import item_encoder
from . import effect_encoder
from .gerador_nomes import GeradorNomes

class BuriedBrainsEnv(gym.Env):
    """
    Ambiente principal do BuriedBrains, compatível com a interface Gymnasium.
    """
    def __init__(self, 
                max_episode_steps: int = 15000, 
                 max_floors: int = 500,
                 max_level: int = 500,
                 budget_multiplier: float = 1.0,
                 guarantee_enemy: bool = False, 
                 verbose: int = 0,
                 sanctum_floor: int = 20,
                 num_agents: int = 2,
                 enable_logging_buffer=True,
                 seed: int = 42):
        super().__init__()        
        
        # Carregando catálogos
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        with open(os.path.join(data_path, 'skill_and_effects.yaml'), 'r', encoding='utf-8') as f:
            skill_data = yaml.safe_load(f)
        with open(os.path.join(data_path, 'equipment_catalog.yaml'), 'r', encoding='utf-8') as f:
            equipment_data = yaml.safe_load(f)
        with open(os.path.join(data_path, 'enemies_and_events.yaml'), 'r', encoding='utf-8') as f:
            enemy_data = yaml.safe_load(f)
            
        self.catalogs = {
            'skills': skill_data['skill_catalog'], 
            'effects': skill_data['effect_ruleset'],
            'equipment': equipment_data['equipment_catalog'], 
            'enemies': enemy_data['pools']['enemies'],
            'enemy_tiers': enemy_data.get('enemy_tiers', {}),
            'room_effects': enemy_data['pools']['room_effects'], 
            'events': enemy_data['pools']['events']
        }        

        self.artifact_catalog = {
            k: v for k, v in self.catalogs.get('equipment', {}).items() 
            if v.get('type') == 'Artifact'
        }
        self.equipment_catalog_no_artifacts = {
            k: v for k, v in self.catalogs.get('equipment', {}).items() 
            if v.get('type') in ['Weapon', 'Armor']
        }
        self.catalogs['artifacts'] = self.artifact_catalog

        # Instancia o encoder de skills
        self.skill_encoder = skill_encoder.SkillEncoder()

        # Instancia o encoder de itens
        self.item_encoder = item_encoder.ItemEncoder(self.catalogs['equipment'])

        # Instancia o encoder de efeitos de sala
        self.effect_encoder = effect_encoder.EffectEncoder()

        # Parâmetros globais
        self.enable_logging_buffer = enable_logging_buffer
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.verbose = verbose 
        self.max_floors = max_floors
        self.max_level = max_level         
        self.budget_multiplier = budget_multiplier
        self.guarantee_enemy = guarantee_enemy        
        self.rarity_map = {'Common': 0.25, 'Rare': 0.5, 'Epic': 0.75, 'Legendary': 1.0}
        self.sanctum_floor = sanctum_floor
        self.num_agents = num_agents 
        self.seed = seed

        self.pvp_durations = []
        self.current_pvp_timer = 0
        
        for catalog_name, catalog_data in self.catalogs.items():
            if isinstance(catalog_data, dict):
                for name, data in catalog_data.items():
                    if isinstance(data, dict):
                        data['name'] = name

        # Definições multiagente
        # Gera IDs: 'a1', 'a2', 'a3', ... até num_agents
        self.agent_ids = [f"a{i+1}" for i in range(self.num_agents)]
            
        self.MAX_NEIGHBORS = 4 

        self.pvp_sessions = {}
        self.arena_entry_steps = {} # Rastreia quando o agente entrou no Santuário
        
        # Estado persistente dinâmico para N agentes
        self.arena_interaction_state = {
            agent_id: {'offered_peace': False}
            for agent_id in self.agent_ids
        }

        # Espaços de ação e observação
        # Ações: 0-3 (Skills), 4 (Equipar), 5 (Social), 6-9 (Mover)
        # 0: Quick Strike, 1: Heavy Blow, 2: Stone Shield, 3: Wait
        # 4: Equipar Item
        # 5: Soltar/Pegar Artefato (Social)
        # 6: Mover Vizinho 0
        # 7: Mover Vizinho 1
        # 8: Mover Vizinho 2
        # 9: Mover Vizinho 3
        ACTION_SHAPE = 10 
                
        # DEFINIÇÃO DO ESPAÇO DE OBSERVAÇÃO (OBS_SHAPE = 182)
        #
        # 1. BLOCO DE SKILLS (0-39) [4 Slots * 10 Features]
        #    Cada slot tem: [9 features do Encoder] + [1 Cooldown Ratio]
        #
        # 2. BLOCO PRÓPRIO (40-42) [3 Estados]
        #    40: HP Ratio
        #    41: Level Ratio
        #    42: EXP Ratio
        #
        # 3. BLOCO CONTEXTO PvE (43-49) [7 Estados]
        #    43: In Combat?
        #    44: Item/Evento?
        #    45: Sala Vazia?
        #    46: Enemy HP Ratio
        #    47: Enemy Level Ratio
        #    48: Equip no chão?
        #    49: Raridade Equip Chão
        #
        # 4. BLOCO SOCIAL/PvP (50-59) [10 Estados]
        #    50: In Arena?
        #    51: Other Agent Present?
        #    52: Other HP Ratio
        #    53: Level Diff
        #    54: Other Karma (Real)
        #    55: Other Karma (Imag)
        #    56: Artifact Floor?
        #    57: Artifact Floor Rarity
        #    58: My Artifact Rarity
        #    59: In PvP Combat?
        #
        # 5. BLOCO MOVIMENTO & AMBIENTE (60-124) [65 Estados]
        #    Agora inclui o Nó Atual + 4 Vizinhos (5 Blocos * 13 Features)
        #    Layout: [Current, Neighbor_1, Neighbor_2, Neighbor_3, Neighbor_4]
        #    Para cada um (13 Features):
        #       [0]: Valid Node?
        #       [1]: Enemy Present?
        #       [2]: Reward (Item/Event/Exit)?
        #       [3]: Danger Tier (0.33=Common, 0.66=Elite, 1.0=Boss)
        #       [4-12]: Efeito de Sala (Encoder de 9 tipos)
        #
        # 6. BLOCO FINAL (125-130) [6 Estados]
        #    (Deslocado para 125)
        #    125: Self Weapon Rarity
        #    126: Self Armor Rarity
        #    127: Other Gear Score
        #    128: Other Just Dropped?
        #    129: Other Skipped Attack?
        #    130: Door Open/Exit?
        #
        # 7. BLOCO DETALHADO DE ITENS (131-181) [51 Estados]
        #    (Deslocado para 131)
        #    3 Slots * 17 Features (Atualizado: sem Fear/Shock, com Sunder)
        #    131-147: Weapon Embedding
        #    148-164: Armor Embedding
        #    165-181: Artifact Embedding   
        OBS_SHAPE = (182,)

        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(ACTION_SHAPE) for agent_id in self.agent_ids
        })
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(low=-1.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32) 
            for agent_id in self.agent_ids
        })

        # Variáveis de estado
        self.agent_states = {}
        self.agent_names = {}
        self.graphs = {}
        self.current_nodes = {}
        self.current_floors = {}
        self.nodes_per_floor_counters = {}
        self.combat_states = {}
        self.current_episode_logs = {}
        self.enemies_defeated_this_episode = {}
        self.invalid_action_counts = {}
        self.last_milestone_floors = {}
        self.damage_dealt_this_episode = {}
        self.equipment_swaps_this_episode = {}
        self.skill_upgrades_this_episode = {}
        self.death_cause = {}
        self.arena_encounters_this_episode = {}
        self.pvp_combats_this_episode = {}
        self.bargains_succeeded_this_episode = {}
        self.bargains_trade_this_episode = {}
        self.bargains_toll_this_episode = {}
        self.cowardice_kills_this_episode = {}
        self.betrayals_this_episode = {}
        self.pve_combat_durations = {}
        self.pvp_combat_durations = {}
        self.karma_history = {}
        self.max_floor_reached_this_episode = {}
        self.previous_nodes = {} # Rastreia nós passados que o agente andou no Santuário
        self.arena_entry_steps = {} # Rastreia quando o agente entrou no Santuário
        self.pve_return_floor = {} # Salva o andar anterior para o agente retornar após sair do Santuário (sem teleporte)                                
        self.agents_in_arena = set()                
        self.matchmaking_queue = [] # Fila de espera [agent_id, agent_id, ...]
        self.active_matches = {}    # Para rastrear combates PvP ativos {match_id: (agent1_id, agent2_id)}
        self.arena_instances = {}   # Para saber em qual zona estão os agentes        

        # Reputação
        potential_params = {
            'z_saint': 0.95 + 0j, 
            'z_villain': -0.95 + 0j, 
            'attraction': 0.5 
        }
        self.reputation_system = reputation.HyperbolicReputationSystem(
            potential_func=reputation.saint_villain_drift,
            potential_params=potential_params,
            dt=0.1, 
            noise_scale=0.01 
        )

    def _generate_and_populate_successors(self, agent_id: str, parent_node: str):
        """
        Orquestra a geração de nós sucessores (para P-Zone).
        """
        
        # Cria a topologia
        graph = self.graphs[agent_id]
        counters = self.nodes_per_floor_counters[agent_id]
        
        new_node_names, updated_counters = map_generation.generate_progression_successors(
            graph, 
            parent_node, 
            counters
        )
        
        self.nodes_per_floor_counters[agent_id] = updated_counters
        
        # Iterar e popular cada nó recém-criado
        for node_name in new_node_names:
            next_floor = graph.nodes[node_name].get('floor', 0)
                        
            # Se o próximo andar for Arena, NÃO gera inimigos nem loot PvE.
            if next_floor % self.sanctum_floor == 0:
                content = {
                    'enemies': [],      
                    'events': [],
                    'items': [],
                    'room_effects': []  
                }
            else:
                # Se não for santuário...                            
                
                content = content_generation.generate_room_content(
                    catalogs=self.catalogs,
                    current_floor=next_floor,
                    budget_multiplier=self.budget_multiplier,
                    guarantee_enemy=self.guarantee_enemy
                )

            self.graphs[agent_id].nodes[node_name]['content'] = content
            
            # Logar os resultados
            enemy_log = content.get('enemies', []) or ['Nenhum']
            event_outcome_log = content.get('events', []) or ['Nenhum']
            item_generated_log = content.get('items', []) or ['Nenhum']
            effect_log = content.get('room_effects', []) or ['Nenhum']

            self._log(agent_id, f"[PVE] {self.agent_names[agent_id]} diante de '{node_name}' (Andar {next_floor}) gerada com: "
                        f"Inimigo: {enemy_log[0]}, "
                        f"Evento: {event_outcome_log[0]}, "
                        f"Item Gerado: {item_generated_log[0]}, "
                        f"Efeito: {effect_log[0]}")

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Coleta e retorna a observação de 136 estados
        """        
        obs = np.full(self.observation_space[agent_id].shape, -1.0, dtype=np.float32) 
        
        if agent_id not in self.agent_states:
            return obs

        agent = self.agent_states[agent_id]
                
        # BLOCO DE SKILLS (Índices 0 a 39) - [4 Slots * 10 Features]        
        # Cada skill vira um vetor de 10 números: 
        # [Dano, MaxCD, Tag1, Tag2...Tag7, CurrentCD_Ratio]
        
        skills_vector = []
        for i in range(4):
            # Pega o nome da skill no slot 'i' (do deck do agente)
            skill_name = agent['skills'][i]
            
            # Pega os dados estáticos (Dano, Tipo, etc) e encoda
            skill_data = self.catalogs['skills'].get(skill_name, {})
            embedding = self.skill_encoder.encode(skill_data) # Retorna 9 floats
            
            # Pega o estado dinâmico (Cooldown Atual)
            current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
            max_cd = skill_data.get('cd', 1)
            norm_current_cd = (current_cd / max_cd) if max_cd > 0 else 0.0
            
            full_skill_data = np.append(embedding, norm_current_cd)
            skills_vector.extend(full_skill_data)

        # INJETA NO OBS (0 a 39)
        obs[0:40] = np.array(skills_vector, dtype=np.float32)        
        
        # Contexto
        current_node_id = self.current_nodes.get(agent_id) 
        is_in_arena = agent_id in self.arena_instances
        
        # Grafo
        if is_in_arena:
            current_graph_to_use = self.arena_instances[agent_id]
        else:
            current_graph_to_use = self.graphs.get(agent_id)

        # Vizinhos
        neighbors = []
        if current_graph_to_use and current_node_id and current_graph_to_use.has_node(current_node_id):            
            try:
                neighbors = list(current_graph_to_use.neighbors(current_node_id))
                neighbors.sort() 
            except: pass

        other_agent_id = self.active_matches.get(agent_id)
        room_content = current_graph_to_use.nodes[current_node_id].get('content', {}) if current_graph_to_use and current_node_id else {}
        pve_combat_state = self.combat_states.get(agent_id)

        # Bloco Próprio (40-42)
        obs[40] = (agent['hp'] / agent['max_hp']) * 2 - 1 if agent['max_hp'] > 0 else -1.0
        obs[41] = (agent['level'] / self.max_level) * 2 - 1
        obs[42] = (agent['exp'] / agent['exp_to_level_up']) if agent['exp_to_level_up'] > 0 else 0.0        

        # Bloco PvE (43-49)        
        idx_pve = 43
        
        enemy_in_combat = pve_combat_state.get('enemy') if pve_combat_state else None
        items_on_floor = room_content.get('items', [])
        events_in_room = room_content.get('events', [])
        
        if enemy_in_combat:
            obs[idx_pve + 0] = 1.0 # In Combat
            obs[idx_pve + 3] = (enemy_in_combat['hp'] / enemy_in_combat['max_hp']) * 2 - 1 if enemy_in_combat['max_hp'] > 0 else -1.0
            obs[idx_pve + 4] = (enemy_in_combat.get('level', 1) / self.max_level) * 2 - 1
        elif items_on_floor or (events_in_room and 'None' not in events_in_room):
            obs[idx_pve + 1] = 1.0 # Item/Evento
        else:
            obs[idx_pve + 2] = 1.0 # Vazio

        # Itens no Chão
        best_wep_arm_rarity = 0.0
        found_wep_arm = False
        for item_name in items_on_floor:
            if item_name in self.equipment_catalog_no_artifacts: 
                found_wep_arm = True
                rarity = self.rarity_map.get(self.catalogs['equipment'][item_name].get('rarity'), 0.0)
                if rarity > best_wep_arm_rarity: best_wep_arm_rarity = rarity
        
        if found_wep_arm:
            obs[idx_pve + 5] = 1.0
            obs[idx_pve + 6] = best_wep_arm_rarity

        # Bloco Social/PvP (50-59)
        idx_soc = 50
        obs[idx_soc + 0] = 1.0 if is_in_arena else -1.0 # In Arena
        
        if other_agent_id and is_in_arena and self.current_nodes.get(agent_id) == self.current_nodes.get(other_agent_id):
             other_agent_state = self.agent_states.get(other_agent_id)
             if other_agent_state:
                obs[idx_soc + 1] = 1.0 # Outro presente
                obs[idx_soc + 2] = (other_agent_state['hp'] / other_agent_state['max_hp']) * 2 - 1
                obs[idx_soc + 3] = (agent['level'] - other_agent_state['level']) / 100.0
                
                k = self.reputation_system.get_karma_state(other_agent_id)
                obs[idx_soc + 4] = k.real
                obs[idx_soc + 5] = k.imag
        
        # Artefatos chão
        best_art = 0.0
        found_art = False
        for item in items_on_floor:
            if item in self.artifact_catalog:
                found_art = True
                r = self.rarity_map.get(self.catalogs['equipment'][item].get('rarity'), 0.0)
                if r > best_art: best_art = r
        
        if found_art:
            obs[idx_soc + 6] = 1.0
            obs[idx_soc + 7] = best_art
            
        # Meu Artefato
        my_art = agent['equipment'].get('Artifact')
        obs[idx_soc + 8] = self.rarity_map.get(self.catalogs['equipment'][my_art].get('rarity'), 0.0) if my_art else 0.0
        
        if agent_id in self.pvp_sessions:
            obs[idx_soc + 9] = 1.0

        # Bloco Movimento (60-75)
        idx_mov = 60
        
        # O agente agora observa o próprio arredor
        
        curr_idx = idx_mov
        # Limpa o slot
        obs[curr_idx : curr_idx + 13] = 0.0 
        
        # Sala Válida ?
        obs[curr_idx + 0] = 1.0
        
        # Tem inimigo? 

        has_opp_here = (other_agent_id and self.current_nodes.get(other_agent_id) == current_node_id)
        if room_content.get('enemies') or has_opp_here:
             obs[curr_idx + 1] = 1.0
        
        # Reward
        if room_content.get('items') or any(e in room_content.get('events', []) for e in ['Treasure', 'Morbid Treasure', 'Fountain of Life']):
             obs[curr_idx + 2] = 1.0
        if is_in_arena and current_graph_to_use.nodes[current_node_id].get('is_exit') and current_graph_to_use.graph.get('meet_occurred'):
             obs[curr_idx + 2] = 1.0
             
        # Danger Tier
        d_level = 0.0
        if room_content.get('enemies'):
            ename = room_content['enemies'][0]
            tier = self.catalogs['enemies'].get(ename, {}).get('tier', 'Common')
            if tier == 'Boss': d_level = 1.0
            elif tier == 'Elite': d_level = 0.66
            else: d_level = 0.33
        elif has_opp_here:
            d_level = 0.66
        obs[curr_idx + 3] = d_level
        
        # Efeitos ambientais
        c_effects = room_content.get('room_effects', [])
        eff_name = c_effects[0] if c_effects else 'None'
        obs[curr_idx + 4 : curr_idx + 13] = self.effect_encoder.encode(eff_name)

        # Preenche os nós vizinhos        
        idx_neighbors_start = idx_mov + 13 # 73

        for i in range(self.MAX_NEIGHBORS):            
            curr_idx = idx_neighbors_start + (i * 13)
            
            # Preenche com zeros/default primeiro pra evitar lixo            
            obs[curr_idx : curr_idx + 13] = 0.0

            if i < len(neighbors) and current_graph_to_use:
                neighbor_node_id = neighbors[i]
                
                # Verifica existência
                if current_graph_to_use.has_node(neighbor_node_id):
                    n_content = current_graph_to_use.nodes[neighbor_node_id].get('content', {})
                    
                    # Features básicas
                    # Sala válida?
                    obs[curr_idx + 0] = 1.0 
                    
                    # Inimigo
                    has_opp = (other_agent_id and self.current_nodes.get(other_agent_id) == neighbor_node_id)
                    if n_content.get('enemies') or has_opp:
                        obs[curr_idx + 1] = 1.0
                        
                    # Reward
                    if n_content.get('items') or any(e in n_content.get('events', []) for e in ['Treasure', 'Morbid Treasure', 'Fountain of Life']):
                         obs[curr_idx + 2] = 1.0
                    if is_in_arena and current_graph_to_use.nodes[neighbor_node_id].get('is_exit') and current_graph_to_use.graph.get('meet_occurred'):
                         obs[curr_idx + 2] = 1.0 # Saída Destrancada conta como Reward
                         
                    # Danger Tier
                    d_level = 0.0
                    if n_content.get('enemies'):
                        ename = n_content['enemies'][0]
                        # Busca os dados do inimigo no catálogo
                        enemy_data = self.catalogs['enemies'].get(ename, {})
                        
                        # Pega o Tier (padrão 'Common' se não encontrar)
                        tier = enemy_data.get('tier', 'Common')
                        
                        if tier == 'Boss': 
                            d_level = 1.0
                        elif tier == 'Elite': 
                            d_level = 0.66                        
                        else: 
                            # Cobre Common e Fodder
                            d_level = 0.33
                            
                    elif has_opp:
                        d_level = 0.66 # Agente inimigo conta como Elite/Perigoso
                    
                    obs[curr_idx + 3] = d_level

                    # Features Ambientais
                    # Pega o efeito da sala vizinha e encoda
                    n_effects = n_content.get('room_effects', [])
                    eff_name = n_effects[0] if n_effects else 'None'
                    
                    eff_vec = self.effect_encoder.encode(eff_name) # Retorna vetor de tamanho 9
                    
                    # Injeta nos slots 4 a 12 deste vizinho
                    obs[curr_idx + 4 : curr_idx + 13] = eff_vec
        
        # BLOCO FINAL (Gear and Items)
        # Deslocamento: 60 + (5 blocos * 13) = 125
        idx_end = 125
        
        # Self Gear
        w = agent['equipment'].get('Weapon')
        a = agent['equipment'].get('Armor')
        obs[idx_end + 0] = self.rarity_map.get(self.catalogs['equipment'][w].get('rarity'), 0.0) if w else 0.0
        obs[idx_end + 1] = self.rarity_map.get(self.catalogs['equipment'][a].get('rarity'), 0.0) if a else 0.0
        
        # Other Gear Score
        if other_agent_id:
            other_eq = self.agent_states[other_agent_id]['equipment']
            total_rarity = 0.0
            count = 0
            for slot in ['Weapon', 'Armor']:
                it = other_eq.get(slot)
                if it:
                    rst = self.catalogs['equipment'].get(it, {}).get('rarity')
                    total_rarity += self.rarity_map.get(rst, 0.0)
                    count += 1
            if count > 0:
                obs[idx_end + 2] = total_rarity / 2.0 
        
        # Flags Sociais
        if other_agent_id:
             if self.social_flags[other_agent_id].get('just_dropped'):
                 obs[idx_end + 3] = 1.0
             if self.social_flags[other_agent_id].get('skipped_attack'):
                 obs[idx_end + 4] = 1.0             
        
        # Porta
        if is_in_arena and current_graph_to_use.graph.get('meet_occurred'):
            obs[idx_end + 5] = 1.0 
        
        # BLOCO DE ITENS DETALHADOS (131 - 182)        
        # Weapon: 131:148 (17 slots)
        # Armor:  148:165 (17 slots)
        # Artifact: 165:182 (17 slots)
        
        w_data = self.catalogs['equipment'].get(w, {})
        a_data = self.catalogs['equipment'].get(a, {})
        art = agent['equipment'].get('Artifact')
        art_data = self.catalogs['equipment'].get(art, {})
                
        obs[131:148] = self.item_encoder.encode(w_data)
        obs[148:165] = self.item_encoder.encode(a_data)
        obs[165:182] = self.item_encoder.encode(art_data)   
                
        return obs         
    
    def reset(self, seed=None, options=None):
        # Chama o reset do pai (Gerencia o self.np_random interno do Gym)
        super().reset(seed=seed)
                
        # Se seed foi passado explicitamente, usamos ele. 
        # Se for None, pegamos do gerador interno do Gym para manter sincronia.
        if seed is not None:
            used_seed = seed
        else:
            # Pega uma semente derivada do estado atual do np_random do Gym            
            used_seed = self.np_random.integers(0, 2**32 - 1)
        
        # Aplica a semente no Python nativo (afeta o módulo 'random')
        random.seed(int(used_seed))
        
        # Aplica a semente no Numpy Global        
        np.random.seed(int(used_seed))
        
        # Limpa todos os dicionários de estado da sessão anterior
        self.agent_states = {}
        self.agent_names = {}
        self.graphs = {}
        self.current_nodes = {}
        self.current_floors = {}
        self.nodes_per_floor_counters = {}
        self.combat_states = {}        
        self.sanctum_dropped_history = {} # Armazena o último item descartado pelo agente na arena para evitar exploit de sinalização
        
        # Métricas
        self.current_episode_logs = {}
        self.enemies_defeated_this_episode = {}
        self.invalid_action_counts = {}
        self.last_milestone_floors = {}                
        
        # Buffers de estatísticas (Duração, Dano, Trocas)
        self.pvp_durations = []
        self.current_pvp_timer = 0
        self.damage_dealt_this_episode = {}
        self.equipment_swaps_this_episode = {}
        self.death_cause = {}
        self.pve_combat_durations = {}
        self.pvp_combat_durations = {}

        # Sociais / PvP
        self.arena_encounters_this_episode = {}
        self.pvp_combats_this_episode = {}
        self.bargains_succeeded_this_episode = {}
        self.bargains_trade_this_episode = {}
        self.bargains_toll_this_episode = {}
        self.cowardice_kills_this_episode = {}
        self.betrayals_this_episode = {}
        self.karma_history = {}
        self.matchmaking_queue = [] 
        self.active_matches = {}    
        self.arena_instances = {}               
        self.pvp_sessions = {} # LIMPA sessões PvP ativas
        self.pve_return_floor = {} # Salva o andar em que estava antes de entrar no Santuário


        # Estado Global
        self.current_step = 0
        # Começa no modo de progressão separado
        self.arena_instances = {}
        self.agents_in_arena = set()        

        # Limpa os agentes do sistema de reputação da run anterior
        self.reputation_system.agent_karma = {}
        
        # Inicialização de estados sociais
        self.social_flags = {agent_id: {} for agent_id in self.agent_ids}     
        self.arena_interaction_state = {
            agent_id: {'offered_peace': False} for agent_id in self.agent_ids
        }   

        # Dicionários de retorno para a API MAE
        observations = {}
        infos = {}

        # Gerador de nomes único para este episódio
        gerador_nomes = GeradorNomes()
        
        # Itera sobre todos os agentes e inicializa
        for agent_id in self.agent_ids:
            
            # Cria o agente
            agent_name = gerador_nomes.gerar_nome()
            self.agent_names[agent_id] = agent_name
            
            # Cria estado inicial
            self.agent_states[agent_id] = progression.create_initial_agent(agent_name)  

            # Reseta efeitos de status
            self.agent_states[agent_id]['effects'] = {}
            self.agent_states[agent_id]['persistent_effects'] = {}      

            # Inicializa métricas individuais
            self.current_episode_logs[agent_id] = []
            self.enemies_defeated_this_episode[agent_id] = 0
            self.invalid_action_counts[agent_id] = 0
            self.last_milestone_floors[agent_id] = 0
            self.combat_states[agent_id] = None
            self.damage_dealt_this_episode[agent_id] = 0.0
            self.equipment_swaps_this_episode[agent_id] = 0
            self.skill_upgrades_this_episode[agent_id] = 0
            self.death_cause[agent_id] = "Sobreviveu (Time Limit)"

            self.arena_encounters_this_episode[agent_id] = 0
            self.pvp_combats_this_episode[agent_id] = 0
            self.bargains_succeeded_this_episode[agent_id] = 0
            self.bargains_trade_this_episode[agent_id] = 0
            self.bargains_toll_this_episode[agent_id] = 0
            self.cowardice_kills_this_episode[agent_id] = 0
            self.betrayals_this_episode[agent_id] = 0
            self.karma_history[agent_id] = []
            self.sanctum_dropped_history[agent_id] = set()

            self.max_floor_reached_this_episode[agent_id] = 0
            
            self.pve_combat_durations[agent_id] = [] 
            self.pvp_combat_durations[agent_id] = []

            # Sistema de reputação
            initial_karma_state = complex(
                self.agent_states[agent_id]['karma']['real'],
                self.agent_states[agent_id]['karma']['imag']
            )
            self.reputation_system.add_agent(agent_id, initial_karma_state)

            # Zona de progressão individual
            self.current_floors[agent_id] = 0
            self.current_nodes[agent_id] = "start"
            self.nodes_per_floor_counters[agent_id] = {0: 1}
            self.graphs[agent_id] = nx.DiGraph()
            self.graphs[agent_id].add_node("start", floor=0)
            
            # Gera conteúdo da sala inicial (vazia)
            start_content = content_generation.generate_room_content(
                self.catalogs, 
                budget_multiplier=0.0,
                current_floor=0,
                guarantee_enemy=False
            )
            self.graphs[agent_id].nodes["start"]['content'] = start_content        
                        
            self._log(agent_id, f"[RESET] Novo episódio iniciado. {agent_name} (Nível {self.agent_states[agent_id]['level']}).\n")                      

            # Pré-gera o primeiro andar para este agente
            self._generate_and_populate_successors(agent_id, "start")
            
            # Coleta a observação inicial
            observations[agent_id] = self._get_observation(agent_id)
            infos[agent_id] = {}

        return observations, infos

    def _handle_combat_turn(self, agent_id: str, action: int, agent_info_dict: dict) -> tuple[float, bool]:
        """Orquestra um único turno de combate PvE e retorna (recompensa, combate_terminou)."""        
        
        # Pega o estado de combate PvE para ESTE agente
        combat_state = self.combat_states.get(agent_id) 

        # Se, por algum motivo, não houver estado de combate, encerra imediatamente.
        if not combat_state:            
            return -1, True # Penalidade pequena, combate terminou (bug)

        agent = combat_state['agent']
        enemy = combat_state['enemy']
        reward = 0
        combat_over = False

        # Vez do agente agir
        hp_before_enemy = enemy['hp']
        action_name = "Wait" # Padrão é esperar

        # Ações 0-3 são skills de combate. Ações 4-9 são inválidas em combate.
        if 0 <= action <= 3: 
            action_name = self.agent_states[agent_id]['skills'][action]
        else:
            # Ação é inválida DENTRO DO COMBATE (tentou mover, equipar, etc.)
            self.invalid_action_counts[agent_id] += 1
            reward = -5 # Penalidade por ação inválida em combate            
        
        # Executa a ação (seja a escolhida ou o "Wait" da penalidade)           
        combat.execute_action(agent, [enemy], action_name, self.catalogs)

        # Recompensa por dano causado
        damage_dealt = hp_before_enemy - enemy['hp']
        reward += damage_dealt * 0.6

        # Registra o dano causado na métrica do episódio
        if damage_dealt > 0:
            self.damage_dealt_this_episode[agent_id] += damage_dealt

        # Verifica se o inimigo morreu e aumenta o XP do agene
        if combat.check_for_death_and_revive(enemy, self.catalogs):            
            reward += 100 # Recompensa grande por vencer
            self.enemies_defeated_this_episode[agent_id] += 1
            
            # Pega o estado *principal* do agente (fora do combate)
            agent_main_state = self.agent_states[agent_id] 
            
            agent_main_state['exp'] += enemy.get('exp_yield', 50) # Ganha XP
            
            # Chamar a lógica de level up
            self._log(
                agent_id, # Passa o agent_id para o log
                f"[PVE] Inimigo '{enemy.get('name')}' derrotado. "
                f"EXP Ganhada: {enemy.get('exp_yield', 50)}. "
                f"EXP Total: {agent_main_state['exp']}. "
                f"EXP Nec: {agent_main_state['exp_to_level_up']}."
            )
            
            # Passa o estado principal para a regra de level up
            leveled_up = progression.check_for_level_up(agent_main_state) 

            if leveled_up:
                self._log(agent_id, f"[PVE] {self.agent_names[agent_id]} subiu para o nível {agent_main_state['level']} durante o combate.")
                reward += 50  # Recompensa extra por subir de nível
                agent_info_dict['level_up'] = True # Usa o dict de info do agente

                # SINCRONIZAÇÃO: Atualiza o agente 'agent' (cópia de combate) 
                # com os novos stats do 'agent_main_state' (principal)
                agent['hp'] = agent_main_state['hp']
                agent['max_hp'] = agent_main_state['max_hp']
                agent['base_stats'] = agent_main_state['base_stats'].copy()
                
                # Reseta os cooldowns no estado principal
                for skill in agent_main_state.get('cooldowns', {}):
                    agent_main_state['cooldowns'][skill] = 0
                
                # Sincroniza os cooldowns zerados de volta para a cópia de combate
                agent['cooldowns'] = agent_main_state['cooldowns'].copy()

            # Agora combate pode ser encerrado
            # Registra a duração do combate para esta luta
            duration = self.current_step - self.combat_states[agent_id]['start_step']
            self.pve_combat_durations[agent_id].append(duration)     

            # DEBUG           
            # print(f"[DEBUG-PVE] Duração do combate PvE: {duration} turnos.")

            # Filtra o que fica e o que sai
            surviving_effects = {}
            
            for tag, data in agent['effects'].items():
                duration = data.get('duration', 0)
                
                # REGRAS DE LIMPEZA:
                # 1. Se duration <= 0: Acabou o tempo, não salva.
                # 2. Se duration == -1: Era efeito da sala (Heat/Fog), não leva pra próxima.
                # 3. Se duration > 0: É um DoT/Buff ativo (Poison, Shield), SALVA.
                
                if duration > 0: 
                    surviving_effects[tag] = data

            # Salva os efeitos sobreviventes no estado mestre do agente
            self.agent_states[agent_id]['persistent_effects'] = surviving_effects

            self.combat_states[agent_id] = None # Limpa o estado de combate deste agente
            combat_over = True
        
        # Se o combate não chegou ao fim, o inimigo age
        if not combat_over:
            hp_before_agent = agent['hp']
            
            # IA Simples 
            available_skills = [s for s, cd in enemy['cooldowns'].items() if cd == 0]
            enemy_action = random.choice(available_skills) if available_skills else "Wait"
            
            if available_skills:
                enemy_action = random.choice(available_skills)
                # print(f"[DEBUG-PVE] 👾 {enemy['name']} escolheu skill '{enemy_action}' entre disponíveis (CDs: {available_skills}).")            

            combat.execute_action(enemy, [agent], enemy_action, self.catalogs)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
            # print(f"[DEBUG-PVE] 💥 {self.agent_names[agent_id]} sofreu {damage_taken} de dano do inimigo '{enemy['name']}'.")
            reward -= damage_taken * 0.5

            # Verifica se o agente morreu
            if combat.check_for_death_and_revive(agent, self.catalogs):
                combat_over = True # O loop principal do step() aplicará a penalidade final

                # Registra a causa da morte para logs e análises
                self.death_cause[agent_id] = f"PvE: {enemy['name']} (Lvl {enemy['level']})"

        # Fim do turno: resolve efeitos e cooldowns
        
        # Verifica se o estado de combate deste agente ainda existe
        # (pode ter sido setado para None acima se o inimigo morreu)
        if self.combat_states.get(agent_id): 
            combat.resolve_turn_effects_and_cooldowns(agent, self.catalogs)
            if not combat_over: # Garante que não tentemos resolver efeitos de um inimigo morto
                combat.resolve_turn_effects_and_cooldowns(enemy, self.catalogs)
        
        # Sincroniza o HP e Cooldowns do agente principal (estado mestre)
        # com a cópia de combate (agente)
        self.agent_states[agent_id]['hp'] = agent['hp'] 
        self.agent_states[agent_id]['cooldowns'] = agent['cooldowns'].copy()
        
        return reward, combat_over
    
    def _handle_pvp_combat_turn(self, combat_session: dict, action_a1: str, action_a2: str) -> tuple[float, float, bool, str, str]:
        """
        Executa turno PvP para uma sessão específica.
        Recebe: combat_session (o dicionário criado no initiate), ações.
        """
        
        # 1. Desempacota da sessão
        a1_combatant = combat_session['a1']
        a2_combatant = combat_session['a2']
        id_a1 = combat_session['a1_id'] # ID real do agente 1 (ex: 'agent_5')
        id_a2 = combat_session['a2_id'] # ID real do agente 2 (ex: 'agent_8')
        
        rew_a1 = 0
        rew_a2 = 0
        combat_over = False
        winner = None
        loser = None

        # Agente 'a1' age
        hp_before_a2 = a2_combatant['hp']
        combat.execute_action(a1_combatant, [a2_combatant], action_a1, self.catalogs)
        damage_dealt_by_a1 = hp_before_a2 - a2_combatant['hp']
        # rew_a1 += damage_dealt_by_a1 * 0.6
        # rew_a2 -= damage_dealt_by_a1 * 0.5 # Não estou priorizando DPS no PvP pra ver a memória deles do PvE em ação
        
        # Verifica se o agente 'a2' morreu
        if combat.check_for_death_and_revive(a2_combatant, self.catalogs):
            rew_a1 += 200 # Recompensa por vencer
            rew_a2 -= 300 # Penalidade por morrer
            combat_over = True
            winner = id_a1 # Retorna o ID real
            loser = id_a2

            # Estado mestre do vencedor e perdedor
            winner_state = self.agent_states[id_a1]
            loser_state = self.agent_states[id_a2]
            
            # Fórmula de XP: Base 100 + (Nível do Perdedor * 50)
            xp_gain = 100 + (loser_state['level'] * 50)
            
            winner_state['exp'] += xp_gain
            self._log(id_a1, f"[PVP] Ganhou {xp_gain} XP por derrotar {self.agent_names[id_a2]} (Lvl {loser_state['level']}).")
            
            # Checa Level Up
            if progression.check_for_level_up(winner_state):
                self._log(id_a1, f"[UPGRADE] Subiu para nível {winner_state['level']} após vitória PvP!")
                rew_a1 += 50 # Bônus extra no RL
                
                # Sincroniza status base novos para o combatente da arena
                a1_combatant['max_hp'] = winner_state['max_hp']
                a1_combatant['hp'] = winner_state['max_hp'] # Cura ao upar? Opcional
                a1_combatant['base_stats'] = winner_state['base_stats'].copy()
                           
            # Resolve Karma
            karma_adjustment = self._resolve_pvp_end_karma(winner, loser)
            rew_a1 += karma_adjustment # O vencedor recebe o bônus/multa
            
            # Dropar o Loot
            self._drop_pvp_loot(loser_id=loser, winner_id=winner)
            
            # Respawnar             
            # Não respawna aqui. Apenas no step() principal.

        # Agente 'a2' age (Se o combate NÃO terminou)
        if not combat_over:
            hp_before_a1 = a1_combatant['hp']
            combat.execute_action(a2_combatant, [a1_combatant], action_a2, self.catalogs)
            damage_dealt_by_a2 = hp_before_a1 - a1_combatant['hp']
            rew_a2 += damage_dealt_by_a2 * 0.6
            rew_a1 -= damage_dealt_by_a2 * 0.5
            
            # Verifica se o agente 'a1' morreu
            if combat.check_for_death_and_revive(a1_combatant, self.catalogs):
                rew_a2 += 200 # Recompensa por vencer
                rew_a1 -= 300 # Penalidade por morrer
                combat_over = True
                winner = id_a2 
                loser = id_a1  
                                
                # Resolve Karma
                self._resolve_pvp_end_karma(winner, loser)
                
                # Dropar o Loot
                self._drop_pvp_loot(loser_id=loser, winner_id=winner)
                
                # Respawnar             
                # Não respawna aqui. Apenas no step() principal.       
                
        # Fim do Turno (Se o combate NÃO terminou)
        if not combat_over:
            combat.resolve_turn_effects_and_cooldowns(a1_combatant, self.catalogs)
            combat.resolve_turn_effects_and_cooldowns(a2_combatant, self.catalogs)
        
        # Sincronização Final com o Estado Mestre
        # A sincronização do perdedor não é mais necessária, pois ele foi resetado.
        # Sincroniza apenas o vencedor (se houver) ou ambos (se o combate continuar).
        if loser != id_a1: # Se a1 não perdeu (ou seja, a1 venceu OU o combate continua)
             self.agent_states[id_a1]['hp'] = a1_combatant['hp']
             self.agent_states[id_a1]['cooldowns'] = a1_combatant['cooldowns'].copy()
        if loser != id_a2: # Se a2 não perdeu
             self.agent_states[id_a2]['hp'] = a2_combatant['hp']
             self.agent_states[id_a2]['cooldowns'] = a2_combatant['cooldowns'].copy()
        
        return rew_a1, rew_a2, combat_over, winner, loser
    
    def _drop_pvp_loot(self, loser_id: str, winner_id: str):
        """
        Pega todos os equipamentos do perdedor (estado mestre)
        e os joga no chão da sala da arena.
        """
        self._log(winner_id, f"[PVP] {self.agent_names[loser_id]} foi derrotado e dropou todos os seus itens.")
        
        # Pega o estado mestre do perdedor (ANTES do respawn)
        loser_main_state = self.agent_states[loser_id]        
        
        # Pega a sala atual (ambos estão na mesma sala)
        current_node_id = self.current_nodes[winner_id]        
                
        # Tenta recuperar a arena do vencedor
        current_arena = self.arena_instances.get(winner_id)
        
        if current_arena is None:
            # Fallback: Tenta recuperar a arena do perdedor (caso o vencedor tenha perdido a ref)
            current_arena = self.arena_instances.get(loser_id)            
            
        if current_arena is None:
            # Se realmente não encontrou a arena em nenhum dos dois, aborta com segurança
            self._log(winner_id, f"[WARN] Arena não encontrada para drop de loot (Winner: {winner_id}, Loser: {loser_id}). Itens perdidos.")
            return        
        
        # Garante que a estrutura de conteúdo exista no nó
        if current_node_id not in current_arena.nodes:
             # Caso raríssimo de dessincronia de nó
             self._log(winner_id, f"[WARN] Nó {current_node_id} não existe na arena. Loot perdido.")
             return

        if 'content' not in current_arena.nodes[current_node_id]:
             current_arena.nodes[current_node_id]['content'] = {}
        
        room_items = current_arena.nodes[current_node_id]['content'].setdefault('items', [])
        
        # Itera sobre os equipamentos do perdedor e joga no chão
        dropped_items = []
        if loser_main_state:
            for item_type, item_name in loser_main_state.get('equipment', {}).items():
                if item_name: # Se houver um item equipado nesse slot
                    room_items.append(item_name)
                    dropped_items.append(item_name)
        
        if dropped_items:
            self._log(winner_id, f"[PVP] Itens no chão: {', '.join(dropped_items)}")
    
    def _log(self, agent_id: str, message: str):
        """
        Log colorido no terminal e bufferização condicional.
        Otimizado para não comer RAM se não precisarmos do Hall da Fama.
        """

        COLORS = {
            "reset": "\033[0m",

            # Personagem / Agente — Ciano Neon
            "agent": "\033[38;5;51m",      

            # Ações — Verde Neon
            "action": "\033[38;5;46m",     

            # PvP — Vermelho Sangue Neon
            "pvp": "\033[38;5;196m",                   

            # Karma Negativo — Vermelho Intenso
            "karma_neg": "\033[38;5;196m", 

            # Karma Neutro — Amarelo Ouro
            "karma_neu": "\033[38;5;226m", 

            # Avisos — Amarelo Neon
            "warn": "\033[38;5;226m",      

            # Erros — Vermelho Neon
            "error": "\033[38;5;196m",     

            # Arena — Magenta Elétrico
            "arena": "\033[38;5;201m",     

            # Mapa — Azul Elétrico
            "map": "\033[38;5;27m",        

            # Upgrade — Âmbar Forte / Dourado
            "upgrade": "\033[38;5;214m",   
        }

        def colorize(text, color):
            return f"{COLORS[color]}{text}{COLORS['reset']}"

        # Print no Terminal (Apenas se verbose > 0)
        if self.verbose > 0:
            msg_upper = message.upper()
            if "[ERRO]" in msg_upper or "ERROR" in msg_upper: color = "error"
            elif  "EVENTO" in msg_upper: color = "upgrade"
            elif "[WARN]" in msg_upper: color = "warn"
            elif "[SOCIAL]" in msg_upper or "UPGRADE" in msg_upper:                
                if "(-)" in message or "NEGAT" in msg_upper: color = "karma_neg"
                else: color = "karma_neu"
            elif "PVP" in msg_upper or "MORTE" in msg_upper: color = "pvp"
            elif "AÇÃO" in msg_upper or "AÇÃO-ARENA" in msg_upper: color = "action"
            elif "SANCTUM" in msg_upper or "ZONA K" in msg_upper: color = "pvp"
            elif "PVE" in msg_upper: color = "map"
            else: color = "agent"

            formatted = f"{colorize(f'[{agent_id.upper()}]', 'agent')} {colorize(message, color)}"
            print(formatted)

        # Só salva no buffer se a flag estiver ativada
        if getattr(self, 'enable_logging_buffer', True): 
            if agent_id not in self.current_episode_logs:
                self.current_episode_logs[agent_id] = []
                        
            # Se passar de 1000 linhas, joga fora as antigas pra não estourar RAM
            if len(self.current_episode_logs[agent_id]) > 1000:
                self.current_episode_logs[agent_id].pop(0)

            self.current_episode_logs[agent_id].append(message + "\n")

    def _transition_to_arena(self, agent_id: str):
        """
        Gerencia a entrada na Zona K para N agentes.
        Usa uma Fila (Queue) para formar pares de matchmaking.
        """
        # Se já está na fila ou em partida, ignora
        if agent_id in self.matchmaking_queue or agent_id in self.active_matches:
            return

        self._log(agent_id, f"[SANCTUM] {self.agent_names[agent_id]} chegou ao Santuário (Andar {self.current_floors[agent_id]}). Entrando na fila...")
        
        # 1. Adiciona à fila de espera
        self.matchmaking_queue.append(agent_id)
        self.agents_in_arena.add(agent_id) # Marca como "não está mais no PvE"
        self.arena_entry_steps[agent_id] = self.current_step

        # 2. Verifica se formou um par
        if len(self.matchmaking_queue) >= 2:
            # Remove os dois primeiros da fila (FIFO)
            p1 = self.matchmaking_queue.pop(0)
            p2 = self.matchmaking_queue.pop(0)
            
            self._log(p1, f"[MATCHMAKING] Par encontrado: {self.agent_names[p2]}!")
            self._log(p2, f"[MATCHMAKING] Par encontrado: {self.agent_names[p1]}!")

            # 3. Registra o Pareamento (Bidirecional)
            self.active_matches[p1] = p2
            self.active_matches[p2] = p1

            # Salva o andar de origem de cada um
            self.pve_return_floor[p1] = self.current_floors[p1]
            self.pve_return_floor[p2] = self.current_floors[p2]

            # Limpa os itens descartados pelo agente no Santuário
            self.sanctum_dropped_history[agent_id] = set()
            
            # 4. Gera a Arena (Instância Única para o par)
            # Usa o andar do p1 como base
            base_floor = self.current_floors[p1]
            
            new_arena = map_generation.generate_k_zone_topology(
                floor_level=base_floor,
                num_nodes=9,
                connectivity_prob=0.4 
            )
            
            # Popula a Arena (Sem inimigos)
            for node in new_arena.nodes():
                content = content_generation.generate_room_content(
                    catalogs=self.catalogs,
                    current_floor=base_floor, 
                    budget_multiplier=self.budget_multiplier, 
                    guarantee_enemy=False 
                )
                content['enemies'] = [] 
                content['items'] = []
                content['events'] = []
                new_arena.nodes[node]['content'] = content

            # 5. Atribui a Arena aos agentes
            self.arena_instances[p1] = new_arena
            self.arena_instances[p2] = new_arena
            
            # 6. Posiciona os agentes
            nodes_list = sorted(list(new_arena.nodes()))
            self.current_nodes[p1] = nodes_list[0]
            self.current_nodes[p2] = nodes_list[-1]
            
            # Atualiza estatísticas
            self.arena_encounters_this_episode[p1] += 1
            self.arena_encounters_this_episode[p2] += 1
            
            # (O estado global self.env_state perde sentido com N agentes, 
            # pois alguns estão em PvE e outros em PvP. 
            # Agora 'agent_id in self.active_matches' dita o estado).

    def _process_pve_step(self, agent_id: str, action: int, global_truncated: bool, infos: dict) -> tuple[float, bool]:
        """
        Processa um passo completo de PvE (PROGRESSION) para um único agente.
        Refatorado com lógica de Respawn: Morte não termina o episódio.
        Retorna (recompensa_do_agente, agente_terminou)
        """
        
        agent_reward = -0.5 # Penalidade de tempo
        agent_terminated = False # 'terminated' agora significa VENCER o jogo.
        game_won = False
        
        # --- Lógica de Jogo ---
        # 1. Verifica Vitória (Chegou ao fim do DAG)
        current_node = self.current_nodes[agent_id]
        current_graph = self.graphs[agent_id]
        is_on_last_floor = (self.current_floors[agent_id] == self.max_floors)
        has_no_successors = not list(current_graph.successors(current_node))

        if is_on_last_floor and has_no_successors and not self.combat_states.get(agent_id):
            agent_terminated = True # Venceu!
            game_won = True 
            agent_reward += 1000
            self._log(agent_id, f"[PVE] FIM: {self.agent_names[agent_id]} VENCEU! (Chegou ao fim do Jogo)")

        # Processa Ação Combate ou Exploração
        if self.combat_states.get(agent_id):
            # Agente está em combate PvE
            reward_combat, combat_over = self._handle_combat_turn(agent_id, action, infos[agent_id])
            agent_reward += reward_combat
            
            if combat_over and self.agent_states[agent_id]['hp'] > 0:
                # Se o combate terminou com vitória, remove o inimigo do grafo
                room_content = current_graph.nodes[current_node].get('content', {})
                if room_content.get('enemies'):
                    room_content['enemies'].pop(0)
        else:
            # Agente está explorando (Ações 4-9)
            reward_explore, _ = self._handle_exploration_turn(agent_id, action)
            agent_reward += reward_explore
        
        # 3. Verifica Morte do Agente
        if self.agent_states[agent_id]['hp'] <= 0:                        
            agent_reward = -300 

            # Captura o vetor de observação do estado de morte (antes de virar Nível 1)
            # A rede neural precisa ver "zeros" na vida e a sala onde morreu.
            terminal_obs = self._get_observation(agent_id)
            infos[agent_id]['terminal_observation'] = terminal_obs
            
            cause = "Dano Ambiental"
            if self.combat_states.get(agent_id):
                enemy_name = self.combat_states[agent_id]['enemy']['name']
                cause = f"{enemy_name} (Lvl {self.combat_states[agent_id]['enemy']['level']}) PVE"
            self.death_cause[agent_id] = cause

            # Faz a limpa nos logs ANTES de salvar o final_status            
            infos[agent_id]['final_status'] = {
                'level': self.agent_states[agent_id]['level'], 
                'hp': 0, # Morreu
                'floor': self.max_floor_reached_this_episode.get(agent_id, self.current_floors[agent_id]),
                'win': False,
                'steps': self.current_step,
                'enemies_defeated': self.enemies_defeated_this_episode[agent_id],
                'invalid_actions': self.invalid_action_counts[agent_id],
                'agent_name': self.agent_names[agent_id],
                'full_log': self.current_episode_logs[agent_id].copy(), # Salva o log antes de limpar
                'equipment': self.agent_states[agent_id].get('equipment', {}).copy(), # Salva itens antes de perder
                'exp': self.agent_states[agent_id]['exp'],
                'damage_dealt': self.damage_dealt_this_episode[agent_id],
                'equipment_swaps': self.equipment_swaps_this_episode[agent_id],
                'death_cause': cause,
                
                # Métricas Sociais
                'pve_durations': self.pve_combat_durations[agent_id].copy(),
                'pvp_durations': self.pvp_combat_durations[agent_id].copy(),
                'arena_encounters': self.arena_encounters_this_episode[agent_id],
                'pvp_combats': self.pvp_combats_this_episode[agent_id],
                'bargains_succeeded': self.bargains_succeeded_this_episode[agent_id],
                'bargains_trade': self.bargains_trade_this_episode[agent_id],
                'bargains_toll': self.bargains_toll_this_episode[agent_id],
                'cowardice_kills': self.cowardice_kills_this_episode[agent_id],
                'betrayals': self.betrayals_this_episode[agent_id],
                'karma_history': self.karma_history[agent_id],
                'karma': self.agent_states[agent_id]['karma']
            }                        

            # AGORA reseta (destroi os dados antigos para a nova vida)
            self._respawn_agent(agent_id, cause=cause)

        # 4. Recompensa de Marco (Milestone)
        current_floor = self.current_floors[agent_id]
        last_recorded = self.last_milestone_floors.get(agent_id, 0)
        
        milestone_bonus = 0
        new_milestone_reached = False
        sanctum_k = self.sanctum_floor # Usamos uma variável local para K 
        
        milestones_rules = [
            # 1. Marco de Sobrevivência à Zona K (Dispara em K+1, ex: 26, 51)            
            # {"check": lambda f: (f - 1) % sanctum_k == 0 and f > 1, "bonus": 100, "msg": "Sobreviveu à Zona K!"},

            # 2. Marco Decimal (Dispara a cada 10 andares: 10, 20, 30...)
            {"check": lambda f: f % 10 == 0 and f > 0, "bonus": 100, "msg": "Marco Decimal!"}
        ]

        for milestone in milestones_rules:
            # Condição: A regra do andar atual é satisfeita E este andar é maior que o último registrado.
            if milestone["check"](current_floor) and current_floor > last_recorded:
                milestone_bonus += milestone["bonus"]
                self._log(agent_id, f"[MARCO] {self.agent_names[agent_id]} alcançou o Andar {current_floor}! {milestone['msg']} Bônus +{milestone['bonus']}.")
                new_milestone_reached = True

        # Aplica o resultado
        if new_milestone_reached:
            agent_reward += milestone_bonus
            # Atualiza o tracker apenas uma vez para não dar erro de condição de corrida
            self.last_milestone_floors[agent_id] = current_floor                       

        # 5. Verifica Transição para Arena
        # (O 'current_floor' será 0 após o respawn, então a checagem '... > 0' previne transição imediata)
        if not agent_terminated and self.current_floors[agent_id] > 0 and \
           self.current_floors[agent_id] % self.sanctum_floor == 0: # Ex: Andares K, 2k, 3K...
            
            if agent_id not in self.agents_in_arena:
                 self._transition_to_arena(agent_id) 

        # 7. Cria 'final_status' se o episódio terminou para este agente
        if agent_terminated or global_truncated:
            # Só gera se não tiver gerado antes (na morte)
            if 'final_status' not in infos[agent_id]:
                infos[agent_id]['final_status'] = {
                    'level': self.agent_states[agent_id]['level'],
                    'hp': self.agent_states[agent_id]['hp'],                    
                    'floor': self.max_floor_reached_this_episode.get(agent_id, 0),
                    'win': game_won and self.agent_states[agent_id]['hp'] > 0,
                    'steps': self.current_step,
                    'enemies_defeated': self.enemies_defeated_this_episode.get(agent_id, 0),
                    'invalid_actions': self.invalid_action_counts.get(agent_id, 0),
                    'agent_name': self.agent_names.get(agent_id, "Roberto"),
                    
                    # Logs e Equipamentos (precisa de copy)
                    'full_log': self.current_episode_logs.get(agent_id, []).copy(), 
                    'equipment': self.agent_states[agent_id].get('equipment', {}).copy(),
                    
                    'exp': self.agent_states[agent_id]['exp'],
                    'damage_dealt': self.damage_dealt_this_episode.get(agent_id, 0.0),
                    'equipment_swaps': self.equipment_swaps_this_episode.get(agent_id, 0),
                    'skill_upgrades': self.skill_upgrades_this_episode.get(agent_id, 0),
                    'skills': self.agent_states[agent_id]['skills'].copy(), 
                    
                    # Segurança caso death_cause não tenha sido setado ainda
                    'death_cause': self.death_cause.get(agent_id, "Desconhecido"),
                    
                    # --- Listas de Duração ---                    
                    'pve_durations': self.pve_combat_durations.get(agent_id, []).copy(),
                    'pvp_durations': self.pvp_combat_durations.get(agent_id, []).copy(),
                    
                    'arena_encounters': self.arena_encounters_this_episode.get(agent_id, 0),
                    'pvp_combats': self.pvp_combats_this_episode.get(agent_id, 0),
                    'bargains_succeeded': self.bargains_succeeded_this_episode.get(agent_id, 0),
                    'bargains_trade': self.bargains_trade_this_episode.get(agent_id, 0),
                    'bargains_toll': self.bargains_toll_this_episode.get(agent_id, 0),
                    'cowardice_kills': self.cowardice_kills_this_episode.get(agent_id, 0),
                    
                    # Histórico de Karma é lista, precisa de copy
                    'karma_history': self.karma_history.get(agent_id, []).copy(), 
                    'betrayals': self.betrayals_this_episode.get(agent_id, 0),
                    
                    # Karma atual é um dict/objeto, precisa de copy
                    'karma': self.agent_states[agent_id]['karma'].copy() 
                }                                
                    
        return agent_reward, agent_terminated

    def step(self, actions: dict):
        """
        Executa um passo no ambiente multiagente (N Agentes).
        Dispatcher Universal: Decide a lógica baseada no estado individual de cada agente.
        """
        
        # Inicialização
        self.current_step += 1
        
        rewards = {agent_id: 0 for agent_id in self.agent_ids}
        terminateds = {agent_id: False for agent_id in self.agent_ids}
        infos = {agent_id: {} for agent_id in self.agent_ids}
        
        global_truncated = self.current_step >= self.max_episode_steps
        truncateds = {agent_id: global_truncated for agent_id in self.agent_ids}
        terminateds['__all__'] = False
        truncateds['__all__'] = global_truncated

        # Usamos sets para evitar processar o mesmo agente múltiplas vezes
        processed_agents = set()

        # Limpeza de flags sociais (Turno Anterior)
        for agent_id in self.agent_ids:
            self.social_flags[agent_id]['just_picked_up'] = False
            self.social_flags[agent_id]['just_dropped'] = False 
            self.social_flags[agent_id]['skipped_attack'] = False

        # Loop principal de agentes
        for agent_id in self.agent_ids:
            
            # Se já foi processado neste loop ou já terminou
            if agent_id in processed_agents or terminateds.get(agent_id, False):
                continue

            action = actions[agent_id]

            # CASO 1: ESTÁ EM COMBATE PVP?
            if agent_id in self.pvp_sessions:
                session = self.pvp_sessions[agent_id]
                p1 = session['a1_id']
                p2 = session['a2_id']
                
                processed_agents.add(p1)
                processed_agents.add(p2)
                
                # Pega as ações
                state_p1 = self.agent_states[p1]
                state_p2 = self.agent_states[p2]

                # Pega a skill do deck do agente (ou 'Wait' se a ação for inválida para combate)                
                idx_p1 = actions[p1].item() if hasattr(actions[p1], 'item') else actions[p1]
                idx_p2 = actions[p2].item() if hasattr(actions[p2], 'item') else actions[p2]

                if 0 <= idx_p1 <= 3:
                    act_p1 = state_p1['skills'][idx_p1]
                else:
                    act_p1 = "Wait"

                if 0 <= idx_p2 <= 3:
                    act_p2 = state_p2['skills'][idx_p2]
                else:
                    act_p2 = "Wait"

                # Resolve o turno PvP
                rew1, rew2, over, winner, loser = self._handle_pvp_combat_turn(session, act_p1, act_p2)
                
                rewards[p1] += rew1
                rewards[p2] += rew2
                
                if over:
                    # Logs e Métricas
                    duration = self.current_step - session['start_step']
                    self.pvp_combat_durations[p1].append(duration)
                    self.pvp_combat_durations[p2].append(duration)

                    # DEBUG
                    # print(f"[DEBUG-PVP] Duração do combate PVP: {duration} turnos")
                    
                    self._log(winner, f"[PVP] VITÓRIA de {self.agent_names[winner]}!")
                    
                    # CICLO DE VIDA: O PERDEDOR
                    # Define Morte
                    terminateds[loser] = True 
                    
                    # Remove da Arena (Transição de Estado: Arena -> PvE)
                    if loser in self.arena_instances: del self.arena_instances[loser]
                    if loser in self.agents_in_arena: self.agents_in_arena.discard(loser)
                    
                    # Reseta o agente derrotado
                    self._respawn_agent(loser, "PVP")

                    # CICLO DE VIDA: O VENCEDOR
                    # Quebra o Pareamento (Ele agora está "Solteiro" na arena)
                    if winner in self.active_matches: del self.active_matches[winner]
                    if loser in self.active_matches: del self.active_matches[loser]

                    # Destranca a Porta 
                    if winner in self.arena_instances:
                        self.arena_instances[winner].graph['meet_occurred'] = True
                        self._log(winner, "[SANCTUM] Oponente eliminado. Saída destrancada.")

                    # Limpa a sessão de PVP
                    if p1 in self.pvp_sessions: del self.pvp_sessions[p1]
                    if p2 in self.pvp_sessions: del self.pvp_sessions[p2]

            # CASO 2: ESTÁ NA FILA DE ESPERA? (Sincronização)
            elif agent_id in self.matchmaking_queue:
                rewards[agent_id] = 0 # Esperando...
                # Não faz nada

            # CASO 3: ESTÁ DENTRO DA ARENA (Interação Social)?
            elif agent_id in self.arena_instances:
                processed_agents.add(agent_id)

                # Verifica há quanto tempo o agente está aqui
                if agent_id in self.arena_entry_steps:
                    steps_in_arena = self.current_step - self.arena_entry_steps[agent_id]
                    
                    # Limite de Tolerância: 20 steps (aprox. 20 ações)
                    if steps_in_arena > 30: 
                        overtime = steps_in_arena - 30
                        
                        # Punição Exponencial no Reward (Desconforto Mental)
                        # Começa em -0.1 e vai piorando rápido
                        pressure_penalty = -0.1 * (1 + (overtime * 0.1))
                        rewards[agent_id] += max(pressure_penalty, -10.0) # Teto de -10.0
                        
                        # Dano Físico (Aperto Real)
                        # A cada 5 turnos extras, perde 5% da vida MÁXIMA
                        if overtime % 5 == 0:
                            damage = self.agent_states[agent_id]['max_hp'] * 0.05
                            self.agent_states[agent_id]['hp'] -= damage
                            self._log(agent_id, f"[SANCTUM] Uma atmosfera opressiva drena sua vida (-{int(damage)} HP)...")
                            
                            # Verifica Morte por Pressão
                            if self.agent_states[agent_id]['hp'] <= 0:
                                self.death_cause[agent_id] = "Profanação do Santuário"
                                self._log(agent_id, "[MORTE] O agente sucumbiu à pressão do Santuário.")
                                
                                # Limpa ao morrer
                                terminateds[agent_id] = True # Avisa o SB3 que acabou
                                
                                # Remove da Arena
                                if agent_id in self.arena_instances: del self.arena_instances[agent_id]
                                if agent_id in self.agents_in_arena: self.agents_in_arena.discard(agent_id)
                                
                                # Liberta o oponente (se houver)
                                opp_id = self.active_matches.get(agent_id)
                                if opp_id:
                                    if opp_id in self.active_matches: del self.active_matches[opp_id]
                                    # O oponente venceu por sobrevivência -> Destranca a porta dele
                                    if opp_id in self.arena_instances:
                                        self.arena_instances[opp_id].graph['meet_occurred'] = True
                                        self._log(opp_id, "[SANCTUM] O oponente sucumbiu. Saída destrancada.")
                                
                                if agent_id in self.active_matches: del self.active_matches[agent_id]
                                
                                # Reseta Posição para evitar crash
                                self._respawn_agent(agent_id, self.death_cause[agent_id])
                                
                                # PULA O RESTO DO TURNO (O agente morreu, não pode agir)
                                continue
                
                # Descobre o oponente
                opponent_id = self.active_matches.get(agent_id)                
                
                # Verifica Encontro/Porta (Rito de Passagem)
                if opponent_id:
                    arena_graph = self.arena_instances[agent_id]
                    if not arena_graph.graph.get('meet_occurred', False):
                        if self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                            arena_graph.graph['meet_occurred'] = True
                            self._log(agent_id, f"[SANCTUM] Encontro entre {self.agent_names[agent_id]} e {self.agent_names[opponent_id]}! Saída liberada.")
                            self._log(opponent_id, f"[SANCTUM] Encontro entre {self.agent_names[opponent_id]} e {self.agent_names[agent_id]}! Saída liberada.")
                            rewards[agent_id] += 100
                            rewards[opponent_id] += 100                        

                # Processa Ação (Traição, Movimento, etc)                
                # Detecta Intenção de Ataque (0-3)
                target_id = None
                if 0 <= action <= 3 and opponent_id:
                    if self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                        target_id = opponent_id
                
                if target_id:
                    # Lógica de Traição (Atacou oferta de paz do outro)
                    if self.arena_interaction_state[target_id]['offered_peace']:
                         self._log(agent_id, f"[SOCIAL] TRAIÇÃO! {self.agent_names[agent_id]} TRAIU a oferta de paz de {self.agent_names[target_id]}! Karma (--).")
                         self.reputation_system.update_karma(agent_id, 'bad')
                         self.reputation_system.update_karma(agent_id, 'bad') # Dupla                         
                         rewards[agent_id] -= 100
                         self.betrayals_this_episode[agent_id] += 1
                         self.arena_interaction_state[target_id]['offered_peace'] = False

                    # Lógica de Perfídia (Atacou segurando a própria oferta)
                    if self.arena_interaction_state[agent_id]['offered_peace']:
                         self._log(agent_id, f"[SOCIAL] PERFÍDIA! {self.agent_names[agent_id]} atacou enquanto oferecia paz! Karma (---).")
                         self.reputation_system.update_karma(agent_id, 'bad')
                         self.reputation_system.update_karma(agent_id, 'bad')
                         self.reputation_system.update_karma(agent_id, 'bad') # Tripla
                         rewards[agent_id] -= 100
                         self.betrayals_this_episode[agent_id] += 1
                         self.arena_interaction_state[agent_id]['offered_peace'] = False

                    # Inicia PvP                    
                    self._initiate_pvp_combat(agent_id, target_id) 
                    # rewards[agent_id] += 20 # Teste de ablação
                    
                    # Marca o oponente como processado (para não agir neste turno)
                    processed_agents.add(target_id)
                    
                    # Limpa ofertas de paz (Guerra começou)
                    self.arena_interaction_state[agent_id]['offered_peace'] = False
                    self.arena_interaction_state[target_id]['offered_peace'] = False

                # Ações Pacíficas (4-9)
                elif 4 <= action <= 9:                    
                    r, _ = self._handle_exploration_turn(agent_id, action)
                    rewards[agent_id] += r
                    
                    # Atualiza estado de Paz (Se dropou com sucesso)
                    if self.social_flags[agent_id].get('just_dropped') and agent_id in self.active_matches:
                        self.arena_interaction_state[agent_id]['offered_peace'] = True
                        self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} ofereceu seu artefato como oferta de paz.")
                    
                    # Atualiza Skipped Attack (Flag Social)
                    if opponent_id and self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                        self.social_flags[agent_id]['skipped_attack'] = True

                    # Verifica Barganha (Pós-Ação)
                    if opponent_id and self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                        # Minha paz ativa + Oponente pegou item AGORA
                        my_peace_accepted = self.arena_interaction_state[agent_id]['offered_peace'] and \
                                            self.social_flags[opponent_id].get('just_picked_up')
                        
                        # Paz do oponente ativa + Eu peguei item AGORA
                        opp_peace_accepted = self.arena_interaction_state[opponent_id]['offered_peace'] and \
                                             self.social_flags[agent_id].get('just_picked_up')
                        
                        if my_peace_accepted or opp_peace_accepted:
                            self._log(agent_id, f"[SOCIAL] Os agentes {self.agent_names[agent_id]} e {self.agent_names[opponent_id]} concluíram uma Barganha Pacífica! Karma (++)")
                            self._log(opponent_id, f"[SOCIAL] Os agentes {self.agent_names[opponent_id]} e {self.agent_names[agent_id]} concluíram uma Barganha Pacífica! Karma (++)")
                            
                            self.reputation_system.update_karma(agent_id, 'good')
                            self.reputation_system.update_karma(agent_id, 'good')
                            self.reputation_system.update_karma(opponent_id, 'good')
                            self.reputation_system.update_karma(opponent_id, 'good')
                            rewards[agent_id] += 200
                            rewards[opponent_id] += 200 
                            
                            # Incrementa estatísticas de barganha bem-sucedida (Geral)
                            self.bargains_succeeded_this_episode[agent_id] += 1
                            self.bargains_succeeded_this_episode[opponent_id] += 1

                            # Incrementa a estatística de trocas
                            self.bargains_trade_this_episode[agent_id] += 1
                            self.bargains_trade_this_episode[opponent_id] += 1
                            
                            self.arena_interaction_state[agent_id]['offered_peace'] = False
                            self.arena_interaction_state[opponent_id]['offered_peace'] = False
                            
                            # Encerra a arena para ambos
                            self._end_arena_encounter(agent_id)
                            self._end_arena_encounter(opponent_id)
                            processed_agents.add(opponent_id)                        

                else: # Ação Inválida (> 9)
                     self.invalid_action_counts[agent_id] += 1
                     rewards[agent_id] -= 5

            # CASO 4: ESTÁ NO PVE?
            else:
                # Processa o passo PvE normal
                agent_reward, agent_terminated = self._process_pve_step(
                    agent_id, action, global_truncated, infos
                )
                rewards[agent_id] = agent_reward
                terminateds[agent_id] = agent_terminated

        # Finalização e coleta
        
        if all(terminateds.get(aid, False) for aid in self.agent_ids) or global_truncated:
            terminateds['__all__'] = True
            truncateds['__all__'] = True
            
            # Salva final_status se truncou
            if global_truncated:
                 for agent_id in self.agent_ids:
                    if 'final_status' not in infos[agent_id]:
                        infos[agent_id]['final_status'] = {
                            'level': self.agent_states[agent_id]['level'],
                            'hp': self.agent_states[agent_id]['hp'],
                                                        
                            'floor': self.max_floor_reached_this_episode.get(agent_id, 0),
                            'win': False, # Se caiu aqui pelo truncation, não venceu
                            'steps': self.current_step,
                            'enemies_defeated': self.enemies_defeated_this_episode.get(agent_id, 0),
                            'invalid_actions': self.invalid_action_counts.get(agent_id, 0),
                            'agent_name': self.agent_names.get(agent_id, "Unknown"),
                                                        
                            'full_log': self.current_episode_logs.get(agent_id, []).copy(),
                            'equipment': self.agent_states[agent_id].get('equipment', {}).copy(),
                            
                            'exp': self.agent_states[agent_id]['exp'],
                            'damage_dealt': self.damage_dealt_this_episode.get(agent_id, 0.0),
                            'equipment_swaps': self.equipment_swaps_this_episode.get(agent_id, 0),
                            'skill_upgrades': self.skill_upgrades_this_episode.get(agent_id, 0), 
                            'skills': self.agent_states[agent_id]['skills'].copy(), 
                            
                            # Se o jogo acabou por tempo, a causa da morte pode não existir no dict
                            'death_cause': self.death_cause.get(agent_id, "Time Limit Reached"),
                            
                            # Listas de duração
                            'pve_durations': self.pve_combat_durations.get(agent_id, []).copy(),
                            'pvp_durations': self.pvp_combat_durations.get(agent_id, []).copy(),
                            
                            'arena_encounters': self.arena_encounters_this_episode.get(agent_id, 0),
                            'pvp_combats': self.pvp_combats_this_episode.get(agent_id, 0),
                            'bargains_succeeded': self.bargains_succeeded_this_episode.get(agent_id, 0),
                            
                            # Méritos de Barganha
                            'bargains_trade': self.bargains_trade_this_episode.get(agent_id, 0),
                            'bargains_toll': self.bargains_toll_this_episode.get(agent_id, 0),
                            
                            'cowardice_kills': self.cowardice_kills_this_episode.get(agent_id, 0),
                            
                            # Histórico e Karma
                            'karma_history': self.karma_history.get(agent_id, []).copy(),
                            'betrayals': self.betrayals_this_episode.get(agent_id, 0),
                            'karma': self.agent_states[agent_id]['karma'].copy()
                        }                             

        observations = {
            agent_id: self._get_observation(agent_id) for agent_id in self.agent_ids
        }
        
        # Coleta de Karma periódico
        if self.current_step % 10 == 0:
             for agent_id in self.agent_ids:
                 z = self.reputation_system.get_karma_state(agent_id)
                 self.karma_history[agent_id].append({'real': z.real, 'imag': z.imag})

        return observations, rewards, terminateds, truncateds, infos
    
    def _handle_exploration_turn(self, agent_id: str, action: int):
        """
        Processa uma ação em modo de exploração (fora de combate).
        Versão FINAL: Inclui segurança, lógica social, equipamentos inteligentes e saída da arena.
        """
        reward = 0
        terminated = False
        
        # Definições
        current_node_id = self.current_nodes[agent_id]
        
        # Verifica se ESTE AGENTE está na Arena        
        is_in_arena = (agent_id in self.arena_instances)
        reward = 0.0
        
        # Seleciona o grafo correto
        current_graph = self.arena_instances[agent_id] if is_in_arena else self.graphs.get(agent_id)

        # Lógica das ações de movimento (6, 7, 8, 9)
        if 6 <= action <= 9:
            neighbor_index = action - 6
                        
            if is_in_arena:
                neighbors = list(current_graph.neighbors(current_node_id))
            else:
                neighbors = list(current_graph.successors(current_node_id))
            
            neighbors.sort() 

            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} na sala '{current_node_id}'. Tentando Mover Vizinho {neighbor_index}.")

            if neighbor_index < len(neighbors):                
                chosen_node = neighbors[neighbor_index]      

                # O agente andou, o tempo passou.
                self._tick_world_cooldowns(agent_id)                                 

                # Se o agente está voltando para o nó de onde acabou de vir (A -> B -> A)
                prev_node = self.previous_nodes.get(agent_id)
                
                # Penalidade base por mover na Arena (Taxa de Oxigênio)
                step_cost = -0.5 if is_in_arena else 0.0
                
                if prev_node and chosen_node == prev_node:
                     # Punição severa por voltar (Ping-Pong)
                     reward -= 1.5
                else:
                     # Recompensa por explorar nó novo
                     if not is_in_arena:
                         reward += 0.5 # PvE: Incentiva explorar
                     else:
                         reward += 0.5 # Arena: Incentiva explorar (necessário pra ativar encontros e saída)
                
                # Aplica a taxa base
                reward += step_cost 

                # Lógica de saída da arena com tributo/pedágio
                if is_in_arena:
                    node_data = current_graph.nodes[chosen_node]

                    if node_data.get('is_exit', False):
                        # Verifica a trava global de encontro
                        meet_occurred = current_graph.graph.get('meet_occurred', False)
                        
                        if not meet_occurred:
                            self._log(agent_id, f"[AÇÃO-ARENA] A saída em '{chosen_node}' está TRANCADA. Encontre o outro agente primeiro!")
                            return -5.0, terminated 
                        
                        # Lógica de Saída com Paz (Pedágio)
                        base_exit_reward = 20.0
                        bonus_peace_reward = 0.0
                        
                        # Verifica se há oferta de paz ativa (minha ou do oponente)
                        my_peace = self.arena_interaction_state[agent_id]['offered_peace']
                        
                        # Identifica o oponente 
                        opponent_id = self.active_matches.get(agent_id)
                        other_peace = False
                        if opponent_id:
                             other_peace = self.arena_interaction_state[opponent_id]['offered_peace']
                        
                        # Se houve oferta de paz e estamos saindo vivos
                        if my_peace or other_peace:
                            bonus_peace_reward = 80.0 # Total 100 (como na barganha por troca)
                            self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} saiu do Santuário em paz, pagando o tributo.")

                            # Incrementa estatísticas de barganha bem-sucedida (Geral)
                            self.bargains_succeeded_this_episode[agent_id] += 1

                            # Incrementa a estatística de trocas
                            self.bargains_toll_this_episode[agent_id] += 1
                            
                            # Atualiza Karma e Estatísticas
                            self.reputation_system.update_karma(agent_id, 'good')
                            if opponent_id:
                                self.reputation_system.update_karma(opponent_id, 'good')
                                self.bargains_succeeded_this_episode[opponent_id] += 1
                                self.bargains_toll_this_episode[opponent_id] += 1 
                                                        
                        else:
                             self._log(agent_id, f"[SANCTUM] Saiu ignorando oferta de paz.")

                        # O agente sai da arena
                        self._end_arena_encounter(agent_id)
                        
                        # Liberta o Oponente também 
                        if opponent_id and opponent_id in self.arena_instances:
                            # Opção A: O oponente também sai automaticamente (Empate técnico)
                            self._end_arena_encounter(opponent_id)
                            self._log(opponent_id, f"[SANCTUM] O oponente partiu. O santuário se desfaz.")                                                        

                        return base_exit_reward + bonus_peace_reward, terminated                            

                self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} fez um movimento VÁLIDO para '{chosen_node}'.")
                
                # Lógica de Poda (Apenas na P-Zone)
                if not is_in_arena:
                    all_successors = list(current_graph.successors(current_node_id))
                    nodes_to_remove = [succ for succ in all_successors if succ != chosen_node]
                    for node in nodes_to_remove:
                        if current_graph.has_node(node):
                            descendants = nx.descendants(current_graph, node)
                            current_graph.remove_nodes_from(list(descendants) + [node])
                
                # Efetua o Movimento
                self.current_nodes[agent_id] = chosen_node
                self.current_floors[agent_id] = current_graph.nodes[chosen_node].get('floor', self.current_floors[agent_id])

                new_floor = current_graph.nodes[chosen_node].get('floor', 0)
                if new_floor > self.max_floor_reached_this_episode[agent_id]:
                    self.max_floor_reached_this_episode[agent_id] = new_floor
                
                reward = -0.1 if is_in_arena else 5

                # Gera novos sucessores e Combate PvE (Apenas P-Zone)
                if not is_in_arena:
                    if not list(current_graph.successors(chosen_node)):
                        self._generate_and_populate_successors(agent_id, chosen_node)

                    room_content = current_graph.nodes[chosen_node].get('content', {})

                    # Verifica Eventos Especiais
                    event_name = room_content.get('events')
                
                    if event_name == 'Fountain of Life':
                        agent_state = self.agents[agent_id]
                        
                        # Configuração da Cura (50% do HP Máximo)
                        heal_percent = 0.50 
                        heal_amount = int(agent_state['max_hp'] * heal_percent)
                        
                        # Aplica a cura (sem ultrapassar o Max HP)
                        old_hp = agent_state['hp']
                        agent_state['hp'] = min(agent_state['max_hp'], agent_state['hp'] + heal_amount)
                        actual_healed = agent_state['hp'] - old_hp
                        
                        # Log para você rastrear (e evitar achar que é cura fantasma!)
                        self._log(agent_id, f"[EVENTO] ⛲ Encontrou '{event_name}'! Recuperou {actual_healed} HP ({old_hp} -> {agent_state['hp']}).")

                        # Isso transforma a fonte em uma sala comum vazia após o primeiro gole
                        room_content['events'].remove('Fountain of Life')
                                                                            
                    enemy_names = room_content.get('enemies', [])
                    if enemy_names:                        
                        self._log(agent_id, f"[AÇÃO] COMBATE INICIADO com {enemy_names[0]}")
                        self._start_combat(agent_id, enemy_names[0])
                        reward += 10 
            else:
                # Movimento INVÁLIDO
                self._log(agent_id, f"[AÇÃO] Movimento INVÁLIDO (Sala {neighbor_index} não existe).")
                self.invalid_action_counts[agent_id] += 1
                reward = -5 

        # --- Ação 4: Equipar Item (Grimórios, Equipamentos e Artefatos) ---        
        elif action == 4:
            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} tentou Ação 4 (Equipar/Aprender).")
            
            # Decrementa o CD das skills do agentes
            self._tick_world_cooldowns(agent_id)
            
            # Segurança de nó
            if not current_graph.has_node(current_node_id) or 'content' not in current_graph.nodes[current_node_id]:
                 self.invalid_action_counts[agent_id] += 1
                 return -5.0, terminated

            room_items = current_graph.nodes[current_node_id]['content'].setdefault('items', [])
            
            item_consumed = False # Flag para saber se já fez algo nesta ação

            # Tenta aprender primeiro skills
            for item_name in room_items:
                details = self.catalogs['equipment'].get(item_name, {})
                if details.get('type') == 'SkillTome':
                    new_skill_name = details.get('skill_name')
                    
                    # Verifica se já conhece a skill
                    if new_skill_name not in self.agent_states[agent_id]['skills']:
                        
                        # Heurística de Substituição:
                        # Tenta substituir skills básicas primeiro, preserva slots avançados se possível
                        current_skills = self.agent_states[agent_id]['skills']
                        slot_to_replace = -1
                        
                        # Prioridade 1: Substituir 'Quick Strike' ou 'Heavy Blow' (Básicas)
                        for i in range(3): # Olha slots 0, 1, 2 (O 3 geralmente é Wait/Utility)
                            if current_skills[i] in ['Quick Strike', 'Heavy Blow']:
                                slot_to_replace = i
                                break
                        
                        # Prioridade 2: Se não tiver básicas, substitui aleatório entre 0 e 2
                        if slot_to_replace == -1:
                            import random
                            slot_to_replace = random.randint(0, 2)
                        
                        old_skill = current_skills[slot_to_replace]
                        
                        # EFETUA O APRENDIZADO
                        self.agent_states[agent_id]['skills'][slot_to_replace] = new_skill_name
                        self.agent_states[agent_id]['cooldowns'][new_skill_name] = 0 # Zera CD

                        # Incrementa o contador de upgrade de skills
                        self.skill_upgrades_this_episode[agent_id] += 1
                        
                        # Remove o livro
                        room_items.remove(item_name)
                        
                        self._log(agent_id, f"[UPGRADE] Aprendeu '{new_skill_name}' (Raridade: {details.get('rarity')}) substituindo '{old_skill}'.")
                        
                        reward += 100.0 # Grande incentivo para evoluir
                        item_consumed = True
                        break # Uma ação = Um livro lido
                        
            # Equipar armas ou armaduras (agora que o livro já foi)
            if not item_consumed:
                best_item_to_equip = None
                max_rarity_diff = 0.0 
                best_item_type = None
                best_diff = 0.0 # Inicializa para evitar erro

                for item_name in room_items:
                    details = self.catalogs['equipment'].get(item_name)
                    
                    # Filtra apenas equipamentos
                    if details and details.get('type') in ['Weapon', 'Armor', 'Artifact']:
                        item_type = details.get('type')
                        floor_rarity = self.rarity_map.get(details.get('rarity'), 0.0)
                        
                        # Compara com o que já tem
                        equipped_name = self.agent_states[agent_id]['equipment'].get(item_type)
                        equipped_rarity = 0.0
                        if equipped_name:
                            r_str = self.catalogs['equipment'].get(equipped_name, {}).get('rarity')
                            equipped_rarity = self.rarity_map.get(r_str, 0.0)
                        
                        diff = floor_rarity - equipped_rarity
                        
                        # Lógica Social: Na Arena, pega qualquer artefato (para barganha)
                        is_social_artifact = (item_type == 'Artifact' and is_in_arena)
                        
                        # Critério de Troca
                        if diff > max_rarity_diff or (is_social_artifact and max_rarity_diff <= 0.0):
                            max_rarity_diff = diff
                            best_item_to_equip = item_name
                            best_item_type = item_type
                            best_diff = diff

                if best_item_to_equip:
                    # Realiza a troca
                    self.agent_states[agent_id]['equipment'][best_item_type] = best_item_to_equip
                    room_items.remove(best_item_to_equip)
                    
                    # Métricas e Logs
                    self.equipment_swaps_this_episode[agent_id] += 1
                    qualidade = self.catalogs['equipment'][best_item_to_equip]['rarity']
                    
                    if best_item_to_equip not in self.sanctum_dropped_history[agent_id]:
                        if best_diff > 0:
                            self._log(agent_id, f"[UPGRADE] {self.agent_names[agent_id]} trocou para: '{best_item_to_equip}' (Tipo: {best_item_type}, Raridade: {qualidade}).")
                        else:
                            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} equipou: '{best_item_to_equip}' (Tipo: {best_item_type}, Raridade: {qualidade}).")

                        
                        # Calcula a recompensa para todos os itens
                        reward = 75 + (max_rarity_diff * 100)
                    if is_in_arena:
                        reward -= 2.0

                    # Lógica específica para artefatos (tokens sociais)
                    if best_item_type == 'Artifact':
                        # Verifica se é um item que ele mesmo descartou
                        is_recycled_loot = is_in_arena and (best_item_to_equip in self.sanctum_dropped_history[agent_id])

                        if is_recycled_loot:
                            # Se pegou o próprio item de volta, cancela o sinal de paz
                            reward = -2.0 
                            self.social_flags[agent_id]['just_picked_up'] = False # Não conta como "pegar oferta"
                            self.arena_interaction_state[agent_id]['offered_peace'] = False # Cancela minha oferta
                            
                            self._log(agent_id, f"[SOCIAL] Pegou seu equipamento de volta. Sinal de paz cancelado.")
                        else:
                            # Item novo/do oponente
                            self.social_flags[agent_id]['just_picked_up'] = True
                
                else:
                    # Falhou: Não tinha nada útil ou sala vazia
                    if room_items:
                        reward = -10.0 
                    else:
                        self.invalid_action_counts[agent_id] += 1                    
                        reward = -5.0

        # Ação 5: Dropar Artefato      
        elif action == 5:
            if not is_in_arena:
                self._log(agent_id, "[AÇÃO] Ação 5 inválida fora do Santuário.")
                self.invalid_action_counts[agent_id] += 1
                reward = -2.0
            else:
                equipped = self.agent_states[agent_id]['equipment'].get('Artifact')
                if equipped:
                    del self.agent_states[agent_id]['equipment']['Artifact']
                    room_items = current_graph.nodes[current_node_id]['content'].setdefault('items', [])
                    room_items.append(equipped)

                    # Registra que este agente já "teve" este item nesta sessão de arena
                    self.sanctum_dropped_history[agent_id].add(equipped)
                    
                    # Flag de drop para barganha
                    self.social_flags[agent_id]['just_dropped'] = True                    
                    
                    self._log(agent_id, f"[AÇÃO-ARENA] Dropou '{equipped}'.")

                    if is_in_arena:
                        reward -= 2.0 # Custa pontos ficar soltando                    
                else:
                    self.invalid_action_counts[agent_id] += 1
                    reward = -5.0
        
        # Outras Ações
        elif 0 <= action <= 3:                
            self.invalid_action_counts[agent_id] += 1
            reward = -5.0
        else:
            self.invalid_action_counts[agent_id] += 1
            reward = -5.0

        return reward, terminated    
    
    def _start_combat(self, agent_id: str, enemy_name: str):
        """
        Inicializa o estado de combate PvE para um agente específico.
        Cria cópias 'combatant' do agente e do inimigo, aplicando Efeitos de Sala.
        """
        
        # Pega o estado mestre do agente e localização
        agent_main_state = self.agent_states[agent_id]
        current_node = self.current_nodes[agent_id]
                
        # Recupera os dados da sala atual no grafo PvE
        room_data = self.graphs[agent_id].nodes[current_node]
        room_effects = room_data.get('content', {}).get('room_effects', [])
        active_effect = room_effects[0] if room_effects else None

        # Pega os status acumulados do agente
        stats_source = agent_main_state['base_stats']

        # Inicializa o agente
        agent_combatant = combat.initialize_combatant(
            name=self.agent_names[agent_id], 
            hp=agent_main_state['hp'], 
            equipment=list(agent_main_state.get('equipment', {}).values()), 
            skills=agent_main_state['skills'], 
            team=1, 
            level=agent_main_state['level'],
            catalogs=self.catalogs,
            room_effect_name=active_effect,                    
            base_damage=stats_source.get('damage', 0),
            base_defense=stats_source.get('defense', 0),
            base_evasion=stats_source.get('evasion', 0),
            base_accuracy=stats_source.get('accuracy', 1.0),
            base_crit=stats_source.get('crit_chance', 0.05)
        )
        
        # Copia e mantem cooldowns atuais (que vieram da exploração)
        current_skills = agent_main_state['skills']
        
        # Garante que o dict existe
        if 'cooldowns' not in agent_main_state:
             agent_main_state['cooldowns'] = {s: 0 for s in current_skills}

        # Copia o CD atual        
        agent_combatant['cooldowns'] = agent_main_state['cooldowns'].copy()
        
        # Inicializa o inimigo
        enemy_combatant = progression.instantiate_enemy(
             enemy_name=enemy_name,
             agent_current_floor=self.current_floors[agent_id],
             catalogs=self.catalogs,             
             room_effect_name=active_effect 
        )

        # Armazena o estado de combate
        self.combat_states[agent_id] = {
            'agent': agent_combatant,
            'enemy': enemy_combatant,
            'start_step': self.current_step
        }
        
        self._log(
            agent_id,
            f"[COMBATE] {self.agent_names[agent_id]} (Nível {agent_main_state['level']}, HP {agent_combatant['hp']}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {enemy_combatant['hp']}) "
            f"[Efeito: {active_effect if active_effect else 'Nenhum'}]"
        )

    def _initiate_pvp_combat(self, attacker_id: str, defender_id: str):
        """
        Inicializa o combate PvP e cria uma sessão compartilhada para os dois agentes.
        Atualizado para incluir status base (evolução do agente).
        """
        # Pega os estados mestres
        attacker_main_state = self.agent_states[attacker_id]
        defender_main_state = self.agent_states[defender_id]

        # Pega o efeito da sala atual
        current_node = self.current_nodes[attacker_id]        
        current_arena = self.arena_instances[attacker_id]
        room_data = current_arena.nodes[current_node]
        room_effects = room_data.get('content', {}).get('room_effects', [])
        active_effect = room_effects[0] if room_effects else None
        
        # Configuração do Atacante
        att_skills = attacker_main_state['skills']
        att_default_cd = {s: 0 for s in att_skills}
        att_stats = attacker_main_state['base_stats'] # Pega os status acumulados

        attacker_combatant = combat.initialize_combatant(
            name=self.agent_names[attacker_id], 
            hp=attacker_main_state['hp'], 
            equipment=list(attacker_main_state.get('equipment', {}).values()), 
            skills=att_skills,
            team=1, 
            level=attacker_main_state['level'],
            catalogs=self.catalogs,
            room_effect_name=active_effect,            
            base_damage=att_stats.get('damage', 0),
            base_defense=att_stats.get('defense', 0),
            base_evasion=att_stats.get('evasion', 0),
            base_accuracy=att_stats.get('accuracy', 1.0),
            base_crit=att_stats.get('crit_chance', 0.05)
        )
        # Carrega cooldowns salvos ou usa o default zerado
        attacker_combatant['cooldowns'] = attacker_main_state.get('cooldowns', att_default_cd).copy()
        
        # Configuração do Defensor
        def_skills = defender_main_state['skills']
        def_default_cd = {s: 0 for s in def_skills}
        def_stats = defender_main_state['base_stats'] # Pega os status acumulados

        defender_combatant = combat.initialize_combatant(
            name=self.agent_names[defender_id], 
            hp=defender_main_state['hp'], 
            equipment=list(defender_main_state.get('equipment', {}).values()), 
            skills=def_skills,
            team=2, 
            level=defender_main_state['level'],
            catalogs=self.catalogs,
            room_effect_name=active_effect,            
            base_damage=def_stats.get('damage', 0),
            base_defense=def_stats.get('defense', 0),
            base_evasion=def_stats.get('evasion', 0),
            base_accuracy=def_stats.get('accuracy', 1.0),
            base_crit=def_stats.get('crit_chance', 0.05)
        )
        # Carrega cooldowns salvos ou usa o default zerado
        defender_combatant['cooldowns'] = defender_main_state.get('cooldowns', def_default_cd).copy()

        # Cria o objeto de estado da sessão
        combat_session = {
            'a1_id': attacker_id, 
            'a2_id': defender_id,
            'a1': attacker_combatant, 
            'a2': defender_combatant, 
            'start_step': self.current_step
        }
        
        # Registra a sessão
        self.pvp_sessions[attacker_id] = combat_session
        self.pvp_sessions[defender_id] = combat_session
        
        # Atualiza estatísticas
        self.pvp_combats_this_episode[attacker_id] += 1
        self.pvp_combats_this_episode[defender_id] += 1        
        
        msg = f"[PVP] COMBATE INICIADO: {self.agent_names[attacker_id]} atacou {self.agent_names[defender_id]}!"
        self._log(attacker_id, msg)
        self._log(defender_id, msg)

    def _resolve_pvp_end_karma(self, winner_id: str, loser_id: str) -> float:
        """
        Calcula as consequências morais do duelo.
        Retorna: Um valor float para ajustar a recompensa do vencedor (bônus ou multa).
        """
        winner_level = self.agent_states[winner_id]['level']
        loser_level = self.agent_states[loser_id]['level']
        
        diff = winner_level - loser_level
        COWARDICE_THRESHOLD = 10
        
        # Ajuste base da recompensa
        reward_adjustment = 0.0

        # Caso 1: Covardia (Smurfing)
        if diff > COWARDICE_THRESHOLD:
            # Fator de Escalabilidade: Quão covarde foi?
            # Ex: Diff 11 -> excess=1 -> scale=1
            # Ex: Diff 50 -> excess=40 -> scale=4
            excess = diff - COWARDICE_THRESHOLD
            scale = 1 + (excess // 10) 
            
            # Punição no Karma (Repetição do update negativo)
            # Quanto maior a covardia, mais "fundo" ele afunda no disco de Poincaré
            for _ in range(scale):
                self.reputation_system.update_karma(winner_id, 'bad')
            
            # Punição na Recompensa (Multa Moral)
            # Se a vitória dá +200, uma multa de -100 * scale anula o lucro rápido.
            penalty = 50.0 * scale
            reward_adjustment = -penalty
            
            self._log(winner_id, f"[SOCIAL] COVARDIA ESCALÁVEL (x{scale})! Nível {winner_level} vs {loser_level}. Karma (---) Reward ({reward_adjustment})")
            self.cowardice_kills_this_episode[winner_id] += 1

        # Caso 2: Virada épica
        elif diff < -COWARDICE_THRESHOLD:
            # Fator de Escalabilidade: Quão heroico foi?
            excess = abs(diff) - COWARDICE_THRESHOLD
            scale = 1 + (excess // 10)
            
            # Bônus de Karma
            for _ in range(scale):
                self.reputation_system.update_karma(winner_id, 'good')
            
            # Bônus de Recompensa (Glória)
            bonus = 100.0 * scale
            reward_adjustment = bonus
            
            self._log(winner_id, f"[SOCIAL] VITÓRIA HEROICA (x{scale})! Nível {winner_level} vs {loser_level}. Karma (+++) Reward (+{reward_adjustment})")

        # Caso 3: Luta justa
        else:
            self.reputation_system.update_karma(winner_id, 'neutral')
            self._log(winner_id, f"[SOCIAL] Luta Justa. (Nível {winner_level} vs {loser_level}). Karma (Neutro)")
            
        return reward_adjustment

    def _end_arena_encounter(self, agent_id: str):
        """
        Finaliza a participação de um agente na Santuário e o move para a próxima P-Zone.
        
        Chamado após uma vitória em PvP ou uma barganha bem-sucedida.
        """        

        self._log(agent_id, f"[SANCTUM] {self.agent_names[agent_id]} saiu do santuário.")
        
        # Remove o agente da arena
        self.agents_in_arena.discard(agent_id)
        
        # Recupera o andar de onde o agente veio
        last_pve_floor = self.pve_return_floor.get(agent_id)

        # Padrão (caso o if falhe)
        next_p_floor = self.current_floors[agent_id] + 1 
        
        if last_pve_floor is not None:
            # O próximo andar deve ser o PVE seguinte ao que ele estava
            next_p_floor = last_pve_floor + 1 # Se arena é K, então o andar do agente é K+1
            # Limpa a memória
            del self.pve_return_floor[agent_id]
        
        # Encontra/Cria o nó inicial da próxima P-Zone            
        # Garante que o contador para este novo andar exista no grafo do agente
        if next_p_floor not in self.nodes_per_floor_counters[agent_id]:
            self.nodes_per_floor_counters[agent_id][next_p_floor] = 0
            
        next_node_index = self.nodes_per_floor_counters[agent_id][next_p_floor]
        next_p_node_id = f"p_{next_p_floor}_{next_node_index}"
        
        # Adiciona este novo nó ao grafo de progressão individual do agente
        agent_graph = self.graphs[agent_id]
        agent_graph.add_node(next_p_node_id, floor=next_p_floor)        
        
        # Atualiza o contador de nós para este andar
        self.nodes_per_floor_counters[agent_id][next_p_floor] += 1

        # Move o agente para este novo nó
        self.current_nodes[agent_id] = next_p_node_id
        self.current_floors[agent_id] = next_p_floor
        
        # Popula o novo nó com conteúdo (vazio, pois é um "hub" de entrada)
        start_content = content_generation.generate_room_content(
            catalogs=self.catalogs, 
            current_floor=0,
            budget_multiplier=0.0, # Multiplicador 0 garante budget 0 (sem itens/inimigos) se a fórmula for multiplicativa
            guarantee_enemy=False
        )
        agent_graph.nodes[next_p_node_id]['content'] = start_content
        
        # Gera os primeiros sucessores da P-Zone (ex: p_22_0, p_22_1)
        self._generate_and_populate_successors(agent_id, next_p_node_id)        

        # Limpeza dos agentes     
        # Limpa o registro do pareamento (Se o agente estava em uma partida)
        if agent_id in self.active_matches:
            opponent_id = self.active_matches.get(agent_id)
            
            # Remove a referência do oponente
            if opponent_id:
                self.active_matches.pop(opponent_id, None) # Remove o registro do par, se existir
            
            # Remove o registro do agente atual (que está saindo)
            self.active_matches.pop(agent_id, None)

        # Remove a referência da instância da arena
        if agent_id in self.arena_instances:
            del self.arena_instances[agent_id]        

        # Limpa a quantidade de passos do agente na arena
        if agent_id in self.arena_entry_steps:
            del self.arena_entry_steps[agent_id]

    def _respawn_agent(self, agent_id: str, cause: str = "Desconhecida"):
        """
        Processa a "morte" de um agente e limpa TODOS os vestígios dele na Arena.
        """
        
        # Loga a Morte
        self._log(agent_id, f"[MORTE] {self.agent_names[agent_id]} foi derrotado por {cause}! Resetando...")

        # Preserva Identidade e Karma
        agent_name = self.agent_names[agent_id]
        preserved_karma_z = self.reputation_system.get_karma_state(agent_id)
        
        # Reseta o estado Mestre
        self.agent_states[agent_id] = progression.create_initial_agent(agent_name)

        # Reseta efeitos de status
        self.agent_states[agent_id]['effects'] = {}
        self.agent_states[agent_id]['persistent_effects'] = {}
        
        # RE-APLICA Karma
        self.agent_states[agent_id]['karma']['real'] = preserved_karma_z.real
        self.agent_states[agent_id]['karma']['imag'] = preserved_karma_z.imag
        
        # Reseta métricas da Run
        self.current_episode_logs[agent_id] = []
        self.enemies_defeated_this_episode[agent_id] = 0
        self.damage_dealt_this_episode[agent_id] = 0.0
        self.equipment_swaps_this_episode[agent_id] = 0
        self.skill_upgrades_this_episode[agent_id] = 0
        self.max_floor_reached_this_episode[agent_id] = 0
        
        self.arena_encounters_this_episode[agent_id] = 0
        self.pvp_combats_this_episode[agent_id] = 0
        self.bargains_succeeded_this_episode[agent_id] = 0        

        self.bargains_trade_this_episode[agent_id] = 0 # Troca de item
        self.bargains_toll_this_episode[agent_id] = 0  # Saída pacífica
        self.betrayals_this_episode[agent_id] = 0
        self.pvp_combat_durations[agent_id] = []
        self.pve_combat_durations[agent_id] = []
        self.arena_entry_steps[agent_id] = 0
        self.previous_nodes[agent_id] = []
        self.pve_return_floor[agent_id] = []
        
        # Limpa estados da arena (caso tenha fantasmas)
        self.combat_states[agent_id] = None 
        
        # Limpa sessão PvP
        if agent_id in self.pvp_sessions:
             session = self.pvp_sessions[agent_id]
             p1 = session['a1_id']
             p2 = session['a2_id']
             self.pvp_sessions.pop(p1, None)
             self.pvp_sessions.pop(p2, None)

        # Limpa pareamento (Active Match)
        if agent_id in self.active_matches:
            opponent_id = self.active_matches[agent_id]
            # Desfaz o par para o oponente também
            if opponent_id in self.active_matches:
                del self.active_matches[opponent_id]                
                # No design atual, o vencedor sai via _end_arena, então ok.
            del self.active_matches[agent_id]

        # Limpa instância de arena
        if agent_id in self.arena_instances:
            del self.arena_instances[agent_id] 

        # Limpa fila de espera
        if agent_id in self.matchmaking_queue:
            self.matchmaking_queue.remove(agent_id)

        # Limpa lista de presença
        if agent_id in self.agents_in_arena:
            self.agents_in_arena.remove(agent_id)

        # Reinicia a lista de itens descartados no Santuário
        self.sanctum_dropped_history[agent_id] = set()

        # Cria uma nova P-Zone (Volta para casa)
        self.current_floors[agent_id] = 0
        self.current_nodes[agent_id] = "start" # <--- Força a posição para o início
        self.nodes_per_floor_counters[agent_id] = {0: 1}
        self.graphs[agent_id] = nx.DiGraph()
        self.graphs[agent_id].add_node("start", floor=0)

        self._log(agent_id, f"[RESPAWN] {agent_name} (Nível 1) voltou a estaca zero.")
        
        # Popula 
        start_content = content_generation.generate_room_content(
            self.catalogs, budget_multiplier=0, current_floor=0, guarantee_enemy=False
        )
        self.graphs[agent_id].nodes["start"]['content'] = start_content
        self._generate_and_populate_successors(agent_id, "start")
        
    def _tick_world_cooldowns(self, agent_id: str):
        """
        Simula a passagem do tempo fora de combate.
        Reduz em 1 o Cooldown de todas as skills ativas.
        """
        agent_state = self.agent_states[agent_id]
        
        # Se não tiver cooldowns inicializados, ignora
        if 'cooldowns' not in agent_state:
            return

        for skill, current_cd in agent_state['cooldowns'].items():
            if current_cd > 0:
                agent_state['cooldowns'][skill] = max(0, current_cd - 1)