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
from . import agent_rules
from . import combat
from . import content_generation
from . import map_generation
from . import reputation
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
            'skills': skill_data['skill_catalog'], 'effects': skill_data['effect_ruleset'],
            'equipment': equipment_data['equipment_catalog'], 'enemies': enemy_data['pools']['enemies'],
            'room_effects': enemy_data['pools']['room_effects'], 'events': enemy_data['pools']['events']
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

        # Parâmetros globais
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.verbose = verbose 
        self.max_floors = max_floors
        self.max_level = max_level         
        self.budget_multiplier = budget_multiplier
        self.guarantee_enemy = guarantee_enemy
        self.pool_costs = content_generation._calculate_costs(enemy_data['pools'])
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
        
        self.agent_skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]
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
                
        # Bloco Próprio (7) + Contexto PvE (7) + Contexto Social (10) + 
        # Vizinhos (16) + Self-Equip/Flags (6) = 46 Total
        OBS_SHAPE = (46,)
        
        # --- Detalhamento do Espaço de Observação (36 estados) ---
        # Bloco Próprio (7 estados):
        # 0: HP Ratio, 1: Level Ratio, 2: EXP Ratio
        # 3: Skill 0 Cooldown, 4: Skill 1 Cooldown, 5: Skill 2 Cooldown, 6: Skill 3 (Wait) Cooldown
        #
        # Bloco Contexto PvE (7 estados):
        # 7: Flag: In Combat (PvE)?
        # 8: Flag: Item/Evento na sala?
        # 9: Flag: Sala Vazia/Segura?
        # 10: Combat: HP Ratio Inimigo PvE
        # 11: Combat: Level Ratio Inimigo PvE
        # 12: Floor: Flag: Equipamento no chão?
        # 13: Floor: Raridade Equipamento
        #
        # Bloco Contexto Social/PvP (10 estados):
        # 14: Flag: Está em Zona K (Arena)?
        # 15: Flag: Outro Agente (Player) na sala?
        # 16: PvP: HP Ratio Outro Agente
        # 17: PvP: Diferença de Nível
        # 18: PvP: Karma Outro Agente (Real)
        # 19: PvP: Karma Outro Agente (Imag)
        # 20: Social: Flag: Artefato no chão?
        # 21: Social: Flag: Eu possuo Artefato?
        # 22: Social: Flag (Evento): Outro Agente dropou? (1-step)
        # 23: Social: Flag: Em combate PvP comigo?
        #
        # Bloco Contexto Movimento (12 estados = 4 vizinhos * 3 features)
        # (is_valid, has_enemy, has_reward)
        # 24-26: Vizinho 0 (is_valid, has_enemy, has_reward)
        # 27-29: Vizinho 1 (is_valid, has_enemy, has_reward)
        # 30-32: Vizinho 2 (is_valid, has_enemy, has_reward)
        # 33-35: Vizinho 3 (is_valid, has_enemy, has_reward)        
        #
        # Bloco Contexto Self-Equip (2 estados):
        # 36: Self: Raridade da Arma *Equipada* (0.0 a 1.0)
        # 37: Self: Raridade da Armadura *Equipada* (0.0 a 1.0)
        # 38: Social: Score de equipamentos do outro agente (0.0 a 1.0)
        #   
        # Flags Sociais Adicionais (2 estados):
        # 39: Flag: Oponente acabou de dropar item? (1.0 se sim)
        # 40: Flag: Oponente "pulou" ataque (estava na sala e não atacou)? (1.0 se sim)
        # 
        # 41: Flag Global: "A heavy door has opened..." 
        # (1.0 = Encontro ocorreu, saída liberada. -1.0 = Saída trancada).

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
                
                # Calcula o budget
                budget = ( 100 + (next_floor * 10) ) * self.budget_multiplier
                
                # Chamar o módulo de conteúdo
                content = content_generation.generate_room_content(
                    self.catalogs,                 
                    budget, 
                    next_floor,
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
        Coleta e retorna a observação de 46 estados para o agente especificado.        
        """
        # Começa com -1.0 (default)
        obs = np.full(self.observation_space[agent_id].shape, -1.0, dtype=np.float32) 
        
        if agent_id not in self.agent_states:
            return obs

        agent = self.agent_states[agent_id]
        current_node_id = self.current_nodes.get(agent_id) 
        
        # Verifica o contexto do agente
        is_in_arena = agent_id in self.arena_instances
        
        # Seleciona o grafo correto
        if is_in_arena:
            current_graph_to_use = self.arena_instances[agent_id]
        else:
            # Pega do dicionário de grafos PvE
            current_graph_to_use = self.graphs.get(agent_id)

        # Identifica a sala atual
        current_node_id = self.current_nodes.get(agent_id)

        # Busca os nós vizinhos
        neighbors = []
        if current_graph_to_use and current_node_id:
            if current_graph_to_use.has_node(current_node_id):
                # Pega a lista de IDs dos nós disponíveis
                neighbors = list(current_graph_to_use.neighbors(current_node_id))
                # Garante que a ordem sempre seja a mesma
                neighbors.sort() 
        
        # Identifica o Outro Agente (para contexto social)
        other_agent_id = self.active_matches.get(agent_id)

        room_content = current_graph_to_use.nodes[current_node_id].get('content', {})
        pve_combat_state = self.combat_states.get(agent_id)
        
        # Bloco próprio (0-6)
        obs[0] = (agent['hp'] / agent['max_hp']) * 2 - 1 if agent['max_hp'] > 0 else -1.0
        obs[1] = (agent['level'] / self.max_level) * 2 - 1
        obs[2] = (agent['exp'] / agent['exp_to_level_up']) if agent['exp_to_level_up'] > 0 else 0.0
        for i in range(4):
            skill_name = self.agent_skill_names[i]
            max_cd = self.catalogs['skills'].get(skill_name, {}).get('cd', 1)
            current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
            obs[3 + i] = (current_cd / max_cd) if max_cd > 0 else 0.0

        # Bloco PvE (7-13)
        enemy_in_combat = pve_combat_state.get('enemy') if pve_combat_state else None
        items_on_floor = room_content.get('items', [])
        events_in_room = room_content.get('events', [])
        
        if enemy_in_combat:
            obs[7] = 1.0
            obs[10] = (enemy_in_combat['hp'] / enemy_in_combat['max_hp']) * 2 - 1 if enemy_in_combat['max_hp'] > 0 else -1.0
            obs[11] = (enemy_in_combat.get('level', 1) / self.max_level) * 2 - 1
        elif items_on_floor or (events_in_room and 'None' not in events_in_room):
            obs[8] = 1.0
        else:
            obs[9] = 1.0

        # Itens no Chão (Weapon/Armor)
        best_wep_arm_rarity = 0.0
        found_wep_arm = False
        for item_name in items_on_floor:
            if item_name in self.equipment_catalog_no_artifacts: 
                found_wep_arm = True
                rarity = self.rarity_map.get(self.catalogs['equipment'][item_name].get('rarity'), 0.0)
                if rarity > best_wep_arm_rarity:
                    best_wep_arm_rarity = rarity
        
        if found_wep_arm:
            obs[12] = 1.0
            obs[13] = best_wep_arm_rarity

        # Bloco Social/PvP (14-19 + 38-40)
        is_in_arena = (agent_id in self.arena_instances)
        obs[14] = 1.0 if is_in_arena else -1.0
        
        # Identificação do Oponente
        other_agent_id = self.active_matches.get(agent_id)
        
        if other_agent_id and is_in_arena:
            # Verifica se estão na mesma sala
            if self.current_nodes[agent_id] == self.current_nodes[other_agent_id]:
                other_agent_state = self.agent_states.get(other_agent_id)
                
                if other_agent_state:
                    obs[15] = 1.0 # Flag: Outro Agente na sala
                    obs[16] = (other_agent_state['hp'] / other_agent_state['max_hp']) * 2 - 1
                    obs[17] = (agent['level'] - other_agent_state['level']) / 100.0
                    
                    opponent_karma_z = self.reputation_system.get_karma_state(other_agent_id)
                    obs[18] = opponent_karma_z.real 
                    obs[19] = opponent_karma_z.imag

        # Artefatos (20-21)
        best_artifact_rarity = 0.0
        found_artifact_floor = False
        for item_name in items_on_floor:
            if item_name in self.artifact_catalog: 
                found_artifact_floor = True
                rarity = self.rarity_map.get(self.catalogs['equipment'][item_name].get('rarity'), 0.0)
                if rarity > best_artifact_rarity:
                    best_artifact_rarity = rarity
        
        if found_artifact_floor:
            obs[20] = 1.0
            obs[21] = best_artifact_rarity

        # Meu Artefato Equipado (22)
        my_equipped_artifact_name = agent['equipment'].get('Artifact')
        obs[22] = self.rarity_map.get(self.catalogs['equipment'][my_equipped_artifact_name].get('rarity'), 0.0) if my_equipped_artifact_name else 0.0

        # Em Combate PvP (23)
        if agent_id in self.pvp_sessions:
            obs[23] = 1.0

        # --- Bloco Contexto Movimento (24-35) ---
        # Agora são 4 features por vizinho: [Valido?, Inimigo?, Loot?, PERIGO?]
        # Índices: 
        # Vizinho 0: 24, 25, 26, 27
        # Vizinho 1: 28, 29, 30, 31
        # Vizinho 2: 32, 33, 34, 35
        # Vizinho 3: 36, 37, 38, 39
        
        base_idx = 24 
        
        for i in range(self.MAX_NEIGHBORS): # 0 a 3            
            idx = base_idx + (i * 4) 

            # Só processa se tiver vizinho e grafo válido
            if i < len(neighbors) and current_graph_to_use:
                neighbor_node_id = neighbors[i]
                
                # Double check: vizinho existe no grafo?
                if current_graph_to_use.has_node(neighbor_node_id):
                    neighbor_content = current_graph_to_use.nodes[neighbor_node_id].get('content', {})
                    
                    # [Feature 0] É Válido?
                    obs[idx + 0] = 1.0 
                    
                    # Verifica presença de inimigos e jogadores
                    enemy_list = neighbor_content.get('enemies', [])
                    has_opponent = (other_agent_id and self.current_nodes.get(other_agent_id) == neighbor_node_id)
                    
                    # [Feature 1] Tem Inimigo/Oponente?
                    if enemy_list or has_opponent:
                        obs[idx + 1] = 1.0 
                    
                    # [Feature 2] Tem Recompensa?
                    # (Loot, Eventos Positivos ou Saída destrancada na arena)
                    has_loot = neighbor_content.get('items')
                    has_good_event = any(evt in neighbor_content.get('events', []) for evt in ['Treasure', 'Morbid Treasure', 'Fountain of Life'])
                    
                    is_exit_open = False
                    if is_in_arena and current_graph_to_use.nodes[neighbor_node_id].get('is_exit', False):
                        if current_graph_to_use.graph.get('meet_occurred', False):
                            is_exit_open = True
                            
                    if has_loot or has_good_event or is_exit_open:
                        obs[idx + 2] = 1.0 

                    # Nível de Perigo (Danger Tier)
                    danger_level = 0.0
                    
                    # A. Analisa Monstros
                    if enemy_list:
                        enemy_name = enemy_list[0]
                        # Tenta pegar tags do catálogo
                        enemy_data = self.catalogs['enemies'].get(enemy_name, {})
                        tags = enemy_data.get('tags', [])
                        
                        if 'Boss' in tags: 
                            danger_level = 1.0   # PERIGO MÁXIMO (Vermelho / Lava Golem)
                        elif 'Elite' in tags: 
                            danger_level = 0.66  # PERIGO ALTO (Laranja)
                        else: 
                            danger_level = 0.33  # PERIGO MÉDIO (Amarelo / Mob comum)
                            
                    # Analisa Jogadores (Arena PvP)
                    elif has_opponent:
                        # Outro jogador é imprevisível = Elite
                        danger_level = 0.66
                    
                    obs[idx + 3] = danger_level                                
            
        # [Index 40] Self Weapon Rarity
        current_eq = self.agent_states[agent_id]['equipment']

        w_name = current_eq.get('Weapon')
        if w_name:
            r_str = self.catalogs['equipment'].get(w_name, {}).get('rarity', 'Common')
            obs[40] = self.rarity_map.get(r_str, 0.0)
            
        # [Index 41] Self Armor Rarity
        a_name = current_eq.get('Armor')
        if a_name:
            r_str = self.catalogs['equipment'].get(a_name, {}).get('rarity', 'Common')
            obs[41] = self.rarity_map.get(r_str, 0.0)

        # [Index 42] Social Score do Outro (Se visível)
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
                obs[42] = total_rarity / 2.0 # Média
        
        # [Index 43] Oponente dropou item recentemente?
        if other_agent_id and self.social_flags[other_agent_id].get('just_dropped'):
            obs[43] = 1.0
            
        # [Index 44] Oponente pulou ataque (sinal de paz)?
        if other_agent_id and self.social_flags[other_agent_id].get('skipped_attack'):
            obs[44] = 1.0
            
        # [Index 45] Encontro Ocorreu? (Global Flag)
        # Avisa ao agente que a porta está destrancada
        if is_in_arena and current_graph_to_use.graph.get('meet_occurred', False):
            obs[45] = 1.0
        elif is_in_arena: # Se está na arena mas não encontrou
            obs[45] = -1.0
        
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
            self.agent_states[agent_id] = agent_rules.create_initial_agent(agent_name) 
            self.agent_states[agent_id]['skills'] = self.agent_skill_names 
            
            # Garante artefato inicial para testar barganha
            self.agent_states[agent_id]['equipment']['Artifact'] = 'Amulet of Vigor'
            
            # Inicializa métricas individuais
            self.current_episode_logs[agent_id] = []
            self.enemies_defeated_this_episode[agent_id] = 0
            self.invalid_action_counts[agent_id] = 0
            self.last_milestone_floors[agent_id] = 0
            self.combat_states[agent_id] = None
            self.damage_dealt_this_episode[agent_id] = 0.0
            self.equipment_swaps_this_episode[agent_id] = 0
            self.death_cause[agent_id] = "Sobreviveu (Time Limit)"

            self.arena_encounters_this_episode[agent_id] = 0
            self.pvp_combats_this_episode[agent_id] = 0
            self.bargains_succeeded_this_episode[agent_id] = 0
            self.bargains_trade_this_episode[agent_id] = 0
            self.bargains_toll_this_episode[agent_id] = 0
            self.cowardice_kills_this_episode[agent_id] = 0
            self.betrayals_this_episode[agent_id] = 0
            self.karma_history[agent_id] = []

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
                budget=0,

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
            action_name = self.agent_skill_names[action] # Mapeia 0-3 para ["Quick Strike", ..., "Wait"]
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
            leveled_up = agent_rules.check_for_level_up(agent_main_state) 

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

            self.combat_states[agent_id] = None # Limpa o estado de combate deste agente
            combat_over = True
        
        # Se o combate não chegou ao fim, o inimigo age
        if not combat_over:
            hp_before_agent = agent['hp']
            
            # IA Simples 
            available_skills = [s for s, cd in enemy['cooldowns'].items() if cd == 0]
            enemy_action = random.choice(available_skills) if available_skills else "Wait"
            
            combat.execute_action(enemy, [agent], enemy_action, self.catalogs)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
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
        Log colorido no terminal, mais fácil de ler.
        Categorias são detectadas automaticamente: AÇÃO, PVP, KARMA, WARN, ERRO, etc.
        """
        # ANSI Color Codes
        COLORS = {
            "reset": "\033[0m",
            "agent": "\033[96m",     # ciano
            "action": "\033[92m",    # verde
            "pvp": "\033[91m",       # vermelho
            "karma_pos": "\033[92m", # verde
            "karma_neg": "\033[91m", # vermelho
            "karma_neu": "\033[93m", # amarelo
            "warn": "\033[93m",      # amarelo
            "error": "\033[91m",     # vermelho
            "arena": "\033[95m",     # magenta
            "map": "\033[94m",       # azul
        }

        def colorize(text, color):
            return f"{COLORS[color]}{text}{COLORS['reset']}"

        # Escolhe cor baseada no conteúdo da mensagem
        msg_upper = message.upper()

        if "[ERRO]" in msg_upper or "ERROR" in msg_upper:
            color = "error"
        elif "[WARN]" in msg_upper:
            color = "warn"
        elif "[SOCIAL]" in msg_upper:
            if " (+)" in message or "POSITIVO" in msg_upper:
                color = "karma_pos"
            elif "(-)" in message or "NEGAT" in msg_upper:
                color = "karma_neg"
            else:
                color = "karma_neu"
        elif "PVP" in msg_upper or "MORTE" in msg_upper:
            color = "pvp"
        elif "AÇÃO" in msg_upper or "AÇÃO-ARENA" in msg_upper:
            color = "action"
        elif "SANCTUM" in msg_upper or "ZONA K" in msg_upper:
            color = "arena"
        elif "PVE" in msg_upper:
            color = "map"
        else:
            color = "agent"

        formatted = f"{colorize(f'[{agent_id.upper()}]', 'agent')} {colorize(message, color)}"

        # Print no terminal
        if self.verbose > 0:
            print(formatted)

        # Log interno (sem cores)
        if agent_id not in self.current_episode_logs:
            self.current_episode_logs[agent_id] = []

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
                budget = (100 + (base_floor * 10)) * self.budget_multiplier
                content = content_generation.generate_room_content(
                    self.catalogs, budget=budget, current_floor=base_floor, guarantee_enemy=False 
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

        # 2. Processa Ação (Combate ou Exploração)
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
            {"check": lambda f: (f - 1) % sanctum_k == 0 and f > 1, "bonus": 400, "msg": "Sobreviveu à Zona K!"},

            # 2. Marco Decimal (Dispara a cada 10 andares: 10, 20, 30...)
            {"check": lambda f: f % 10 == 0 and f > 0, "bonus": 250, "msg": "Marco Decimal!"}
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
                act_p1 = self.agent_skill_names[actions[p1]] if 0 <= actions[p1] <= 3 else "Wait"
                act_p2 = self.agent_skill_names[actions[p2]] if 0 <= actions[p2] <= 3 else "Wait"

                # Resolve o turno PvP
                rew1, rew2, over, winner, loser = self._handle_pvp_combat_turn(session, act_p1, act_p2)
                
                rewards[p1] += rew1
                rewards[p2] += rew2
                
                if over:
                    # Logs e Métricas
                    duration = self.current_step - session['start_step']
                    self.pvp_combat_durations[p1].append(duration)
                    self.pvp_combat_durations[p2].append(duration)
                    
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
                                self._respawn_agent(agent_id)
                                
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
                        base_exit_reward = 50.0
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
                            bonus_peace_reward = 150.0 # Total 200 (igual Barganha)
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

        # Ação 4: Equipar Item (Universal + Social)
        elif action == 4:
            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} tentou Ação 4 (Equipar Item).")
            
            if not current_graph.has_node(current_node_id) or 'content' not in current_graph.nodes[current_node_id]:
                 self.invalid_action_counts[agent_id] += 1
                 return -5.0, terminated

            room_items = current_graph.nodes[current_node_id]['content'].setdefault('items', [])
            
            best_item_to_equip = None
            max_rarity_diff = 0.0 
            best_item_type = None

            for item_name in room_items:
                details = self.catalogs['equipment'].get(item_name)
                if details and details.get('type') in ['Weapon', 'Armor', 'Artifact']:
                    item_type = details.get('type')
                    floor_rarity = self.rarity_map.get(details.get('rarity'), 0.0)
                    
                    equipped_name = self.agent_states[agent_id]['equipment'].get(item_type)
                    equipped_rarity = 0.0
                    if equipped_name:
                        r_str = self.catalogs['equipment'].get(equipped_name, {}).get('rarity')
                        equipped_rarity = self.rarity_map.get(r_str, 0.0)
                    
                    diff = floor_rarity - equipped_rarity
                    
                    # Lógica Social: Aceita qualquer artefato SE estiver na Arena
                    is_social_artifact = (item_type == 'Artifact' and is_in_arena)
                    
                    # Equipa se for Upgrade OU se for um artefato social na arena
                    if diff > max_rarity_diff or (is_social_artifact and max_rarity_diff <= 0.0):
                        max_rarity_diff = diff
                        best_item_to_equip = item_name
                        best_item_type = item_type

            if best_item_to_equip:
                self.agent_states[agent_id]['equipment'][best_item_type] = best_item_to_equip
                room_items.remove(best_item_to_equip)
                
                # Registra a troca
                self.equipment_swaps_this_episode[agent_id] += 1
                qualidade = self.catalogs['equipment'][best_item_to_equip]['rarity']
                self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} equipou: '{best_item_to_equip}' (Tipo: {best_item_type}, Raridade: {qualidade}).")

                # Recompensa pela raridade do item pego
                base_pve_reward = 75 + (max_rarity_diff * 100)
                
                # Se for social, garante um mínimo de 10 pts mesmo se downgrade
                if is_in_arena and best_item_type == 'Artifact':
                    reward = max(10.0, base_pve_reward)
                else:
                    reward = base_pve_reward
                
                # Flag de pick-up para barganha
                if best_item_type == 'Artifact':
                    self.social_flags[agent_id]['just_picked_up'] = True                

            else:
                if room_items:
                    reward = -5.0 # Itens ignorados
                else:
                    self.invalid_action_counts[agent_id] += 1                    
                    reward = -2.0 # Chão vazio

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
        Cria cópias 'combatant' do agente e do inimigo.
        """
        
        # Pega o estado mestre do agente
        agent_main_state = self.agent_states[agent_id]
        
        # Inicializa o 'combatant' do AGENTE        
        agent_combatant = combat.initialize_combatant(
            name=self.agent_names[agent_id], 
            hp=agent_main_state['hp'], 
            equipment=list(agent_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=1, 
            catalogs=self.catalogs
        )
        # Copia os cooldowns atuais
        agent_combatant['cooldowns'] = agent_main_state.get('cooldowns', {s: 0 for s in self.agent_skill_names}).copy()

        # Inicializa o 'combatant' do INIMIGO        
        enemy_combatant = agent_rules.instantiate_enemy(
            enemy_name=enemy_name,
            agent_current_floor=self.current_floors[agent_id],
            catalogs=self.catalogs
        )

        # Fallback de Segurança (Caso o nome do inimigo esteja errado no YAML)
        if enemy_combatant is None:
            self._log(agent_id, f"[ERRO] Inimigo '{enemy_name}' não encontrado ou inválido. Gerando Dummy.")
            # Cria um boneco de treino fraco para não quebrar o jogo
            enemy_combatant = combat.initialize_combatant(
                name="Dummy", hp=50, equipment=[], skills=[], team=2, catalogs=self.catalogs
            )

        # 3. Armazena o estado de combate
        self.combat_states[agent_id] = {
            'agent': agent_combatant,
            'enemy': enemy_combatant,
            'start_step': self.current_step
        }
        
        # Loga o início
        self._log(
            agent_id,
            f"[COMBATE] {self.agent_names[agent_id]} (Nível {agent_main_state['level']}, HP {agent_combatant['hp']}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {enemy_combatant['hp']})"
        )

    def _initiate_pvp_combat(self, attacker_id: str, defender_id: str):
        """
        Inicializa o combate PvP e cria uma sessão compartilhada para os dois agentes.
        """
        # Pega os estados mestres
        attacker_main_state = self.agent_states[attacker_id]
        defender_main_state = self.agent_states[defender_id]

        # Inicializa combatants
        attacker_combatant = combat.initialize_combatant(
            name=self.agent_names[attacker_id], 
            hp=attacker_main_state['hp'], 
            equipment=list(attacker_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=1, catalogs=self.catalogs
        )
        attacker_combatant['cooldowns'] = attacker_main_state.get('cooldowns', {}).copy()

        defender_combatant = combat.initialize_combatant(
            name=self.agent_names[defender_id], 
            hp=defender_main_state['hp'], 
            equipment=list(defender_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=2, catalogs=self.catalogs
        )
        defender_combatant['cooldowns'] = defender_main_state.get('cooldowns', {}).copy()

        # Cria o objeto de estado da sessão
        combat_session = {
            'a1_id': attacker_id, # Guardamos os IDs reais para saber quem é quem
            'a2_id': defender_id,
            'a1': attacker_combatant, # Combatant do Atacante
            'a2': defender_combatant, # Combatant do Defensor
            'start_step': self.current_step
        }
        
        # 4. Registra a sessão para AMBOS os agentes
        self.pvp_sessions[attacker_id] = combat_session
        self.pvp_sessions[defender_id] = combat_session
        
        # Atualiza estatísticas
        self.pvp_combats_this_episode[attacker_id] += 1
        self.pvp_combats_this_episode[defender_id] += 1        
        
        self._log(attacker_id, f"[PVP] {self.agent_names[attacker_id]} iniciou ataque contra {self.agent_names[defender_id]}!")
        self._log(defender_id, f"[PVP] {self.agent_names[defender_id]} está sendo atacado por {self.agent_names[attacker_id]}!")

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
            self.catalogs, budget=0, current_floor=next_p_floor, guarantee_enemy=False
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
        self.agent_states[agent_id] = agent_rules.create_initial_agent(agent_name)
        self.agent_states[agent_id]['skills'] = self.agent_skill_names
        self.agent_states[agent_id]['equipment']['Artifact'] = 'Amulet of Vigor'
        
        # RE-APLICA Karma
        self.agent_states[agent_id]['karma']['real'] = preserved_karma_z.real
        self.agent_states[agent_id]['karma']['imag'] = preserved_karma_z.imag
        
        # Reseta métricas da Run
        self.current_episode_logs[agent_id] = []
        self.enemies_defeated_this_episode[agent_id] = 0
        self.damage_dealt_this_episode[agent_id] = 0.0
        self.equipment_swaps_this_episode[agent_id] = 0
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

        # Cria uma nova P-Zone (Volta para casa)
        self.current_floors[agent_id] = 0
        self.current_nodes[agent_id] = "start" # <--- Força a posição para o início
        self.nodes_per_floor_counters[agent_id] = {0: 1}
        self.graphs[agent_id] = nx.DiGraph()
        self.graphs[agent_id].add_node("start", floor=0)

        self._log(agent_id, f"[RESPAWN] {agent_name} (Nível 1) voltou a estaca zero.")
        
        # Popula 
        start_content = content_generation.generate_room_content(
            self.catalogs, budget=0, current_floor=0, guarantee_enemy=False
        )
        self.graphs[agent_id].nodes["start"]['content'] = start_content
        self._generate_and_populate_successors(agent_id, "start")
        