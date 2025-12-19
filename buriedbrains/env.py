# buriedbrains/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
import networkx as nx

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

# -------------------------------------------
# CONSTANTES DE RECOMPENSA (REWARD SHAPING)
# -------------------------------------------

# Existência e Progresso
REW_EXISTENCE_PVE = -0.5       # Custo por turno (Time Step) no PvE
REW_EXISTENCE_PVP = -0.1       # Custo base por turno na Arena (sem pressão)
REW_VICTORY = 1000.0           # Chegar ao final do jogo
REW_DEATH = -300.0             # Morte (PvE ou PvP)
REW_INVALID_ACTION = -5.0      # Colisão, ação impossível, slot vazio

# Exploração e Movimento
REW_NEW_ROOM_PVE = 0.5         # Entrar em sala nova (PvE)
REW_NEW_ROOM_PVP = 0.5         # Entrar em sala nova (PvP)
REW_PING_PONG = -1.5           # Voltar para a sala de onde veio imediatamente
REW_MOVE_PVE_SUCCESS = 5.0     # Recompensa alta por mover com sucesso no PvE
REW_COMBAT_START = 10.0        # Incentivo para iniciar combate PvE

# Combate PvE
REW_DMG_DEALT_SCALAR = 0.6     # Multiplicador do Dano Causado
REW_DMG_TAKEN_SCALAR = 0.5     # Multiplicador do Dano Sofrido
REW_KILL_ENEMY = 100.0         # Matar Mob PvE
REW_LEVEL_UP = 50.0            # Subir de Nível

# Itens e Equipamento
REW_LEARN_SKILL = 100.0        # Aprender Grimório
REW_EQUIP_BASE = 75.0          # Base por equipar item melhor
REW_EQUIP_RARITY_MULT = 100.0  # Multiplicador por diferença de raridade
REW_EMPTY_LOOT = -10.0         # Tentar pegar item onde não tem

# Social / PvP
REW_PVP_WIN = 200.0            # Vitória no PvP
REW_PVP_LOSS = -300.0          # Derrota no PvP (Equivale a Morte)
REW_TRADE_SUCESS = 200.0    # Troca pacífica de itens
REW_EXIT_PEACE_BASE = 20.0     # Sair da Arena (Pedágio) - Base
REW_EXIT_PEACE_BONUS = 80.0    # Sair da Arena (Pedágio) - Bônus se houve paz
REW_MEET_BONUS = 100.0         # Encontrar o oponente (destrancar porta)
REW_TRIBUTE_PAID = -50.0        # Pagar tributo (Custo)
COMBAT_DETAILS = True          # Se os detalhes do combate PvP são logados no Terminal

# Penalidades Sociais
REW_BETRAYAL = -100.0          # Traição / Perfídia
REW_SIGNAL_COST = -2.0         # Custo de Drop/Sinalização (evitar spam)

# Purgo do Santuário (pune agentes que ficam muito tempo lá dentro)
REW_SANCTUM_PRESSURE_BASE = -0.1 # Multiplicador inicial da pressão
REW_SANCTUM_PRESSURE_CAP = -10.0 # Teto máximo da punição por turno

# Dinâmica de Recompensas por Força do Inimigo
REW_SMURF_PENALTY_BASE = 50.0  # Penalidade base por matar fracos
REW_HEROIC_BONUS_BASE = 100.0  # Bônus base por matar fortes

# Marcos de Andares
MILESTONES_EACH = 10          # A cada quantos andares
REW_MILESTONE_DECIMAL = 100.0  # Multiplicador por marco alcançado

# LOGGING
DEATH_LOG_CUTOFF = 300         # Quantas linhas do log final mostrar após a morte
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

        # Instancia o encoder de skills
        self.skill_encoder = skill_encoder.SkillEncoder(self.catalogs['skills'], self.catalogs['effects'])

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
        # Ações: 0-3 (Skills), 4 (Equipar), 5-8 (Drops), 9-12 (Mover)
        # 0: Quick Strike, 1: Heavy Blow, 2: Stone Shield, 3: Wait
        # 4: Equipar Item / Aprender Skill
        # 5: Drop Weapon (Sinal Alto Risco)
        # 6: Drop Armor (Sinal Alto Risco)
        # 7: Drop Artifact (Sinal Médio Risco)
        # 8: Drop Consumable (Sinal Baixo Risco)
        # 9: Mover Vizinho 0
        # 10: Mover Vizinho 1
        # 11: Mover Vizinho 2
        # 12: Mover Vizinho 3
        # 13: Usar Consumível
        ACTION_SHAPE = 14
                
        # DEFINIÇÃO DO ESPAÇO DE OBSERVAÇÃO (OBS_SHAPE = 198)
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
        #    44: Item/Evento (Genérico - Baú/Fonte)?
        #    45: Sala Vazia?
        #    46: Enemy HP Ratio
        #    47: Enemy Level Ratio
        #    48: Loot no chão? (Unificado: Wep/Arm/Art/Cons)
        #    49: Raridade Loot (Melhor item da pilha)
        #
        # 4. BLOCO SOCIAL/PvP (50-56) [7 Estados]
        #    50: In Arena?
        #    51: Other Agent Present?
        #    52: Other HP Ratio
        #    53: Level Diff
        #    54: Other Karma (Real)
        #    55: Other Karma (Imag)
        #    56: In PvP Combat?
        #
        # 5. BLOCO MOVIMENTO & AMBIENTE (57-121) [65 Estados]
        #    Inclui o Nó Atual + 4 Vizinhos (5 Blocos * 13 Features)
        #    Layout: [Current, Neighbor_1, Neighbor_2, Neighbor_3, Neighbor_4]
        #    Para cada um (13 Features):
        #       [0]: Valid Node?
        #       [1]: Enemy/Opponent Present?
        #       [2]: Reward (Item/Event/Exit)?
        #       [3]: Danger Tier (0.33=Common, 0.66=Elite, 1.0=Boss)
        #       [4-12]: Efeito de Sala (Encoder de 9 tipos)
        #
        # 6. BLOCO FINAL (122-129) [8 Estados]
        #    122: Self Weapon Rarity
        #    123: Self Armor Rarity
        #    124: Self Consumable Rarity
        #    125: Self Artifact Rarity
        #    126: Other Gear Score (Unificado / 4.0)
        #    127: Other Just Dropped?
        #    128: Other Skipped Attack?
        #    129: Door Open/Exit?
        #
        # 7. BLOCO DETALHADO DE ITENS (130-197) [68 Estados]
        #    4 Slots * 17 Features (Weapon, Armor, Artifact, Consumable)
        #    130-146: Weapon Embedding
        #    147-163: Armor Embedding
        #    164-180: Artifact Embedding   
        #    181-197: Consumable Embedding
        OBS_SHAPE = (198,)

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
        self.chests_opened_this_episode = {}
        self.consumables_used_this_episode = {}
        self.highest_damage_single_hit = {}
        self.healing_received_this_episode = {}
        self.wait_actions_this_episode = {}
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
            
            # Formata o conteúdo para log
            str_enemy  = ", ".join(content.get('enemies', [])) or "Nenhum"
            str_event  = ", ".join(content.get('events', [])) or "Nenhum"
            str_items  = ", ".join(content.get('items', [])) or "Vazio"
            str_effect = ", ".join(content.get('room_effects', [])) or "Normal"

            # Log da geração
            log_msg = (
                f"Sala '{node_name}' (Andar {next_floor}) >> "
                f"Inimigo: [{str_enemy}] | "
                f"Loot: [{str_items}] | "
                f"Evento: [{str_event}] | "
                f"Ambiente: [{str_effect}]"
            )

            self._log(agent_id, f"[PVE] {log_msg}")


    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Coleta e retorna a observação de 198 estados (Shape unificado)
        """        
        obs = np.full(self.observation_space[agent_id].shape, -1.0, dtype=np.float32) 
        
        if agent_id not in self.agent_states:
            return obs

        agent = self.agent_states[agent_id]
                
        # BLOCO DE SKILLS (0-39)
        skills_vector = []
        for i in range(4):
            skill_name = agent['skills'][i]
            skill_data = self.catalogs['skills'].get(skill_name, {})
            embedding = self.skill_encoder.encode(skill_data) 
            
            current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
            max_cd = skill_data.get('cd', 1)
            norm_current_cd = (current_cd / max_cd) if max_cd > 0 else 0.0
            
            full_skill_data = np.append(embedding, norm_current_cd)
            skills_vector.extend(full_skill_data)

        obs[0:40] = np.array(skills_vector, dtype=np.float32)        
        
        # Contexto e Grafos
        current_node_id = self.current_nodes.get(agent_id) 
        is_in_arena = agent_id in self.arena_instances # Use is_in_arena para consistência
        
        if is_in_arena:
            current_graph_to_use = self.arena_instances[agent_id]
        else:
            current_graph_to_use = self.graphs.get(agent_id)

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

        # Itens no Chão (Unificado)
        best_loot_rarity = 0.0
        has_any_loot = False
        
        for item_name in items_on_floor:
            details = self.catalogs['equipment'].get(item_name)
            if details: 
                has_any_loot = True
                rarity = self.rarity_map.get(details.get('rarity'), 0.0)
                if rarity > best_loot_rarity: 
                    best_loot_rarity = rarity
        
        if has_any_loot:
            obs[idx_pve + 5] = 1.0              
            obs[idx_pve + 6] = best_loot_rarity 

        # Bloco Social/PvP (50-56) [7 Slots]
        idx_soc = 50
        obs[idx_soc + 0] = 1.0 if is_in_arena else -1.0 
        
        if other_agent_id and is_in_arena and self.current_nodes.get(agent_id) == self.current_nodes.get(other_agent_id):
             other_agent_state = self.agent_states.get(other_agent_id)
             if other_agent_state:
                obs[idx_soc + 1] = 1.0 
                obs[idx_soc + 2] = (other_agent_state['hp'] / other_agent_state['max_hp']) * 2 - 1
                obs[idx_soc + 3] = (agent['level'] - other_agent_state['level']) / self.max_level
                
                k = self.reputation_system.get_karma_state(other_agent_id)
                obs[idx_soc + 4] = k.real
                obs[idx_soc + 5] = k.imag
        
        # Flag de PvP 
        if agent_id in self.pvp_sessions:
            obs[idx_soc + 6] = 1.0

        # Bloco Movimento (57-121) [65 Slots]        
        idx_mov = 57 
        
        curr_idx = idx_mov
        obs[curr_idx : curr_idx + 13] = 0.0 
        
        # Sala Atual
        obs[curr_idx + 0] = 1.0 # Válida
        
        has_opp_here = (other_agent_id and self.current_nodes.get(other_agent_id) == current_node_id)
        if room_content.get('enemies') or has_opp_here:
             obs[curr_idx + 1] = 1.0
        
        if room_content.get('items') or any(e in room_content.get('events', []) for e in ['Treasure', 'Morbid Treasure', 'Fountain of Life']):
             obs[curr_idx + 2] = 1.0
        if is_in_arena and current_graph_to_use.nodes[current_node_id].get('is_exit') and current_graph_to_use.graph.get('meet_occurred'):
             obs[curr_idx + 2] = 1.0
             
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
        
        c_effects = room_content.get('room_effects', [])
        eff_name = c_effects[0] if c_effects else 'None'
        obs[curr_idx + 4 : curr_idx + 13] = self.effect_encoder.encode(eff_name)

        # Vizinhos
        idx_neighbors_start = idx_mov + 13 

        for i in range(self.MAX_NEIGHBORS):            
            curr_idx = idx_neighbors_start + (i * 13)
            obs[curr_idx : curr_idx + 13] = 0.0

            if i < len(neighbors) and current_graph_to_use:
                neighbor_node_id = neighbors[i]
                
                if current_graph_to_use.has_node(neighbor_node_id):
                    n_content = current_graph_to_use.nodes[neighbor_node_id].get('content', {})
                    
                    obs[curr_idx + 0] = 1.0 
                    
                    has_opp = (other_agent_id and self.current_nodes.get(other_agent_id) == neighbor_node_id)
                    if n_content.get('enemies') or has_opp:
                        obs[curr_idx + 1] = 1.0
                        
                    if n_content.get('items') or any(e in n_content.get('events', []) for e in ['Treasure', 'Morbid Treasure', 'Fountain of Life']):
                         obs[curr_idx + 2] = 1.0
                    if is_in_arena and current_graph_to_use.nodes[neighbor_node_id].get('is_exit') and current_graph_to_use.graph.get('meet_occurred'):
                         obs[curr_idx + 2] = 1.0 
                         
                    d_level = 0.0
                    if n_content.get('enemies'):
                        ename = n_content['enemies'][0]
                        tier = self.catalogs['enemies'].get(ename, {}).get('tier', 'Common')
                        if tier == 'Boss': d_level = 1.0
                        elif tier == 'Elite': d_level = 0.66                        
                        else: d_level = 0.33
                    elif has_opp:
                        d_level = 0.66
                    obs[curr_idx + 3] = d_level

                    n_effects = n_content.get('room_effects', [])
                    eff_name = n_effects[0] if n_effects else 'None'
                    obs[curr_idx + 4 : curr_idx + 13] = self.effect_encoder.encode(eff_name)
        
        # BLOCO FINAL (122-130)        
        idx_end = 122
        
        # Self Gear
        w = agent['equipment'].get('Weapon')
        a = agent['equipment'].get('Armor')
        c = agent['equipment'].get('Consumable')
        art = agent['equipment'].get('Artifact')
        
        obs[idx_end + 0] = self.rarity_map.get(self.catalogs['equipment'][w].get('rarity'), 0.0) if w else 0.0
        obs[idx_end + 1] = self.rarity_map.get(self.catalogs['equipment'][a].get('rarity'), 0.0) if a else 0.0
        obs[idx_end + 2] = self.rarity_map.get(self.catalogs['equipment'][c].get('rarity'), 0.0) if c else 0.0
        obs[idx_end + 3] = self.rarity_map.get(self.catalogs['equipment'][art].get('rarity'), 0.0) if art else 0.0
        
        # Other Gear Score (Unificado)
        if other_agent_id:
            other_eq = self.agent_states[other_agent_id]['equipment']
            total_rarity = 0.0
            count = 0
            for slot in ['Weapon', 'Armor', 'Artifact', 'Consumable']:
                it = other_eq.get(slot)
                if it:
                    rst = self.catalogs['equipment'].get(it, {}).get('rarity')
                    total_rarity += self.rarity_map.get(rst, 0.0)
                    count += 1
            if count > 0:
                obs[idx_end + 4] = total_rarity / 4.0 
        
        # Flags Sociais
        if other_agent_id:
             if self.social_flags[other_agent_id].get('just_dropped'):
                 obs[idx_end + 5] = 1.0
             if self.social_flags[other_agent_id].get('skipped_attack'):
                 obs[idx_end + 6] = 1.0             
        
        # Porta
        if is_in_arena and current_graph_to_use.graph.get('meet_occurred'):
            obs[idx_end + 7] = 1.0 
        
        # BLOCO DETALHADO (130-198)
        w_data = self.catalogs['equipment'].get(w, {})
        a_data = self.catalogs['equipment'].get(a, {})
        art_data = self.catalogs['equipment'].get(art, {})
        cons_data = self.catalogs['equipment'].get(c, {}) 
        
        obs[130:147] = self.item_encoder.encode(w_data)
        obs[147:164] = self.item_encoder.encode(a_data)
        obs[164:181] = self.item_encoder.encode(art_data)   
        obs[181:198] = self.item_encoder.encode(cons_data)
                
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
            
            # Gera um novo nome para o agente
            agent_name = gerador_nomes.gerar_nome()
            
            # Chama a rotina de inicialização do agente
            self._initialize_agent_instance(agent_id, agent_name, karma_override=None)
            
            self._log(agent_id, f"[RESET] Novo episódio iniciado. {agent_name}.\n")                      
            
            # Coleta a observação inicial
            observations[agent_id] = self._get_observation(agent_id)
            infos[agent_id] = {}

        return observations, infos

    def _handle_combat_turn(self, agent_id: str, action: int, agent_info_dict: dict) -> tuple[float, bool]:
        """Orquestra um único turno de combate PvE e retorna (recompensa, combate_terminou)."""        
        
        # Pega o estado de combate PvE para o agente atual:
        combat_state = self.combat_states.get(agent_id) 

        # Se, por algum motivo, não houver estado de combate, encerra imediatamente.
        if not combat_state:            
            return -1.0, True, False # Penalidade pequena, combate terminou (bug)

        agent = combat_state['agent']
        enemy = combat_state['enemy']
        reward = 0
        combat_over = False
        terminated = False

        # Vez do agente agir
        hp_before_enemy = enemy['hp']
        hp_before_agent = agent['hp']
        action_name = "Wait" # Padrão

        # Mapeamento de Ações
        if 0 <= action <= 3: 
            action_name = self.agent_states[agent_id]['skills'][action]
        elif action == 13: 
            # AÇÃO 13: Usar Consumível             
            action_name = "Use Consumable"
            self.consumables_used_this_episode[agent_id] += 1
        else:
            # Se tentou usar outra ação inválida no combate
            self.invalid_action_counts[agent_id] += 1
            reward = REW_INVALID_ACTION 
            # Mantém action_name="Wait"
                
        # Executa ação do agente
        bonus_reward, success, msg, details = combat.execute_action(agent, [enemy], action_name, self.catalogs)

        # Se o HP subiu após a ação, foi cura.
        hp_gain = agent['hp'] - hp_before_agent
        if hp_gain > 0:
            self.healing_received_this_episode[agent_id] += hp_gain
        
        reward += bonus_reward
        
        # Log Público (Terminal + Buffer)
        if msg: 
            self._log(agent_id, f"[COMBATE] {msg}")
            
        # Log Privado/Detalhado (Hit rates, Dano exato)
        # Usa a flag global COMBAT_DETAILS para decidir se imprime no terminal ou só buffer
        if details:
            for d in details:
                self._log(agent_id, d, echo=COMBAT_DETAILS)  

        # Recompensa por dano causado (se foi ataque)
        damage_dealt = hp_before_enemy - enemy['hp']

        # Atualiza o maior dano em um único golpe
        if damage_dealt > self.highest_damage_single_hit[agent_id]:
            self.highest_damage_single_hit[agent_id] = damage_dealt
        reward += damage_dealt * REW_DMG_DEALT_SCALAR

        # Registra o dano causado na métrica do episódio
        if damage_dealt > 0:
            self.damage_dealt_this_episode[agent_id] += damage_dealt

        # Verifica se o inimigo morreu
        if combat.check_for_death_and_revive(enemy, self.catalogs):            
            reward += REW_KILL_ENEMY 
            self.enemies_defeated_this_episode[agent_id] += 1
            
            # Pega o estado *principal* do agente (fora do combate)
            agent_main_state = self.agent_states[agent_id] 
            
            agent_main_state['exp'] += enemy.get('exp_yield', 50) 
            
            # Log de Vitória
            self._log(
                agent_id, 
                f"[VITÓRIA!] {agent_main_state['name']} derrotou {enemy.get('name')} (LVL {enemy.get('level')})!"
                f"EXP: +{enemy.get('exp_yield', 50)}."
            )
            
            # Level Up Check
            leveled_up = progression.check_for_level_up(agent_main_state) 

            if leveled_up:
                self._log(agent_id, f"[LEVEL UP!] {agent_main_state['name']} subiu para o Nível {agent_main_state['level']}!")
                reward += REW_LEVEL_UP 
                agent_info_dict['level_up'] = True 

                # Quando o agente sobe de nível, atualiza os stats no combate                 
                agent['hp'] = agent_main_state['hp']
                agent['max_hp'] = agent_main_state['max_hp']
                agent['base_stats'] = agent_main_state['base_stats'].copy()
                
                # Reseta e Sincroniza Cooldowns
                for skill in agent_main_state.get('cooldowns', {}):
                    agent_main_state['cooldowns'][skill] = 0
                agent['cooldowns'] = agent_main_state['cooldowns'].copy()

            # Encerra Combate
            duration = self.current_step - self.combat_states[agent_id]['start_step']
            self.pve_combat_durations[agent_id].append(duration)     

            # Salva efeitos persistentes
            surviving_effects = {}
            for tag, data in agent['effects'].items():
                duration = data.get('duration', 0)
                if duration > 0: 
                    surviving_effects[tag] = data

            self.agent_states[agent_id]['persistent_effects'] = surviving_effects
            self.combat_states[agent_id] = None 
            combat_over = True
        
        # Se o combate continua, o inimigo age
        if not combat_over:
            hp_before_agent = agent['hp']
            
            # IA Simples do Inimigo
            available_skills = [s for s, cd in enemy['cooldowns'].items() if cd == 0]
            enemy_action = random.choice(available_skills) if available_skills else "Wait"
            
            # Executa Ação do Inimigo            
            _, _, enemy_msg, enemy_details = combat.execute_action(enemy, [agent], enemy_action, self.catalogs)
            
            # Log do inimigo:            
            if enemy_msg: 
                self._log(agent_id, f"[INIMIGO] {enemy_msg}")

            # Log detalhado do inimigo:            
            if enemy_details:
                for d in enemy_details:                    
                    self._log(agent_id, d, echo=COMBAT_DETAILS)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
            reward -= damage_taken * REW_DMG_TAKEN_SCALAR

            # Verifica se o agente morreu
            if combat.check_for_death_and_revive(agent, self.catalogs):
                combat_over = True 
                reward = REW_DEATH                                
                terminated = True # Agente morreu
                self.death_cause[agent_id] = f"PvE: {enemy['name']} (Lvl {enemy['level']})"                

        # Fim do turno: resolve efeitos e cooldowns
        if self.combat_states.get(agent_id): 
            combat.resolve_turn_effects_and_cooldowns(agent, self.catalogs)
            if not combat_over: 
                combat.resolve_turn_effects_and_cooldowns(enemy, self.catalogs)
        
        # Sincroniza com o estado mestre fora do combate
        master_state = self.agent_states[agent_id]
        
        # HP e Cooldowns
        master_state['hp'] = agent['hp'] 
        master_state['cooldowns'] = agent['cooldowns'].copy()
                
        # Se o item sumiu no combate, deve sumir no mestre
        if agent['equipment'].get('Consumable') is None:
            master_state['equipment']['Consumable'] = None
                    
        # Copia efeitos ganhos no combate para o mestre (para persistirem se o combate durar)
        master_state['effects'] = agent.get('effects', {}).copy()
        
        return reward, combat_over, terminated
    
    def _handle_pvp_combat_turn(self, combat_session: dict, action_a1: str, action_a2: str) -> tuple[float, float, bool, str, str]:
        """
        Executa turno PvP. Integra Ação 13 (Consumível) e Logging Detalhado.
        """
        
        # 1. Desempacota da sessão
        a1_combatant = combat_session['a1']
        a2_combatant = combat_session['a2']
        id_a1 = combat_session['a1_id'] 
        id_a2 = combat_session['a2_id'] 
        
        rew_a1 = 0
        rew_a2 = 0
        combat_over = False
        winner = None
        loser = None

        # --- TURNO AGENTE 1 ---
        hp_before_a2 = a2_combatant['hp']
        hp_before_a1 = a1_combatant['hp'] # Para checar cura própria
        
        # Executa (Skill ou Item)
        # Retorna 4 valores agora: reward_bonus, success, msg, details
        r1, _, log_message, details = combat.execute_action(a1_combatant, [a2_combatant], action_a1, self.catalogs)
        rew_a1 += r1

        # Checa a cura do A1
        heal_a1 = a1_combatant['hp'] - hp_before_a1
        if heal_a1 > 0: self.healing_received_this_episode[id_a1] += heal_a1

        # Atualiza maior dano em um único golpe
        dmg_a1 = hp_before_a2 - a2_combatant['hp']
        if dmg_a1 > self.highest_damage_single_hit[id_a1]:
            self.highest_damage_single_hit[id_a1] = dmg_a1
        
        # Logs
        if log_message: self._log(id_a1, f"[PVP] {log_message}")
        if details:
            for detail in details:
                self._log(id_a1, detail, echo=COMBAT_DETAILS) # Log técnico silencioso
        
        # Recompensas por Dano
        damage_dealt_by_a1 = hp_before_a2 - a2_combatant['hp']
        rew_a1 += damage_dealt_by_a1 * REW_DMG_DEALT_SCALAR
        rew_a2 -= damage_dealt_by_a1 * REW_DMG_TAKEN_SCALAR 
        
        # Checa Morte de A2
        if combat.check_for_death_and_revive(a2_combatant, self.catalogs):
            rew_a1 += REW_PVP_WIN 
            rew_a2 += REW_PVP_LOSS 
            combat_over = True
            winner = id_a1 
            loser = id_a2

            # Lógica de Vitória (XP, Level Up, Loot)
            winner_state = self.agent_states[id_a1]
            loser_state = self.agent_states[id_a2]
            
            xp_gain = 100 + (loser_state['level'] * 50)
            winner_state['exp'] += xp_gain
            self._log(id_a1, f"[PVP] Venceu! Ganhou {xp_gain} XP.")
            
            if progression.check_for_level_up(winner_state):
                self._log(id_a1, f"[LEVEL UP!] {winner_state['name']} subiu para o Nível {winner_state['level']}!")
                rew_a1 += REW_LEVEL_UP 
                # Atualiza combatente para refletir stats novos imediatamente
                a1_combatant['max_hp'] = winner_state['max_hp']
                a1_combatant['hp'] = winner_state['max_hp'] 
                a1_combatant['base_stats'] = winner_state['base_stats'].copy()
                           
            karma_adj = self._resolve_pvp_end_karma(winner, loser)
            rew_a1 += karma_adj
            
            self._drop_pvp_loot(loser_id=loser, winner_id=winner)

        # Turno do agente 2 (se combate não acabou)
        if not combat_over:
            hp_before_a1 = a1_combatant['hp']
            hp_before_a2 = a2_combatant['hp'] # Para checar cura
            
            r2, _, log_message, details = combat.execute_action(a2_combatant, [a1_combatant], action_a2, self.catalogs)
            rew_a2 += r2

            # Logando no ID correto (id_a2)
            if log_message: self._log(id_a2, f"[PVP] {log_message}")
            if details:
                for detail in details:
                    self._log(id_a2, detail, echo=COMBAT_DETAILS)

            # Checa cura do A2
            heal_a2 = a2_combatant['hp'] - hp_before_a2
            if heal_a2 > 0: self.healing_received_this_episode[id_a2] += heal_a2

            # Checa o dano crítico de A2
            dmg_a2 = hp_before_a1 - a1_combatant['hp']
            if dmg_a2 > self.highest_damage_single_hit[id_a2]:
                self.highest_damage_single_hit[id_a2] = dmg_a2

            damage_dealt_by_a2 = hp_before_a1 - a1_combatant['hp']
            rew_a2 += damage_dealt_by_a2 * REW_DMG_DEALT_SCALAR
            rew_a1 -= damage_dealt_by_a2 * REW_DMG_TAKEN_SCALAR
            
            # Checa Morte de A1
            if combat.check_for_death_and_revive(a1_combatant, self.catalogs):                
                rew_a2 += REW_PVP_WIN  
                rew_a1 += REW_PVP_LOSS 
                combat_over = True
                winner = id_a2 
                loser = id_a1  
                                
                self._resolve_pvp_end_karma(winner, loser)
                self._drop_pvp_loot(loser_id=loser, winner_id=winner)
                
        # Resolve Efeitos de Fim de Turno (Cooldowns, Dots)
        if not combat_over:
            combat.resolve_turn_effects_and_cooldowns(a1_combatant, self.catalogs)
            combat.resolve_turn_effects_and_cooldowns(a2_combatant, self.catalogs)
                
        # Salva HP, CD, Buffs e INVENTÁRIO (caso poção tenha sido gasta)
        for agent_id, combatant in [(id_a1, a1_combatant), (id_a2, a2_combatant)]:
            # Se perdeu, vai resetar no respawn, não precisa salvar.
            if loser != agent_id:
                master_state = self.agent_states[agent_id]
                
                # Sincroniza Vida e CD
                master_state['hp'] = combatant['hp']
                master_state['cooldowns'] = combatant['cooldowns'].copy()
                
                # Sincroniza Inventário (Evita Poção Infinita)
                if combatant['equipment'].get('Consumable') is None:
                    master_state['equipment']['Consumable'] = None
                
                # Sincroniza Buffs (ex: Força da Poção)
                master_state['effects'] = combatant.get('effects', {}).copy()
        
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
    
    def _log(self, agent_id: str, message: str, echo: bool = True):
        """
        Log colorido. 
        Se echo=False, salva apenas no buffer interno (para análise/debug) 
        e não imprime no terminal, mesmo que verbose > 0.
        """

        COLORS = {
            "reset": "\033[0m",
            "agent": "\033[38;5;51m",      
            "action": "\033[38;5;46m",     
            "pvp": "\033[38;5;196m",                   
            "karma_neg": "\033[38;5;196m", 
            "karma_neu": "\033[38;5;226m", 
            "warn": "\033[38;5;226m",      
            "error": "\033[38;5;196m",     
            "arena": "\033[38;5;201m",     
            "map": "\033[38;5;27m",        
            "upgrade": "\033[38;5;214m",   
            "detail": "\033[38;5;240m", # Cinza escuro para detalhes (se forçar print)
        }

        def colorize(text, color):
            return f"{COLORS[color]}{text}{COLORS['reset']}"

        # Print no Terminal (Apenas se verbose > 0 E echo for True)
        if self.verbose > 0 and echo:
            msg_upper = message.upper()

            # Mapeamento simples de padrões → cores
            patterns = {
                ("[ERRO]", "ERROR"): "error",
                ("EVENTO",): "upgrade",
                ("[WARN]",): "warn",
                ("PVP", "MORTE"): "pvp",
                ("AÇÃO", "AÇÃO-ARENA"): "action",
                ("SANCTUM", "ZONA K"): "pvp",
                ("PVE",): "map",
                ("DEBUG", "DETALHE"): "detail",
            }

            color = "agent"  # default

            # Caso especial: SOCIAL / UPGRADE / LEVEL UP / VITÓRIA
            if any(key in msg_upper for key in ("[SOCIAL]", "[UPGRADE]", "[LEVEL UP!]", "VITÓRIA!")):
                if "(-)" in message or "NEGAT" in msg_upper:
                    color = "karma_neg"
                else:
                    color = "karma_neu"
            else:
                # Procura nos padrões
                for keys, value in patterns.items():
                    if any(key in msg_upper for key in keys):
                        color = value
                        break

            formatted = f"{colorize(f'[{agent_id.upper()}]', 'agent')} {colorize(message, color)}"
            print(formatted)

        # Só salva no buffer se a flag estiver ativada (independente do echo)
        if getattr(self, 'enable_logging_buffer', True): 
            if agent_id not in self.current_episode_logs:
                self.current_episode_logs[agent_id] = []
                        
            # Limite de 1k linhas (Rotativo)
            if len(self.current_episode_logs[agent_id]) > DEATH_LOG_CUTOFF:
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
            
            # Popula a Arena
            nodes_list = sorted(list(new_arena.nodes()))
            center_node = nodes_list[len(nodes_list) // 2] # Pega o nó do meio

            for node in new_arena.nodes():
                content = content_generation.generate_room_content(
                    catalogs=self.catalogs,
                    budget_multiplier=1.0,
                    current_floor=base_floor,
                    guarantee_enemy=False
                )
                content['enemies'] = [] 
                content['items'] = []
                
                # Se for o nó do meio, força uma Fonte. Nos outros, apaga.
                if node == center_node:
                    content['events'] = ['Fountain of Life']
                else:
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
        
        agent_reward = REW_EXISTENCE_PVE # Penalidade de tempo
        terminated = False # 'terminated' significa morte ou vitória
        game_won = False
                
        # Verifica Vitória (Chegou ao fim do DAG)
        current_node = self.current_nodes[agent_id]
        current_graph = self.graphs[agent_id]
        is_on_last_floor = (self.current_floors[agent_id] == self.max_floors)
        has_no_successors = not list(current_graph.successors(current_node))

        if is_on_last_floor and has_no_successors and not self.combat_states.get(agent_id):
            terminated = True # Venceu!
            game_won = True 
            agent_reward += REW_VICTORY
            self._log(agent_id, f"[PVE] FIM: {self.agent_names[agent_id]} VENCEU! (Chegou ao fim do Jogo)")

        # Processa Ação Combate ou Exploração
        if self.combat_states.get(agent_id):
            # Agente está em combate PvE
            reward_combat, combat_over, terminated = self._handle_combat_turn(agent_id, action, infos[agent_id])
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
        
        # Verifica Morte do Agente
        if self.agent_states[agent_id]['hp'] <= 0:                        
            agent_reward = REW_DEATH # Penalidade grande por morrer 
                        
            terminated = True # Morte         

            # Captura o vetor de observação do estado de morte...
            terminal_obs = self._get_observation(agent_id)
            infos[agent_id]['terminal_observation'] = terminal_obs
            
            cause = "Dano Ambiental"
            if self.combat_states.get(agent_id):
                enemy_name = self.combat_states[agent_id]['enemy']['name']
                cause = f"{enemy_name} (Lvl {self.combat_states[agent_id]['enemy']['level']}) PVE"
            self.death_cause[agent_id] = cause

            # Faz a limpa nos logs ANTES de salvar o final_status            
            infos[agent_id]['final_status'] = self._generate_final_status(
                agent_id=agent_id,
                cause=cause, 
                win=False    
            )

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
            {"check": lambda f: f % MILESTONES_EACH == 0 and f > 0, "bonus": REW_MILESTONE_DECIMAL, "msg": "Marco Decimal!"}
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
        if not terminated and self.current_floors[agent_id] > 0 and \
           self.current_floors[agent_id] % self.sanctum_floor == 0: # Ex: Andares K, 2k, 3K...
            
            if agent_id not in self.agents_in_arena:
                 self._transition_to_arena(agent_id) 

        # 7. Cria 'final_status' se o episódio terminou para este agente
        if terminated or global_truncated:
            # Só gera se não tiver gerado antes (na morte)
            if 'final_status' not in infos[agent_id]:
                infos[agent_id]['final_status'] = self._generate_final_status(
                    agent_id=agent_id,
                    cause="Vitória" if game_won else "Tempo Esgotado",
                    win=game_won
                )                                                                 
                    
        return agent_reward, terminated

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
                elif idx_p1 == 13:
                    act_p1 = "Use Consumable"
                    self.consumables_used_this_episode[idx_p1] += 1
                else:
                    act_p1 = "Wait" # Ação inválida em combate PvP
                    self.wait_actions_this_episode[idx_p1] += 1

                if 0 <= idx_p2 <= 3:
                    act_p2 = state_p2['skills'][idx_p2]
                elif idx_p2 == 13:
                    act_p2 = "Use Consumable"
                    self.consumables_used_this_episode[idx_p2] += 1
                else:
                    act_p2 = "Wait" # Ação inválida em combate PvP
                    self.wait_actions_this_episode[idx_p2] += 1

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
                        pressure_penalty = -REW_SANCTUM_PRESSURE_BASE * (1 + (overtime * 0.1))
                        rewards[agent_id] += max(pressure_penalty, REW_SANCTUM_PRESSURE_CAP) # Teto de -10.0
                        
                        # Dano Físico (Aperto Real)
                        # A cada 5 turnos extras, perde 5% da vida MÁXIMA
                        if overtime % 5 == 0:
                            damage = self.agent_states[agent_id]['max_hp'] * 0.05
                            self.agent_states[agent_id]['hp'] -= damage
                            self._log(agent_id, f"[SANCTUM] Uma atmosfera opressiva drena sua vida (-{int(damage)} HP)...")
                            
                            # Verifica Morte por Pressão (DoT do Santuário)
                            if self.agent_states[agent_id]['hp'] <= 0:
                                self.death_cause[agent_id] = "Profanação do Santuário"
                                self._log(agent_id, "[MORTE] O agente sucumbiu à pressão do Santuário.")
                                
                                # Encerra a run do agente morto
                                rewards[agent_id] += REW_DEATH  # Garante a punição (-100 ou -300)
                                terminateds[agent_id] = True
                                
                                # Trata o oponente
                                opponent_id = self.active_matches.get(agent_id)
                                
                                if opponent_id:
                                    # Desvincula a partida (O oponente fica "solteiro" na sala dele)
                                    if opponent_id in self.active_matches:
                                        del self.active_matches[opponent_id]
                                    
                                    # Destranca a saída do oponente (Vitória por Sobrevivência)                                    
                                    if opponent_id in self.arena_instances:
                                        self.arena_instances[opponent_id].graph['meet_occurred'] = True
                                        self._log(opponent_id, "[SANCTUM] O oponente sucumbiu. A saída se destrancou.")
                                    
                                    # Limpa flags de interação do oponente (ele não está mais negociando com ninguém)
                                    self.arena_interaction_state[opponent_id]['offered_peace'] = False

                                # Limpa o Agente Morto
                                # Usa a função auxiliar para garantir limpeza completa
                                self._end_arena_encounter(agent_id) 
                                
                                # Reseta para o próximo episódio
                                self._respawn_agent(agent_id, self.death_cause[agent_id])
                                
                                # PULA O RESTO DO TURNO
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
                            rewards[agent_id] += REW_MEET_BONUS
                            rewards[opponent_id] += REW_MEET_BONUS                

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
                         rewards[agent_id] += REW_BETRAYAL
                         self.betrayals_this_episode[agent_id] += 1
                         self.arena_interaction_state[target_id]['offered_peace'] = False

                    # Lógica de Perfídia (Atacou segurando a própria oferta)
                    if self.arena_interaction_state[agent_id]['offered_peace']:
                         self._log(agent_id, f"[SOCIAL] PERFÍDIA! {self.agent_names[agent_id]} atacou enquanto oferecia paz! Karma (---).")
                         self.reputation_system.update_karma(agent_id, 'bad')
                         self.reputation_system.update_karma(agent_id, 'bad')
                         self.reputation_system.update_karma(agent_id, 'bad') # Tripla
                         rewards[agent_id] += REW_BETRAYAL
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

                elif 4 <= action <= 12:                
                    # 1. Executa a ação individual (Mecânica)
                    # O 'r' aqui traz a recompensa de movimento, ou de equipar item melhor.
                    # Se for Drop, r=0. Se for Saída Pacífica, r=REW_EXIT...
                    r, terminated = self._handle_exploration_turn(agent_id, action)
                    rewards[agent_id] += r
                    terminateds[agent_id] = terminated                                        
                    
                    # Atualiza flag de oferta (se dropou algo agora)
                    if self.social_flags[agent_id].get('just_dropped') and agent_id in self.active_matches:
                        self.arena_interaction_state[agent_id]['offered_peace'] = True                        
                    
                    # Atualiza flag de Ataque Pulado 
                    if opponent_id and self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                        self.social_flags[agent_id]['skipped_attack'] = True

                    # Detecta se alguém pegou o item de alguém                    
                    if opponent_id and self.current_nodes[agent_id] == self.current_nodes[opponent_id]:
                                                
                        # Agente atual pegou algo agora?
                        i_picked = self.social_flags[agent_id].get('just_picked_up')
                        
                        # O outro agente tinha ofertado algo antes (ou agora)?
                        he_offered = self.arena_interaction_state[opponent_id]['offered_peace']
                                            
                        if i_picked and he_offered:
                            
                            # Contexto importa:
                            # Eu também ofertei? (Troca Justa) ou só peguei? (Oportunismo)
                            i_offered = self.arena_interaction_state[agent_id]['offered_peace']
                            
                            if i_offered:
                                # Cenário 1: Comércio (ambos ofertaram)
                                # Ambos ofertaram. Ambos ganham.
                                self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} e {self.agent_names[opponent_id]} trocaram itens entre si! (++).")
                                
                                # Recompensas de Sucesso (Alta)
                                rewards[agent_id] += REW_TRADE_SUCESS
                                rewards[opponent_id] += REW_TRADE_SUCESS
                                
                                # Recompensa boa de reputação para ambos (todos amam um bom comerciante)
                                self.reputation_system.update_karma(agent_id, 'good')
                                self.reputation_system.update_karma(agent_id, 'good')
                                self.reputation_system.update_karma(opponent_id, 'good')
                                self.reputation_system.update_karma(opponent_id, 'good')
                                
                                # Stats
                                self.bargains_trade_this_episode[agent_id] += 1
                                self.bargains_trade_this_episode[opponent_id] += 1
                                
                                # Encerra o ciclo de ofertas (Negócio fechado)
                                self.arena_interaction_state[agent_id]['offered_peace'] = False
                                self.arena_interaction_state[opponent_id]['offered_peace'] = False
                                
                            else:
                                # Cenário 2: Tributo / Assalto (Unilateral)
                                # Ele ofertou, eu peguei. Eu não dei nada.
                                self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} aceitou o tributo de {self.agent_names[opponent_id]}.")
                                
                                # A vítima (ele): ganha boa reputação mas perde o item
                                # NOTA: Perder o item já é a "recompensa negativa" dele.
                                self.reputation_system.update_karma(opponent_id, 'good')  
                                self.bargains_toll_this_episode[agent_id] += 1
                                self.bargains_toll_this_episode[opponent_id] += 1                              
                                
                                # O OPORTUNISTA (Eu): Ganha o Item (Recompensa Material implícita).
                                # Não ganha Karma 'good' (não foi generoso).
                                # Não ganha Karma 'bad' (não violou regras, aceitou oferta).
                                # A recompensa dele é ter um item novo no inventário.
                                
                                # Reseta apenas a oferta dele (item consumido)
                                self.arena_interaction_state[opponent_id]['offered_peace'] = False
                                
                            # Nota: Não encerramos a arena aqui. 
                            # Eles podem continuar trocando ou sair pacificamente depois.
                
                # Inválido (ação > 12)
                else:
                    self.invalid_action_counts[agent_id] += 1
                    rewards[agent_id] += REW_INVALID_ACTION            
            # CASO 4: ESTÁ NO PVE?
            else:
                # Processa o passo PvE normal
                agent_reward, terminated = self._process_pve_step(
                    agent_id, action, global_truncated, infos
                )
                rewards[agent_id] = agent_reward
                terminateds[agent_id] = terminated

        # Finalização e coleta
        if all(terminateds.get(aid, False) for aid in self.agent_ids) or global_truncated:
            terminateds['__all__'] = True
            truncateds['__all__'] = True
            
            # Salva final_status se truncou
            if global_truncated:
             for agent_id in self.agent_ids:
                # Só gera se não tiver gerado antes (caso o agente já tenha morrido neste mesmo step)
                if 'final_status' not in infos[agent_id]:
                    
                    # Define causa padrão se não houver (ex: Tempo Esgotado)
                    current_cause = self.death_cause.get(agent_id, "Time Limit Reached")                
                    
                    infos[agent_id]['final_status'] = self._generate_final_status(
                        agent_id=agent_id,
                        cause=current_cause,
                        win=False 
                    )

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
        is_in_sanctum = (agent_id in self.arena_instances)
        reward = 0.0
        
        # Seleciona o grafo correto
        current_graph = self.arena_instances[agent_id] if is_in_sanctum else self.graphs.get(agent_id)

        # Lógica das ações de movimento (6, 7, 8, 9)
        if 9 <= action <= 12:
            neighbor_index = action - 9
                        
            if is_in_sanctum:
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
                step_cost = REW_EXISTENCE_PVP if is_in_sanctum else 0.0
                
                if prev_node and chosen_node == prev_node:
                     # Punição severa por voltar (Ping-Pong)
                     reward += REW_PING_PONG
                else:
                     # Recompensa por explorar nó novo
                     if not is_in_sanctum:
                         reward += REW_NEW_ROOM_PVE # PvE: Incentiva explorar
                     else:
                         reward += REW_NEW_ROOM_PVP # Arena: Incentiva explorar (necessário pra ativar encontros e saída)
                
                # Aplica a taxa base
                reward += step_cost 

                # Lógica de saída do Santuário
                if is_in_sanctum:
                    node_data = current_graph.nodes[chosen_node]

                    if node_data.get('is_exit', False):
                        # Verifica a trava global de encontro (Obrigatório se ver antes de sair)
                        meet_occurred = current_graph.graph.get('meet_occurred', False)
                        
                        if not meet_occurred:
                            self._log(agent_id, f"[AÇÃO-ARENA] A saída está TRANCADA. Encontre o outro agente primeiro!")
                            return REW_INVALID_ACTION, terminated 
                        
                        # Recompensa Base de Saída
                        # Não há mais bônus aqui. O bônus social já foi pago no step() se houve troca.
                        base_exit_reward = REW_EXIT_PEACE_BASE
                        
                        self._log(agent_id, f"[SANCTUM] {self.agent_names[agent_id]} deixou o santuário e seguiu em frente.")

                        # Encerra o encontro para mim
                        self._end_arena_encounter(agent_id)        
                        
                        # Atualiza o status de quem ficou (oponente)
                        opponent_id = self.active_matches.get(agent_id)
                        if opponent_id:
                            # Se o oponente ainda está na arena (pode ser que ele já tenha saído no turno dele)
                            if opponent_id in self.agents_in_arena:
                                
                                # Apenas remove o vínculo da partida
                                # Ele continua na sala, mas agora está "Solteiro" (Sem oponente)
                                if opponent_id in self.active_matches:
                                    del self.active_matches[opponent_id]
                                
                                # Reseta flags de interação para ele não ficar "falando sozinho"
                                self.arena_interaction_state[opponent_id]['offered_peace'] = False
                                
                                self._log(opponent_id, f"[SANCTUM] O oponente partiu. Está está sozinho no Santuário.")
                            
                        # Retorna (Eu vou para o PvE, ele fica lá decidindo a vida dele)
                        return base_exit_reward, terminated            

                self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} fez um movimento VÁLIDO para '{chosen_node}'.")
                
                # Lógica de Poda (Apenas na P-Zone)
                if not is_in_sanctum:
                    all_successors = list(current_graph.successors(current_node_id))
                    nodes_to_remove = [succ for succ in all_successors if succ != chosen_node]
                    for node in nodes_to_remove:
                        if current_graph.has_node(node):
                            descendants = nx.descendants(current_graph, node)
                            current_graph.remove_nodes_from(list(descendants) + [node])
                
                # Salva onde eu estava antes de mudar (para detectar ping-pong no próximo turno)
                self.previous_nodes[agent_id] = current_node_id

                # Efetua o Movimento
                self.current_nodes[agent_id] = chosen_node
                self.current_floors[agent_id] = current_graph.nodes[chosen_node].get('floor', self.current_floors[agent_id])

                new_floor = current_graph.nodes[chosen_node].get('floor', 0)
                if new_floor > self.max_floor_reached_this_episode[agent_id]:
                    self.max_floor_reached_this_episode[agent_id] = new_floor
                
                reward = REW_EXISTENCE_PVP if is_in_sanctum else REW_MOVE_PVE_SUCCESS

                # Gera novos sucessores e Combate PvE (Apenas P-Zone)
                if not is_in_sanctum:
                    if not list(current_graph.successors(chosen_node)):
                        self._generate_and_populate_successors(agent_id, chosen_node)

                    room_content = current_graph.nodes[chosen_node].get('content', {})

                    # Verifica Eventos Especiais
                    events_list = room_content.get('events', []) # Garante que é lista
                
                    # Verifica se está NA lista, não se É a lista
                    if 'Fountain of Life' in events_list:
                        agent_state = self.agent_states[agent_id]
                        
                        # Configuração da Cura (50% do HP Máximo)
                        heal_percent = 0.50 
                        heal_amount = int(agent_state['max_hp'] * heal_percent)
                        
                        # Aplica a cura (sem ultrapassar o Max HP)
                        old_hp = agent_state['hp']
                        agent_state['hp'] = min(agent_state['max_hp'], agent_state['hp'] + heal_amount)
                        actual_healed = agent_state['hp'] - old_hp
                        
                        # Registra cura nas estatísticas
                        if actual_healed > 0:
                            self.healing_received_this_episode[agent_id] += actual_healed

                        # Log visual
                        self._log(agent_id, f"[EVENTO] ⛲ Encontrou Fountain of Life! Recuperou {int(actual_healed)} HP ({int(old_hp)} -> {int(agent_state['hp'])}).")

                        # Remove para não curar infinitamente na mesma sala
                        room_content['events'].remove('Fountain of Life')
                                                                            
                    enemy_names = room_content.get('enemies', [])
                    if enemy_names:                        
                        self._log(agent_id, f"[AÇÃO] COMBATE INICIADO com {enemy_names[0]}")
                        self._start_combat(agent_id, enemy_names[0])
                        reward += REW_COMBAT_START
            else:
                # Movimento INVÁLIDO
                self._log(agent_id, f"[AÇÃO] Movimento INVÁLIDO (Sala {neighbor_index} não existe).")
                self.invalid_action_counts[agent_id] += 1
                reward = REW_INVALID_ACTION 

        #Ação 4: Equipar Item (Grimórios, Equipamentos e Artefatos)
        elif action == 4:
            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} tentou Ação 4 (Equipar/Aprender).")

            room_content = current_graph.nodes[current_node_id].get('content', {}) # Garante acesso ao content
            room_items = room_content.setdefault('items', [])
            room_events = room_content.get('events', [])
            
            # Decrementa o CD das skills do agentes
            self._tick_world_cooldowns(agent_id)
            
            # Segurança de nó
            if not current_graph.has_node(current_node_id) or 'content' not in current_graph.nodes[current_node_id]:
                 self.invalid_action_counts[agent_id] += 1
                 return REW_INVALID_ACTION, terminated

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
                        
                        self._log(agent_id, f"[UPGRADE] Aprendeu {new_skill_name} ({details.get('rarity')}) substituindo {old_skill}.")
                        
                        reward += REW_LEARN_SKILL # Grande incentivo para evoluir
                        item_consumed = True

                        if 'Treasure' in room_events:
                            self.chests_opened_this_episode[agent_id] += 1
                            room_events.remove('Treasure') # Remove para não contar 2x o mesmo baú
                            # self._log(agent_id, "[CHEST] Baú de Skill aberto!")
                        elif 'Morbid Treasure' in room_events:
                            self.chests_opened_this_episode[agent_id] += 1
                            room_events.remove('Morbid Treasure')
                            # self._log(agent_id, "[CHEST] Baú Mórbido aberto!")
                        break # Uma ação = Um livro lido
                        
            # Equipar armas ou armaduras (agora que o livro já foi verificado)
            if not item_consumed:
                best_item_to_equip = None
                max_rarity_diff = 0.0 
                best_item_type = None
                best_diff = 0.0 

                for item_name in room_items:
                    details = self.catalogs['equipment'].get(item_name)
                    
                    if details and details.get('type') in ['Weapon', 'Armor', 'Artifact', 'Consumable']:
                        item_type = details.get('type')
                        floor_rarity = self.rarity_map.get(details.get('rarity'), 0.0)
                        
                        # Compara com o que já tem
                        equipped_name = self.agent_states[agent_id]['equipment'].get(item_type)
                        equipped_rarity = 0.0
                        if equipped_name:
                            r_str = self.catalogs['equipment'].get(equipped_name, {}).get('rarity')
                            equipped_rarity = self.rarity_map.get(r_str, 0.0)
                        
                        diff = floor_rarity - equipped_rarity                                                
                        
                        # Critério de Troca
                        if diff > max_rarity_diff or max_rarity_diff <= 0.0:
                            max_rarity_diff = diff
                            best_item_to_equip = item_name
                            best_item_type = item_type
                            best_diff = diff

                if best_item_to_equip:
                    # Realiza a troca 
                    self.agent_states[agent_id]['equipment'][best_item_type] = best_item_to_equip
                    room_items.remove(best_item_to_equip)
                    
                    # Salva métricas 
                    self.equipment_swaps_this_episode[agent_id] += 1
                    finesse = self.catalogs['equipment'][best_item_to_equip]['rarity']
                    
                    if best_diff > 0:
                        self._log(agent_id, f"[UPGRADE] Equipou {best_item_to_equip} ({finesse}).")
                    else:
                        self._log(agent_id, f"[AÇÃO] Equipou {best_item_to_equip} ({finesse}).")


                    # Se equipou algo de um baú, conta o baú
                    if 'Treasure' in room_events:
                        self.chests_opened_this_episode[agent_id] += 1
                        room_events.remove('Treasure') # Consome o evento
                        # self._log(agent_id, "[CHEST] Baú de Equipamento aberto!")
                    elif 'Morbid Treasure' in room_events:
                        self.chests_opened_this_episode[agent_id] += 1
                        room_events.remove('Morbid Treasure')
                        # self._log(agent_id, "[CHEST] Baú Mórbido aberto!")

                    # Calcula a recompensa baseada na melhoria
                    reward = REW_EQUIP_BASE + (max_rarity_diff * REW_EQUIP_RARITY_MULT)

                    # Se estiver no Santuário, registra sinais sociais
                    if is_in_sanctum:
                        # Verifica se é reciclagem (item que eu mesmo dropei)
                        is_recycled_loot = (best_item_to_equip in self.sanctum_dropped_history[agent_id])

                        if is_recycled_loot:
                            # Se pegou o próprio item de volta (evita exploit de recompensa)                            
                            reward = REW_INVALID_ACTION 
                            
                            # Cancela os sinais sociais
                            self.social_flags[agent_id]['just_picked_up'] = False 
                            self.arena_interaction_state[agent_id]['offered_peace'] = False 
                            
                            self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} pegou de volta seu próprio {best_item_to_equip}.")
                        else:
                            # É um item novo ou do oponente -> Valida o sinal
                            self.social_flags[agent_id]['just_picked_up'] = True
                
                else:
                    # Falhou: Não tinha nada útil ou sala vazia
                    if room_items:
                        reward = REW_EMPTY_LOOT 
                    else:
                        self.invalid_action_counts[agent_id] += 1                    
                        reward = REW_INVALID_ACTION

        # Ação 5: Dropar Artefato      
        elif 5 <= action <= 8:
            if not is_in_sanctum:
                # Drop proibido fora do santuário (ou pune severamente)
                self.invalid_action_counts[agent_id] += 1
                return REW_INVALID_ACTION, terminated

            # Mapeamento do Slot
            slot_map = {
                5: 'Weapon',      # Alto Risco
                6: 'Armor',       # Alto Risco
                7: 'Artifact',    # Médio Risco (Era o antigo 5)
                8: 'Consumable'   # Baixo Risco (Novo!)
            }
            slot_name = slot_map[action]
            
            # Tenta pegar o item do inventário
            equipped_item = self.agent_states[agent_id]['equipment'].get(slot_name)
            
            if equipped_item:
                # Verifica se o agente já tentou usar este item específico como barganha antes
                if equipped_item in self.sanctum_dropped_history[agent_id]:
                    # "Eu já joguei essa carta na mesa."
                    # O agente tenta dropar, mas o ambiente diz "Não, você já negociou isso".
                    self.invalid_action_counts[agent_id] += 1
                    return REW_INVALID_ACTION, terminated

                # Remove do Agente
                self.agent_states[agent_id]['equipment'][slot_name] = None
                
                # Adiciona no Chão
                room_items = current_graph.nodes[current_node_id]['content'].setdefault('items', [])
                room_items.append(equipped_item)
                
                # Registra Flags Sociais
                self.sanctum_dropped_history[agent_id].add(equipped_item)
                self.social_flags[agent_id]['just_dropped'] = True                                
                
                self._log(agent_id, f"[SOCIAL] {self.agent_names[agent_id]} dropou {slot_name}: {equipped_item} ({self.catalogs['equipment'][equipped_item]['rarity']}) no Santuário (Sala '{current_node_id}').")                           
            
            else:
                # Tentou dropar slot vazio
                self.invalid_action_counts[agent_id] += 1
                reward = REW_INVALID_ACTION

        # Para ações inválidas
        else:
            reward = REW_INVALID_ACTION
            
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
            f"[COMBATE] {self.agent_names[agent_id]} (Nível {agent_main_state['level']}, HP {int(agent_combatant['hp'])}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {int(enemy_combatant['hp'])}) "
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
            penalty = REW_SMURF_PENALTY_BASE * scale
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
            bonus = REW_HEROIC_BONUS_BASE * scale
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
        Processa a "morte" de um agente, limpa vestígios na Arena e recria o corpo.
        """
        self._log(agent_id, f"[MORTE] {self.agent_names[agent_id]} foi derrotado por {cause}! Resetando...")
        
        # Salva o Karma atual antes de resetar
        current_name = self.agent_names[agent_id]
        preserved_karma_z = self.reputation_system.get_karma_state(agent_id)

        # Limpezas necessárias antes do reset
        # Limpa sessão PvP
        if agent_id in self.pvp_sessions:
             session = self.pvp_sessions[agent_id]
             p1, p2 = session['a1_id'], session['a2_id']
             self.pvp_sessions.pop(p1, None)
             self.pvp_sessions.pop(p2, None)

        # Limpa pareamento (Active Match)
        if agent_id in self.active_matches:
            opponent_id = self.active_matches[agent_id]
            if opponent_id in self.active_matches:
                del self.active_matches[opponent_id]                
            del self.active_matches[agent_id]

        # Limpa instância de arena e filas
        if agent_id in self.arena_instances: del self.arena_instances[agent_id] 
        if agent_id in self.matchmaking_queue: self.matchmaking_queue.remove(agent_id)
        if agent_id in self.agents_in_arena: self.agents_in_arena.remove(agent_id)

        # Reinicializa o agente
        # Recria o agente do zero, mas passando o Karma antigo para ser mantido
        self._initialize_agent_instance(agent_id, current_name, karma_override=preserved_karma_z)

        self._log(agent_id, f"[RESPAWN] {current_name} (Nível 1) voltou a estaca zero.")

    def _initialize_agent_instance(self, agent_id: str, agent_name: str, karma_override: complex = None):
        """
        Método unificado para inicializar ou resetar o estado físico, métricas e mapa de um agente.
        Usado tanto pelo reset() quanto pelo _respawn_agent().
        """
        self.agent_names[agent_id] = agent_name
        
        # Cria estado físico (Nível 1, Stats Base)
        self.agent_states[agent_id] = progression.create_initial_agent(agent_name)  
        self.agent_states[agent_id]['effects'] = {}
        self.agent_states[agent_id]['persistent_effects'] = {}      

        # Inicializa/Reseta TODAS as métricas
        self.current_episode_logs[agent_id] = []
        self.enemies_defeated_this_episode[agent_id] = 0
        self.invalid_action_counts[agent_id] = 0
        self.last_milestone_floors[agent_id] = 0
        self.combat_states[agent_id] = None
        self.damage_dealt_this_episode[agent_id] = 0.0
        self.equipment_swaps_this_episode[agent_id] = 0
        self.skill_upgrades_this_episode[agent_id] = 0
        self.death_cause[agent_id] = "Sobreviveu (Time Limit)" # Default, muda se morrer

        self.arena_encounters_this_episode[agent_id] = 0
        self.pvp_combats_this_episode[agent_id] = 0
        self.bargains_succeeded_this_episode[agent_id] = 0
        self.bargains_trade_this_episode[agent_id] = 0
        self.bargains_toll_this_episode[agent_id] = 0
        self.cowardice_kills_this_episode[agent_id] = 0
        self.betrayals_this_episode[agent_id] = 0
        self.karma_history[agent_id] = []
        self.sanctum_dropped_history[agent_id] = set()
        
        # Novas métricas
        self.chests_opened_this_episode[agent_id] = 0
        self.consumables_used_this_episode[agent_id] = 0
        self.highest_damage_single_hit[agent_id] = 0
        self.healing_received_this_episode[agent_id] = 0
        self.wait_actions_this_episode[agent_id] = 0

        self.max_floor_reached_this_episode[agent_id] = 0
        
        self.pve_combat_durations[agent_id] = [] 
        self.pvp_combat_durations[agent_id] = []
        
        # Limpezas auxiliares de navegação
        self.arena_entry_steps[agent_id] = 0
        self.previous_nodes[agent_id] = []
        self.pve_return_floor[agent_id] = []

        # Configura o Karma (Novo ou Preservado)
        if karma_override is not None:
            # Respawn: Usa o karma que o agente tinha antes de morrer
            karma_z = karma_override
            # Atualiza o estado do agente para refletir o sistema
            self.agent_states[agent_id]['karma']['real'] = karma_z.real
            self.agent_states[agent_id]['karma']['imag'] = karma_z.imag
        else:
            # Reset: Usa o karma padrão da criação do personagem
            karma_z = complex(
                self.agent_states[agent_id]['karma']['real'],
                self.agent_states[agent_id]['karma']['imag']
            )
        
        # Adiciona ao sistema de reputação (Isso reseta a posição dele no sistema se já existisse)
        self.reputation_system.add_agent(agent_id, karma_z)

        # Cria o Mapa Inicial (Floor 0)
        self.current_floors[agent_id] = 0
        self.current_nodes[agent_id] = "start"
        self.nodes_per_floor_counters[agent_id] = {0: 1}
        self.graphs[agent_id] = nx.DiGraph()
        self.graphs[agent_id].add_node("start", floor=0)
        
        # Gera conteúdo da sala inicial
        start_content = content_generation.generate_room_content(
            self.catalogs, 
            budget_multiplier=0.0,
            current_floor=0,
            guarantee_enemy=False
        )
        self.graphs[agent_id].nodes["start"]['content'] = start_content        

        # Gera sucessores
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

    def _generate_final_status(self, agent_id: str, cause: str, win: bool = False) -> dict:
        """
        Gera o dicionário padronizado de status final/analítico do agente.
        Usado tanto na morte (pve/trap) quanto no fim do episódio (truncation/win).
        """
        state = self.agent_states[agent_id]
        
        return {
            # Dados básicos
            'agent_name': self.agent_names.get(agent_id, "Unknown"),
            'level': state['level'],
            'hp': state['hp'], # Se morreu, virá 0 ou negativo. Se acabou o tempo, virá o atual.
            'floor': self.max_floor_reached_this_episode.get(agent_id, self.current_floors[agent_id]),
            'win': win,
            'steps': self.current_step,
            'enemies_defeated': self.enemies_defeated_this_episode.get(agent_id, 0),
            'exp': state['exp'],
            'death_cause': cause,

            # Combate e estatísticas de ação
            'damage_dealt': self.damage_dealt_this_episode.get(agent_id, 0.0),
            'invalid_actions': self.invalid_action_counts.get(agent_id, 0),
            'wait_actions': self.wait_actions_this_episode.get(agent_id, 0),         # [NOVO]
            'total_healing': self.healing_received_this_episode.get(agent_id, 0),    # [NOVO]
            'highest_crit': self.highest_damage_single_hit.get(agent_id, 0),         # [NOVO]

            # Economia & Inventário
            'equipment': state.get('equipment', {}).copy(),
            'equipment_swaps': self.equipment_swaps_this_episode.get(agent_id, 0),
            'skill_upgrades': self.skill_upgrades_this_episode.get(agent_id, 0),
            'skills': state.get('skills', []).copy(),
            'consumables_used': self.consumables_used_this_episode.get(agent_id, 0), # [NOVO]
            'chests_opened': self.chests_opened_this_episode.get(agent_id, 0),       # [NOVO]

            # Listas de duração
            'pve_durations': self.pve_combat_durations.get(agent_id, []).copy(),
            'pvp_durations': self.pvp_combat_durations.get(agent_id, []).copy(),
            'arena_encounters': self.arena_encounters_this_episode.get(agent_id, 0),
            'pvp_combats': self.pvp_combats_this_episode.get(agent_id, 0),
            
            # Social 
            'bargains_succeeded': self.bargains_succeeded_this_episode.get(agent_id, 0),
            'bargains_trade': self.bargains_trade_this_episode.get(agent_id, 0),
            'bargains_toll': self.bargains_toll_this_episode.get(agent_id, 0),
            'cowardice_kills': self.cowardice_kills_this_episode.get(agent_id, 0),
            'betrayals': self.betrayals_this_episode.get(agent_id, 0),
            'karma_history': self.karma_history.get(agent_id, []).copy(),
            'karma': state['karma'].copy(),

            # Log completo da run (com corte definido por DEATH_LOG_CUTOFF)
            'full_log': self.current_episode_logs.get(agent_id, []).copy()
        }