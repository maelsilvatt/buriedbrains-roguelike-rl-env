# buriedbrains/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
import networkx as nx

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
                 max_level: int = 400,
                 budget_multiplier: float = 1.0,
                 guarantee_enemy: bool = False, 
                 verbose: int = 0,
                 sanctum_floor: int = 20):
        super().__init__()        
        
        # --- 1. CARREGAMENTO DE CATÁLOGOS (Sem Mudanças) ---
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

        # --- Extrai catálogos filtrados para lógica específica ---
        self.artifact_catalog = {
            k: v for k, v in self.catalogs.get('equipment', {}).items() 
            if v.get('type') == 'Artifact'
        }
        self.equipment_catalog_no_artifacts = {
            k: v for k, v in self.catalogs.get('equipment', {}).items() 
            if v.get('type') in ['Weapon', 'Armor']
        }

        # Adiciona o catálogo filtrado ao self.catalogs principal    
        self.catalogs['artifacts'] = self.artifact_catalog

        # --- 2. PARÂMETROS GLOBAIS DO AMBIENTE ---
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.verbose = verbose 
        self.max_floors = max_floors
        self.max_level = max_level         
        self.budget_multiplier = budget_multiplier
        self.guarantee_enemy = guarantee_enemy
        self.pool_costs = content_generation._calculate_costs(enemy_data['pools'])
        self.rarity_map = {'Common': 0.25, 'Rare': 0.5, 'Epic': 0.75, 'Legendary': 1.0}
        self.sanctum_floor = sanctum_floor  # Andar em que ocorre a transição para a arena
        
        for catalog_name, catalog_data in self.catalogs.items():
            if isinstance(catalog_data, dict):
                for name, data in catalog_data.items():
                    if isinstance(data, dict):
                        data['name'] = name

        # --- 3. DEFINIÇÕES MULTIAGENTE (MAE) ---
        self.agent_ids = ["a1", "a2"]
        # Mapeamento fixo de skills base. O Agente sempre terá estas.
        self.agent_skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]
        # Máximo de vizinhos que uma sala pode ter
        self.MAX_NEIGHBORS = 4 

        self.pvp_state = None # Armazena o estado de combate PvP (ex: {'a1_combatant': ..., 'a2_combatant': ...})
        self.arena_interaction_state = {
            'a1': {'offered_peace': False},
            'a2': {'offered_peace': False}
        }#

        # --- 4. ESPAÇOS DE AÇÃO E OBSERVAÇÃO (Refatorados para MAE) ---
        
        # Ações: 0-3 (Skills), 4 (Equipar), 5 (Social), 6-9 (Mover)
        # 0: Quick Strike, 1: Heavy Blow, 2: Stone Shield, 3: Wait
        # 4: Equipar Item
        # 5: Soltar/Pegar Artefato (Social)
        # 6: Mover Vizinho 0
        # 7: Mover Vizinho 1
        # 8: Mover Vizinho 2
        # 9: Mover Vizinho 3
        ACTION_SHAPE = 10 
                
        # 14 PvE + 14 Social + 12 Vizinhos + 2 Self-Equip = 42
        OBS_SHAPE = (42,)
        
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

        # --- 5. VARIÁVEIS DE ESTADO ---
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
        self.cowardice_kills_this_episode = {}
        self.pve_combat_durations = {}
        self.pvp_combat_durations = {}
        self.karma_history = {}


        # --- 6. ESTADO GLOBAL DO AMBIENTE (Novo para MAE) ---
        self.env_state = 'PROGRESSION'
        self.arena_graph = None
        self.agents_in_arena = set()

        self.arena_meet_occurred = False # Trava a saída até o encontro

        # --- 7. SISTEMA DE REPUTAÇÃO ---
        # Define os parâmetros para o potencial
        potential_params = {
            'z_saint': 0.95 + 0j,  # Polo "santo" 
            'z_villain': -0.95 + 0j, # Polo "vilão" 
            'attraction': 0.5 # Força da atração (precisa ser ajustada)
        }
        # Instancia o motor de Karma
        self.reputation_system = reputation.HyperbolicReputationSystem(
            potential_func=reputation.saint_villain_potential,
            potential_params=potential_params,
            dt=0.1, # "Passo" da atualização
            noise_scale=0.01 # Ruído
        )

    def _generate_and_populate_successors(self, agent_id: str, parent_node: str):
        """
        Orquestra a geração de nós sucessores (para P-Zone):
        1. Chama map_generation para criar a topologia (nós e arestas).
        2. Popula os novos nós com conteúdo (inimigos, eventos).
        3. Loga os resultados.
        """
        
        # 1. Chamar o módulo externo para criar a topologia
        graph = self.graphs[agent_id]
        counters = self.nodes_per_floor_counters[agent_id]
        
        # A função map_generation modifica 'graph' e 'counters'
        new_node_names, updated_counters = map_generation.generate_progression_successors(
            graph, 
            parent_node, 
            counters
        )
        
        # Atualiza o contador do ambiente com a versão retornada
        self.nodes_per_floor_counters[agent_id] = updated_counters
        
        # 2. Iterar e popular cada nó recém-criado
        for node_name in new_node_names:
            next_floor = graph.nodes[node_name].get('floor', 0)
            
            # Calcula o budget
            budget = ( 100 + (next_floor * 10) ) * self.budget_multiplier
            
            # 3. Chamar o módulo de conteúdo
            content = content_generation.generate_room_content(
                self.catalogs,                 
                budget, 
                next_floor,
                guarantee_enemy=self.guarantee_enemy 
            )
            self.graphs[agent_id].nodes[node_name]['content'] = content
            
            # 4. Logar os resultados
            enemy_log = content.get('enemies', []) or ['Nenhum']
            event_outcome_log = content.get('events', []) or ['Nenhum']
            item_generated_log = content.get('items', []) or ['Nenhum']
            effect_log = content.get('room_effects', []) or ['Nenhum']

            self._log(agent_id, f"[MAPA] Sala '{node_name}' (Andar {next_floor}) gerada com: "
                        f"Inimigo: {enemy_log[0]}, "
                        f"Evento: {event_outcome_log[0]}, "
                        f"Item Gerado: {item_generated_log[0]}, "
                        f"Efeito: {effect_log[0]}")

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Coleta e retorna a observação de 41 estados para o agente especificado.        
        """
        # Começa com -1.0 (default)
        obs = np.full(self.observation_space[agent_id].shape, -1.0, dtype=np.float32) 
        
        if agent_id not in self.agent_states:
            return obs

        agent = self.agent_states[agent_id]
        current_node_id = self.current_nodes[agent_id]
                
        # Usa o grafo de progressão individual
        current_graph = self.graphs[agent_id]
        
        # Só muda para o grafo da arena SE estiver na arena e o grafo existir
        if self.env_state != 'PROGRESSION' and \
           agent_id in self.agents_in_arena and \
           self.arena_graph is not None:
            current_graph = self.arena_graph        

        # se por algum milagre ainda for None, retorna obs vazia para não crashar
        if current_graph is None:
             return obs

        if not current_graph.has_node(current_node_id):
             return obs
             
        room_content = current_graph.nodes[current_node_id].get('content', {})
        pve_combat_state = self.combat_states.get(agent_id)
        
        # --- Bloco Próprio (0-6) ---
        obs[0] = (agent['hp'] / agent['max_hp']) * 2 - 1 if agent['max_hp'] > 0 else -1.0
        obs[1] = (agent['level'] / self.max_level) * 2 - 1
        obs[2] = (agent['exp'] / agent['exp_to_level_up']) if agent['exp_to_level_up'] > 0 else 0.0
        for i in range(4):
            skill_name = self.agent_skill_names[i]
            max_cd = self.catalogs['skills'].get(skill_name, {}).get('cd', 1)
            current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
            obs[3 + i] = (current_cd / max_cd) if max_cd > 0 else 0.0

        # --- Bloco Contexto PvE (7-13) ---
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

        # --- Bloco Contexto Social/PvP (14-19 + 38-40) ---
        obs[14] = 1.0 if self.env_state != 'PROGRESSION' else -1.0
        
        other_agent_id = next((aid for aid in self.agent_ids if aid != agent_id), None)
        
        if other_agent_id and (self.env_state != 'PROGRESSION'):
            is_in_same_room = (self.current_nodes[agent_id] == self.current_nodes[other_agent_id])

            if is_in_same_room:
                other_agent_state = self.agent_states.get(other_agent_id)
                if other_agent_state:
                    obs[15] = 1.0 # Flag: Outro Agente na sala
                    obs[16] = (other_agent_state['hp'] / other_agent_state['max_hp']) * 2 - 1
                    obs[17] = (agent['level'] - other_agent_state['level']) / 100.0
                    
                    opponent_karma_z = self.reputation_system.get_karma_state(other_agent_id)
                    obs[18] = opponent_karma_z.real 
                    obs[19] = opponent_karma_z.imag

                    # Gear Score do Oponente (obs 38)
                    opp_equip = other_agent_state.get('equipment', {})
                    total_rarity = 0.0
                    for item in opp_equip.values():
                        if item:
                            r_str = self.catalogs['equipment'].get(item, {}).get('rarity')
                            total_rarity += self.rarity_map.get(r_str, 0.0)
                    
                    # Normaliza (Média de raridade: 0.0 a 1.0)
                    obs[38] = total_rarity / 3.0 

                    # Intenções Sociais do Oponente (obs 39, 40)
                    # Lê as flags do OUTRO agente
                    opp_flags = self.social_flags.get(other_agent_id, {})
                    
                    # Obs 39: Oponente dropou item neste turno?
                    obs[39] = 1.0 if opp_flags.get('just_dropped', False) else -1.0
                    
                    # Obs 40: Oponente podia atacar e não atacou?
                    obs[40] = 1.0 if opp_flags.get('skipped_attack', False) else -1.0                

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
        if self.pvp_state and agent_id in self.pvp_state:
            obs[23] = 1.0

        # --- Bloco Contexto Movimento (24-35) ---
        if self.env_state == 'PROGRESSION':
            neighbors = list(current_graph.successors(current_node_id))
        else:
            # Na arena, se o grafo existir
            if current_graph:
                try:
                    neighbors = list(current_graph.neighbors(current_node_id))
                except:
                    neighbors = []
            else:
                neighbors = []
                
        neighbors.sort() 

        for i in range(self.MAX_NEIGHBORS):
            if i < len(neighbors):
                neighbor_node_id = neighbors[i]
                if not current_graph.has_node(neighbor_node_id): continue 
                neighbor_content = current_graph.nodes[neighbor_node_id].get('content', {})
                
                obs[24 + i*3 + 0] = 1.0 # Válido
                if neighbor_content.get('enemies') or (other_agent_id and self.current_nodes.get(other_agent_id) == neighbor_node_id):
                    obs[24 + i*3 + 1] = 1.0 # Inimigo
                if neighbor_content.get('items') or any(evt in neighbor_content.get('events', []) for evt in ['Treasure', 'Morbid Treasure', 'Fountain of Life']):
                    obs[24 + i*3 + 2] = 1.0 # Recompensa
        
        # --- Equipamento Atual (36-37) ---
        equipped_weapon = agent['equipment'].get('Weapon')
        obs[36] = self.rarity_map.get(self.catalogs['equipment'][equipped_weapon].get('rarity'), 0.0) if equipped_weapon else 0.0
            
        equipped_armor = agent['equipment'].get('Armor')
        obs[37] = self.rarity_map.get(self.catalogs['equipment'][equipped_armor].get('rarity'), 0.0) if equipped_armor else 0.0

        # O agente "ouve" ou "sente" que a condição de saída foi satisfeita        
        obs[41] = 1.0 if self.arena_meet_occurred else -1.0

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para um novo episódio multiagente.
        Inicializa o estado, o grafo e as métricas para todos os agentes ('a1', 'a2').
        """
        super().reset(seed=seed) # Semeia self.np_random
        if seed is not None:
            # Garante que o 'random' global também seja semeado para consistência
            random.seed(seed) 

        # --- 1. RESETAR ESTADOS GLOBAIS E DE AGENTE ---
        # Limpa todos os dicionários de estado da sessão anterior
        self.agent_states = {}
        self.agent_names = {}
        self.graphs = {}
        self.current_nodes = {}
        self.current_floors = {}
        self.nodes_per_floor_counters = {} # Antiga 'nodes_per_floor'
        self.combat_states = {}
        
        # Métricas
        self.current_episode_logs = {}
        self.enemies_defeated_this_episode = {}
        self.invalid_action_counts = {}
        self.last_milestone_floors = {}

        # Estado Global
        self.current_step = 0
        self.env_state = 'PROGRESSION' # Começa no modo de progressão separado
        self.arena_graph = None
        self.agents_in_arena = set()
                
        # Limpa os agentes do sistema de reputação da run anterior
        self.reputation_system.agent_karma = {}
        # Inicializa as flags sociais de 1 turno
        self.social_flags = {agent_id: {} for agent_id in self.agent_ids}     
        self.arena_interaction_state = {
            'a1': {'offered_peace': False},
            'a2': {'offered_peace': False}
        }   

        # Dicionários de retorno para a API MAE
        observations = {}
        infos = {}

        # Gerador de nomes único para este episódio
        gerador_nomes = GeradorNomes()

        # --- 2. LOOP DE INICIALIZAÇÃO POR AGENTE ---
        # Itera sobre ['a1', 'a2'] e cria um mundo PvE para cada um
        for agent_id in self.agent_ids:
            
            # --- Criação do Agente ---
            agent_name = gerador_nomes.gerar_nome()
            self.agent_names[agent_id] = agent_name
            # create_initial_agent agora define o karma e o artefato inicial
            self.agent_states[agent_id] = agent_rules.create_initial_agent(agent_name) 
            # As skills base são compartilhadas e definidas no __init__
            self.agent_states[agent_id]['skills'] = self.agent_skill_names 
            
            # --- Métricas e Logs ---
            self.current_episode_logs[agent_id] = []
            self.enemies_defeated_this_episode[agent_id] = 0
            self.invalid_action_counts[agent_id] = 0
            self.last_milestone_floors[agent_id] = 0
            self.combat_states[agent_id] = None
            self.damage_dealt_this_episode[agent_id] = 0.0
            self.equipment_swaps_this_episode[agent_id] = 0
            self.death_cause[agent_id] = "Sobreviveu (Time Limit)" # Padrão

            # Sociais / PvP
            self.arena_encounters_this_episode[agent_id] = 0
            self.pvp_combats_this_episode[agent_id] = 0
            self.bargains_succeeded_this_episode[agent_id] = 0
            self.cowardice_kills_this_episode[agent_id] = 0
            self.karma_history[agent_id] = []
            
            # Duração de Combate (Listas para guardar a duração de CADA luta)
            self.pve_combat_durations[agent_id] = [] 
            self.pvp_combat_durations[agent_id] = []

            # --- ADIÇÃO AO SISTEMA DE KARMA ---
            # O karma (0+0j) já é criado pelo create_initial_agent
            initial_karma_state = complex(
                self.agent_states[agent_id]['karma']['real'],
                self.agent_states[agent_id]['karma']['imag']
            )
            # Adiciona o agente ao motor de reputação
            self.reputation_system.add_agent(agent_id, initial_karma_state)

            # --- Geração do Grafo de Progressão Individual ---
            self.current_floors[agent_id] = 0
            self.current_nodes[agent_id] = "start"
            self.nodes_per_floor_counters[agent_id] = {0: 1} # Cada agente tem seu contador
            self.graphs[agent_id] = nx.DiGraph() # Cada agente tem seu grafo
            self.graphs[agent_id].add_node("start", floor=0)
            
            # Gera conteúdo da sala inicial (vazia)
            start_content = content_generation.generate_room_content(
                self.catalogs, 
                budget=0,
                current_floor=0,
                guarantee_enemy=False
            )
            self.graphs[agent_id].nodes["start"]['content'] = start_content
            
            # Pré-gera o primeiro andar para este agente
            self._generate_and_populate_successors(agent_id, "start")
            
            # --- Logging e Retorno ---
            self._log(agent_id, 
                      f"[RESET] Novo episódio iniciado. {agent_name} (Nível {self.agent_states[agent_id]['level']}).\n"
                      f"[RESET] Mapa de progressão individual gerado. Sala inicial: 'start'.")
            
            successors = list(self.graphs[agent_id].successors(self.current_nodes[agent_id]))
            if successors:
                self._log(agent_id, f"[RESET] Próximas salas disponíveis: {', '.join(successors)}")
            
            # Coleta a observação inicial (agora com 36 estados)
            observations[agent_id] = self._get_observation(agent_id)
            infos[agent_id] = {} # Dicionário de info inicial é vazio

        # Retorna os dicionários de observação e info, conforme a API MAE
        return observations, infos

    def _handle_combat_turn(self, agent_id: str, action: int, agent_info_dict: dict) -> tuple[float, bool]:
        """Orquestra um único turno de combate PvE e retorna (recompensa, combate_terminou)."""
        
        # Pega o estado de combate PvE para ESTE agente
        combat_state = self.combat_states.get(agent_id) # Usa .get() por segurança

        # Se, por algum motivo, não houver estado de combate, encerra imediatamente.
        if not combat_state:            
            return -1, True # Penalidade pequena, combate terminou (bug)

        agent = combat_state['agent']
        enemy = combat_state['enemy']
        reward = 0
        combat_over = False

        # --- 1. AGENTE AGE ---
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

        # --- 2. VERIFICA MORTE DO INIMIGO, ADICIONA EXP E FAZ O LEVEL UP ---
        if combat.check_for_death_and_revive(enemy, self.catalogs):
            # PASSO 1: Dar a recompensa pela vitória e adicionar a EXP
            reward += 100 # Recompensa grande por vencer
            self.enemies_defeated_this_episode[agent_id] += 1
            
            # Pega o estado *principal* do agente (fora do combate)
            agent_main_state = self.agent_states[agent_id] 
            
            agent_main_state['exp'] += enemy.get('exp_yield', 50) # Ganha XP
            
            # PASSO 2: Chamar a lógica de level up IMEDIATAMENTE
            self._log(
                agent_id, # Passa o agent_id para o log
                f"[DIAGNÓSTICO] Inimigo '{enemy.get('name')}' derrotado. "
                f"EXP Ganhada: {enemy.get('exp_yield', 50)}. "
                f"EXP Total: {agent_main_state['exp']}. "
                f"EXP Nec: {agent_main_state['exp_to_level_up']}."
            )
            
            # Passa o estado principal para a regra de level up
            leveled_up = agent_rules.check_for_level_up(agent_main_state) 

            if leveled_up:
                self._log(agent_id, f"[DEBUG] {self.agent_names[agent_id]} subiu para o nível {agent_main_state['level']} durante o combate.")
                reward += 50  # Recompensa extra por subir de nível
                agent_info_dict['level_up'] = True # Usa o dict de info do agente

                # SINCRONIZAÇÃO CRÍTICA: Atualiza o agente 'agent' (cópia de combate) 
                # com os novos stats do 'agent_main_state' (principal)
                agent['hp'] = agent_main_state['hp']
                agent['max_hp'] = agent_main_state['max_hp']
                agent['base_stats'] = agent_main_state['base_stats'].copy()
                
                # Reseta os cooldowns no estado principal
                for skill in agent_main_state.get('cooldowns', {}):
                    agent_main_state['cooldowns'][skill] = 0
                
                # Sincroniza os cooldowns zerados de volta para a cópia de combate
                agent['cooldowns'] = agent_main_state['cooldowns'].copy()

            # PASSO 3: AGORA combate pode ser encerrado

            # Registra a duração do combate para esta luta
            duration = self.current_step - self.combat_states[agent_id]['start_step']
            self.pve_combat_durations[agent_id].append(duration)

            self.combat_states[agent_id] = None # Limpa o estado de combate deste agente
            combat_over = True
        
        # --- 3. INIMIGO AGE (se o combate não terminou) ---
        if not combat_over:
            hp_before_agent = agent['hp']
            
            # IA Simples (mantida)
            available_skills = [s for s, cd in enemy['cooldowns'].items() if cd == 0]
            enemy_action = random.choice(available_skills) if available_skills else "Wait"
            
            combat.execute_action(enemy, [agent], enemy_action, self.catalogs)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
            reward -= damage_taken * 0.5

            # 4. VERIFICA MORTE DO AGENTE
            if combat.check_for_death_and_revive(agent, self.catalogs):
                combat_over = True # O loop principal do step() aplicará a penalidade final

                # Registra a causa da morte para logs e análises
                self.death_cause[agent_id] = f"PvE: {enemy['name']} (Lvl {enemy['level']})"

        # --- 5. FIM DO TURNO: RESOLVE EFEITOS E COOLDOWNS ---
        
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
    
    def _handle_pvp_combat_turn(self, action_a1: str, action_a2: str) -> tuple[float, float, bool, str, str]:
        """
        Orquestra um único turno de combate PvP entre 'a1' e 'a2'.
        Morte = Dropa Loot + Chama Respawn.
        Retorna: (recompensa_a1, recompensa_a2, combate_terminou, id_vencedor, id_perdedor)
        """
        
        a1_combatant = self.pvp_state['a1']
        a2_combatant = self.pvp_state['a2']
        
        rew_a1 = 0
        rew_a2 = 0
        combat_over = False
        winner = None
        loser = None

        # --- 2. Agente 'a1' Age ---
        hp_before_a2 = a2_combatant['hp']
        combat.execute_action(a1_combatant, [a2_combatant], action_a1, self.catalogs)
        damage_dealt_by_a1 = hp_before_a2 - a2_combatant['hp']
        rew_a1 += damage_dealt_by_a1 * 0.6
        rew_a2 -= damage_dealt_by_a1 * 0.5
        
        # --- 3. Verifica se Agente 'a2' Morreu ---
        if combat.check_for_death_and_revive(a2_combatant, self.catalogs):
            rew_a1 += 200 # Recompensa por vencer
            rew_a2 -= 300 # Penalidade por morrer
            combat_over = True
            winner = 'a1'
            loser = 'a2'
                           
            # 1. Resolve Karma
            self._resolve_pvp_end_karma(winner, loser)
            
            # 2. Dropar o Loot
            self._drop_pvp_loot(loser_id=loser, winner_id=winner)
            
            # 3. Respawnar 
            self._respawn_agent(loser)     

        # --- 4. Agente 'a2' Age (Se o combate NÃO terminou) ---
        if not combat_over:
            hp_before_a1 = a1_combatant['hp']
            combat.execute_action(a2_combatant, [a1_combatant], action_a2, self.catalogs)
            damage_dealt_by_a2 = hp_before_a1 - a1_combatant['hp']
            rew_a2 += damage_dealt_by_a2 * 0.6
            rew_a1 -= damage_dealt_by_a2 * 0.5
            
            # --- 5. Verifica se Agente 'a1' Morreu ---
            if combat.check_for_death_and_revive(a1_combatant, self.catalogs):
                rew_a2 += 200 # Recompensa por vencer
                rew_a1 -= 300 # Penalidade por morrer
                combat_over = True
                winner = 'a2'
                loser = 'a1'
                                
                # 1. Resolve Karma
                self._resolve_pvp_end_karma(winner, loser)
                
                # 2. Dropar o Loot
                self._drop_pvp_loot(loser_id=loser, winner_id=winner)
                
                # 3. Respawnar 
                self._respawn_agent(loser)                 
                
        # --- 6. Fim do Turno (Se o combate NÃO terminou) ---
        if not combat_over:
            combat.resolve_turn_effects_and_cooldowns(a1_combatant, self.catalogs)
            combat.resolve_turn_effects_and_cooldowns(a2_combatant, self.catalogs)
        
        # --- 7. Sincronização Final com o Estado Mestre ---
        # A sincronização do perdedor não é mais necessária, pois ele foi resetado.
        # Sincroniza apenas o vencedor (se houver) ou ambos (se o combate continuar).
        if loser != 'a1': # Se a1 não perdeu (ou seja, a1 venceu OU o combate continua)
             self.agent_states['a1']['hp'] = a1_combatant['hp']
             self.agent_states['a1']['cooldowns'] = a1_combatant['cooldowns'].copy()
        if loser != 'a2': # Se a2 não perdeu
             self.agent_states['a2']['hp'] = a2_combatant['hp']
             self.agent_states['a2']['cooldowns'] = a2_combatant['cooldowns'].copy()        
        
        return rew_a1, rew_a2, combat_over, winner, loser
    
    def _drop_pvp_loot(self, loser_id: str, winner_id: str):
        """
        Pega todos os equipamentos do perdedor (estado mestre)
        e os joga no chão da sala da arena.
        """
        self._log(winner_id, f"[PVP] {self.agent_names[loser_id]} dropou seu equipamento.")
        
        # Pega o estado mestre do perdedor (ANTES do respawn)
        loser_main_state = self.agent_states[loser_id]
        
        # Pega a sala atual (ambos estão na mesma sala)
        current_node_id = self.current_nodes[winner_id] # Usa o nó do vencedor
        
        # Garante que a estrutura de conteúdo exista
        if 'content' not in self.arena_graph.nodes[current_node_id]:
             self.arena_graph.nodes[current_node_id]['content'] = {}
        room_items = self.arena_graph.nodes[current_node_id]['content'].setdefault('items', [])
        
        # Itera sobre os equipamentos do perdedor e joga no chão
        dropped_items = []
        if loser_main_state:
            for item_type, item_name in loser_main_state.get('equipment', {}).items():
                if item_name: # Se houver um item equipado nesse slot
                    room_items.append(item_name)
                    dropped_items.append(item_name)
        
        if dropped_items:
            self._log(winner_id, f"[PVP] Itens no chão: {', '.join(dropped_items)}")
            # O _get_observation do vencedor agora verá esses itens
    
    def _log(self, agent_id: str, message: str):
            """
            Função auxiliar para salvar na história do episódio E
            printar no console se self.verbose > 0.
            Agora usa agent_id para salvar no log correto.
            """
            if self.verbose > 0:
                # Adiciona um prefixo [A1] ou [A2] para clareza no console
                print(f"[{agent_id.upper()}] {message}") 
            
            # Garante que a lista de log para este agente existe
            if agent_id not in self.current_episode_logs:
                self.current_episode_logs[agent_id] = []
            
            # Salva na lista de log do agente específico
            self.current_episode_logs[agent_id].append(message + "\n")

    def _transition_to_arena(self, agent_id: str):
        """
        Lida com a transição de um agente para a Zona K (Santuário).
        Esta função será o "portão" para a lógica de sincronização.
        """        
        # 1. Adicionar agent_id ao self.agents_in_arena
        self.agents_in_arena.add(agent_id)
        self._log(agent_id, f"[ZONA K] {self.agent_names[agent_id]} chegou à Zona K e está esperando.")
        
        # 2. Verificar se *todos* os agentes estão em self.agents_in_arena
        if all(aid in self.agents_in_arena for aid in self.agent_ids):
            self._log(agent_id, "[ZONA K] Ambos os agentes presentes. Iniciando Arena PvP!")
            self.env_state = 'ARENA_INTERACTION'

            # Incrementa o contador de encontros na arena para ambos os agentes
            self.arena_encounters_this_episode['a1'] += 1
            self.arena_encounters_this_episode['a2'] += 1
            
            # 1. Gera a Topologia 
            self.arena_graph = map_generation.generate_k_zone_topology(
                floor_level=self.current_floors[agent_id],
                num_nodes=9,        # Fixo: Pequeno
                connectivity_prob=0.4 
            )
            
            # 2. POPULAR A ARENA 
            # Iteramos por todas as salas criadas para dar conteúdo a elas
            for node in self.arena_graph.nodes():            
                # Usamos o budget baseado no andar atual dos agentes
                budget = (100 + (self.current_floors[agent_id] * 10)) * self.budget_multiplier
                
                # Garante inimigos na arena
                content = content_generation.generate_room_content(
                    self.catalogs, 
                    budget=budget, 
                    current_floor=self.current_floors[agent_id],
                    guarantee_enemy=False # Deixa o RNG decidir, ou True se quiser caos total
                )

                # Força salas vazias (sem inimigos) a permanecerem vazias para validar a hipótese social
                content['enemies'] = []

                self.arena_graph.nodes[node]['content'] = content

            # 3. Mover Agentes (Posiciona em extremos opostos)
            nodes_list = sorted(list(self.arena_graph.nodes()))
            self.current_nodes['a1'] = nodes_list[0]
            self.current_nodes['a2'] = nodes_list[-1] # Última sala
            
            self._log('a1', f"[ZONA K] Entrou na Arena em {self.current_nodes['a1']}")
            self._log('a2', f"[ZONA K] Entrou na Arena em {self.current_nodes['a2']}")

        else:
            self.env_state = 'ARENA_SYNC'

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
            agent_reward += 400
            self._log(agent_id, f"[EPISÓDIO] FIM: {self.agent_names[agent_id]} VENCEU! (Chegou ao fim do Jogo)")

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
        
        # --- 3. Verifica Morte (LÓGICA DE RESPAWN) --- [cite: 43-46]
        if self.agent_states[agent_id]['hp'] <= 0:                        
            
            agent_reward = -300 # Penalidade de morte
            
            # Chama a função de respawn que reseta o agente para o Nível 1
            # (O log da morte já está dentro desta função)
            self._respawn_agent(agent_id)
            
            # O agente não está 'terminated', pois o episódio continua
            # O 'agent_reward' (-300) será retornado
        # --- FIM DA MUDANÇA ---

        # 4. Recompensa de Marco (Milestone) [cite: 48-52]
        # (Adicionada verificação 'not agent_terminated' para não recompensar um agente vencedor)
        if not agent_terminated and self.current_floors[agent_id] % 10 == 0 and self.current_floors[agent_id] > self.last_milestone_floors[agent_id]:
            agent_reward += 400                
            self._log(agent_id, f"[MARCO] {self.agent_names[agent_id]} alcançou o Andar {self.current_floors[agent_id]}! Bônus de +400.")
            self.last_milestone_floors[agent_id] = self.current_floors[agent_id]

        # 5. Verifica Transição para Arena
        # (O 'current_floor' será 0 após o respawn, então a checagem '... > 0' previne transição imediata)
        if not agent_terminated and self.current_floors[agent_id] > 0 and \
           self.current_floors[agent_id] % self.sanctum_floor == 0: # Ex: Andares 20, 40, 60...
            
            if agent_id not in self.agents_in_arena:
                 self._transition_to_arena(agent_id) 

        # 7. Cria 'final_status' se o episódio terminou para este agente
        if agent_terminated or global_truncated:
            infos[agent_id]['final_status'] = {
                'level': self.agent_states[agent_id]['level'],
                'hp': self.agent_states[agent_id]['hp'],
                'floor': self.current_floors[agent_id],
                'win': game_won and self.agent_states[agent_id]['hp'] > 0,
                'steps': self.current_step,
                'enemies_defeated': self.enemies_defeated_this_episode[agent_id],
                'invalid_actions': self.invalid_action_counts[agent_id],
                'agent_name': self.agent_names[agent_id],
                'full_log': self.current_episode_logs[agent_id],
                'exp': self.agent_states[agent_id]['exp'],
                'damage_dealt': self.damage_dealt_this_episode[agent_id],
                'equipment_swaps': self.equipment_swaps_this_episode[agent_id],
                'death_cause': self.death_cause[agent_id],
                'pve_durations': self.pve_combat_durations[agent_id],
                'pvp_durations': self.pvp_combat_durations[agent_id],
                'arena_encounters': self.arena_encounters_this_episode[agent_id],
                'pvp_combats': self.pvp_combats_this_episode[agent_id],
                'bargains_succeeded': self.bargains_succeeded_this_episode[agent_id],
                'cowardice_kills': self.cowardice_kills_this_episode[agent_id],
                'karma_history': self.karma_history[agent_id],
            }
            
        return agent_reward, agent_terminated

    def step(self, actions: dict):
        """
        Executa um passo no ambiente multiagente.
        
        :param actions: Dicionário de ações (ex: {'a1': 5, 'a2': 6})
        :return: Tupla MAE (obs_dict, rew_dict, term_dict, trunc_dict, info_dict)
        """
        
        # --- 1. Inicialização do Passo ---
        self.current_step += 1
        
        rewards = {agent_id: 0 for agent_id in self.agent_ids}
        terminateds = {agent_id: False for agent_id in self.agent_ids}
        infos = {agent_id: {} for agent_id in self.agent_ids}
        
        global_truncated = self.current_step >= self.max_episode_steps
        truncateds = {agent_id: global_truncated for agent_id in self.agent_ids}
        
        terminateds['__all__'] = False
        truncateds['__all__'] = global_truncated

        # --- 2. Máquina de Estados ---

        if self.env_state == 'PROGRESSION':
            # --- MODO PROGRESSÃO (Agentes separados) ---
            for agent_id, action in actions.items():
                if terminateds.get(agent_id, False): # Pula agente morto
                    continue
                
                # Processa o passo PvE para o agente
                agent_reward, agent_terminated = self._process_pve_step(
                    agent_id, action, global_truncated, infos
                )
                
                # Salva os resultados
                rewards[agent_id] = agent_reward
                terminateds[agent_id] = agent_terminated

        elif self.env_state == 'ARENA_SYNC':
            # --- MODO SINCRONIZAÇÃO (Um espera, o outro joga PvE) ---
            for agent_id, action in actions.items():
                if agent_id in self.agents_in_arena:
                    # Este agente está ESPERANDO
                    self._log(agent_id, "[ZONA K] Esperando o outro agente chegar...")
                    rewards[agent_id] = 0 # Nenhuma recompensa/penalidade por esperar
                    terminateds[agent_id] = False
                else:
                    # Este agente ainda está na P-Zone, processa seu passo PvE normalmente
                    agent_reward, agent_terminated = self._process_pve_step(
                        agent_id, action, global_truncated, infos
                    )
                    rewards[agent_id] = agent_reward
                    terminateds[agent_id] = agent_terminated
            
        elif self.env_state == 'ARENA_INTERACTION':
            # --- MODO ARENA (Lógica Social/PvP) ---
            
            # --- 0. VERIFICAR RITO DE PASSAGEM (Encontro) ---
            # Se ainda não se encontraram, verifica se estão na mesma sala agora
            if not self.arena_meet_occurred:
                if self.current_nodes['a1'] == self.current_nodes['a2']:
                    self.arena_meet_occurred = True
                    
                    self._log('a1', "[ZONA K] Encontro realizado! A saída foi destrancada.")
                    self._log('a2', "[ZONA K] Encontro realizado! A saída foi destrancada.")
                    
                    # Recompensa pelo "Rito de Passagem" (objetivo cumprido)
                    rewards['a1'] += 100
                    rewards['a2'] += 100

            # 1. Limpeza e Inicialização de Flags
            # Guardamos o passado para verificar traição 
            prev_social_flags = {
                agent_id: self.social_flags[agent_id].copy() 
                for agent_id in self.agent_ids
            }
            
            # Reseta flags do turno atual
            for agent_id in self.agent_ids:
                self.social_flags[agent_id]['just_picked_up'] = False
                # Drop Flag
                self.social_flags[agent_id]['just_dropped'] = False 
                # Nova Flag
                self.social_flags[agent_id]['skipped_attack'] = False 

            # 2. Captura Intenções
            action_a1 = actions['a1']
            action_a2 = actions['a2']
            
            # Verifica se estão na mesma sala para considerar "oportunidade de ataque"
            in_same_room = (self.current_nodes['a1'] == self.current_nodes['a2'])
            
            is_a1_attacking = (0 <= action_a1 <= 3)
            is_a2_attacking = (0 <= action_a2 <= 3)
            
            # Verifica se um dos agentes "pulou" o ataque
            if in_same_room:
                # Se A1 NÃO atacou (usou ação > 3), ele "pulou" o ataque
                if not is_a1_attacking:
                    self.social_flags['a1']['skipped_attack'] = True
                
                # Se A2 NÃO atacou
                if not is_a2_attacking:
                    self.social_flags['a2']['skipped_attack'] = True            
            
            # --- LÓGICA DE TRAIÇÃO (Baseada em Estado Persistente) ---
            # Cenário: Alguém ofereceu paz (estado True) e o outro ataca AGORA.
            
            # Traição de A2 contra A1
            if self.arena_interaction_state['a1']['offered_peace'] and is_a2_attacking:
                self._log('a2', f"[KARMA] TRAIÇÃO! {self.agent_names['a2']} atacou após oferta de paz. (--)")
                self.reputation_system.update_karma('a2', 'bad') 
                self.reputation_system.update_karma('a2', 'bad') # Penalidade dupla
                self.reputation_system.update_karma('a1', 'bad') # A1 aprende a desconfiar
                rewards['a2'] -= 50 # Penalidade imediata
                # A oferta foi traída, então ela é cancelada
                self.arena_interaction_state['a1']['offered_peace'] = False

            # Traição de A1 contra A2
            if self.arena_interaction_state['a2']['offered_peace'] and is_a1_attacking:
                self._log('a1', f"[KARMA] TRAIÇÃO! {self.agent_names['a1']} atacou após oferta de paz. (--)")
                self.reputation_system.update_karma('a1', 'bad')
                self.reputation_system.update_karma('a1', 'bad')
                self.reputation_system.update_karma('a2', 'bad')
                rewards['a1'] -= 50
                # A oferta foi traída, cancela
                self.arena_interaction_state['a2']['offered_peace'] = False

            # --- 3. PROCESSAR COMBATE (Se ativo ou iniciado agora) ---
            start_combat_now = False
            if not self.pvp_state and self.current_nodes['a1'] == self.current_nodes['a2']:
                if is_a1_attacking or is_a2_attacking:
                    attacker = 'a1' if is_a1_attacking else 'a2'
                    defender = 'a2' if attacker == 'a1' else 'a1'
                    
                    # Só inicia se o defensor estiver vivo
                    if not terminateds.get(defender, False):
                        self._log(attacker, f"[PVP] Combate iniciado!")
                        self._initiate_pvp_combat(attacker, defender)
                        rewards[attacker] += 20 
                        start_combat_now = True
                        # Se combate começou, qualquer oferta de paz é esquecida
                        self.arena_interaction_state['a1']['offered_peace'] = False
                        self.arena_interaction_state['a2']['offered_peace'] = False

            if self.pvp_state:
                # Resolve o turno de combate
                c_act_a1 = self.agent_skill_names[action_a1] if is_a1_attacking else "Wait"
                c_act_a2 = self.agent_skill_names[action_a2] if is_a2_attacking else "Wait"
                
                rew_a1, rew_a2, combat_over, winner, loser = self._handle_pvp_combat_turn(c_act_a1, c_act_a2)
                rewards['a1'] += rew_a1
                rewards['a2'] += rew_a2

                if combat_over:
                    self._log(winner, f"[PVP] VITORIA de {self.agent_names[winner]}!")                    
                    
                    terminateds[loser] = False # O perdedor respawnou, não terminou
                    self.pvp_state = None
                    self._end_arena_encounter(winner)

            # --- 4. PROCESSAR AÇÕES FORA DE COMBATE ---
            else:
                for agent_id in self.agent_ids:
                    if terminateds.get(agent_id, False): continue
                    action = actions[agent_id]
                    
                    # Ações de Exploração (Equipar, Social, Mover)
                    if 4 <= action <= 9:
                        # Chama a _handle_exploration_turn
                        # Ela seta 'just_picked_up' se equipar Artefato (Ação 4)
                        r_explore, _ = self._handle_exploration_turn(agent_id, action)
                        rewards[agent_id] += r_explore
                        
                        # ATUALIZAÇÃO DO ESTADO DE PAZ (Se dropar Artefato - Ação 5)
                        if action == 5 and r_explore > 0: # Se dropou com sucesso
                             self.arena_interaction_state[agent_id]['offered_peace'] = True
                    
                    elif not (0 <= action <= 3): # Inválida
                         self.invalid_action_counts[agent_id] += 1
                         rewards[agent_id] -= 5

                # --- 5. RESOLVER BARGANHA (Baseado em Estado Persistente + Ação Atual) ---
                if not self.pvp_state and self.current_nodes['a1'] == self.current_nodes['a2']:
                    
                    # Verifica se alguém pegou um item (Ação 4) NESTE turno
                    a1_picked_up = self.social_flags['a1'].get('just_picked_up', False)
                    a2_picked_up = self.social_flags['a2'].get('just_picked_up', False)

                    # Cenário 1: A1 ofereceu paz (estado persistente) E A2 pegou (ação agora)
                    bargain_1 = self.arena_interaction_state['a1']['offered_peace'] and a2_picked_up
                    
                    # Cenário 2: A2 ofereceu paz (estado persistente) E A1 pegou (ação agora)
                    bargain_2 = self.arena_interaction_state['a2']['offered_peace'] and a1_picked_up

                    if bargain_1 or bargain_2:
                        self._log('a1', "[KARMA] Barganha aceita! (+)")
                        self._log('a2', "[KARMA] Barganha aceita! (+)")
                        
                        self.reputation_system.update_karma('a1', 'good')
                        self.reputation_system.update_karma('a2', 'good')
                        
                        rewards['a1'] += 200
                        rewards['a2'] += 200
                        
                        # Reseta estados de paz
                        self.arena_interaction_state['a1']['offered_peace'] = False
                        self.arena_interaction_state['a2']['offered_peace'] = False
                        
                        self._end_arena_encounter('a1')
                        self._end_arena_encounter('a2')

                        # Registra a barganha sucedida
                        self.bargains_succeeded_this_episode['a1'] += 1
                        self.bargains_succeeded_this_episode['a2'] += 1                                    

        # --- 3. Finalização do Passo ---
        
        # Se *ambos* os agentes terminaram (morreram/venceram), o episódio global termina
        if all(terminateds.get(agent_id, False) for agent_id in self.agent_ids) or global_truncated:
            terminateds['__all__'] = True
            truncateds['__all__'] = True # Garante que truncado também termine tudo

        # Coleta as novas observações para todos os agentes
        observations = {
            agent_id: self._get_observation(agent_id) for agent_id in self.agent_ids
        }

        # Coleta de Histórico de Karma (Amostra a cada 100 steps)
        if self.current_step % 100 == 0:
            for agent_id in self.agent_ids:
                # Pega do sistema de reputação (número complexo)
                z = self.reputation_system.get_karma_state(agent_id)
                # Salva como dict
                self.karma_history[agent_id].append({'real': z.real, 'imag': z.imag})

        # Salva o 'final_status' se truncou globalmente
        if global_truncated:
            for agent_id in self.agent_ids:
                # Só preenche se ainda não tiver sido preenchido (ex: por morte no PvE)
                if 'final_status' not in infos[agent_id]:
                    infos[agent_id]['final_status'] = {
                        'level': self.agent_states[agent_id]['level'],
                        'hp': self.agent_states[agent_id]['hp'],
                        'floor': self.current_floors[agent_id],
                        'win': False, # Se truncou pelo tempo, não venceu
                        'steps': self.current_step,
                        'enemies_defeated': self.enemies_defeated_this_episode[agent_id],
                        'invalid_actions': self.invalid_action_counts[agent_id],
                        'agent_name': self.agent_names[agent_id],
                        'full_log': self.current_episode_logs[agent_id],
                        'equipment': self.agent_states[agent_id].get('equipment', {}),
                        'exp': self.agent_states[agent_id]['exp'],
                        'damage_dealt': self.damage_dealt_this_episode[agent_id],
                        'equipment_swaps': self.equipment_swaps_this_episode[agent_id],
                        'death_cause': self.death_cause[agent_id],
                        'pve_durations': self.pve_combat_durations[agent_id],
                        'pvp_durations': self.pvp_combat_durations[agent_id],
                        'arena_encounters': self.arena_encounters_this_episode[agent_id],
                        'pvp_combats': self.pvp_combats_this_episode[agent_id],
                        'bargains_succeeded': self.bargains_succeeded_this_episode[agent_id],
                        'cowardice_kills': self.cowardice_kills_this_episode[agent_id],
                        'karma_history': self.karma_history[agent_id],
                        'karma': self.agent_states[agent_id]['karma'] 
                    }
        
        # Retorna os dicionários no formato MAE completo
        return observations, rewards, terminateds, truncateds, infos
    
    def _handle_exploration_turn(self, agent_id: str, action: int):
        """
        Processa uma ação em modo de exploração (fora de combate).
        Versão FINAL: Inclui segurança, lógica social, equipamentos inteligentes e saída da arena.
        """
        reward = 0
        terminated = False
        
        # --- 1. Definições Comuns (Segurança) ---
        current_node_id = self.current_nodes[agent_id]
        
        # Verifica se ESTE AGENTE está na Arena
        is_in_arena = (self.env_state != 'PROGRESSION' and agent_id in self.agents_in_arena)
        
        # Seleciona o grafo correto
        current_graph = self.arena_graph if is_in_arena else self.graphs[agent_id]
            
        if current_graph is None:
            self._log(agent_id, f"[ERRO CRÍTICO] current_graph é None na ação {action}. Ignorando.")
            return -5.0, terminated

        # --- 2. Lógica das Ações ---

        # --- Ações de Movimento (6, 7, 8, 9) ---        
        if 6 <= action <= 9:
            neighbor_index = action - 6
            
            
            neighbors = list(current_graph.successors(current_node_id))
            
            neighbors.sort() 

            self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} na sala '{current_node_id}'. Tentando Mover Vizinho {neighbor_index}.")

            if neighbor_index < len(neighbors):
                # --- Movimento VÁLIDO ---
                chosen_node = neighbors[neighbor_index]                                             

                self._log(agent_id, f"[AÇÃO] Movimento VÁLIDO para '{chosen_node}'.")
                
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
                
                reward = -0.1 if is_in_arena else 5

                # Gera novos sucessores e Combate PvE (Apenas P-Zone)
                if not is_in_arena:
                    if not list(current_graph.successors(chosen_node)):
                        self._generate_and_populate_successors(agent_id, chosen_node)

                    room_content = current_graph.nodes[chosen_node].get('content', {})
                    enemy_names = room_content.get('enemies', [])
                    if enemy_names:                        
                        self._log(agent_id, f"[AÇÃO] >>> COMBATE INICIADO com {enemy_names[0]} <<<")
                        self._start_combat(agent_id, enemy_names[0])
                        reward += 10 
            else:
                # Movimento INVÁLIDO
                self._log(agent_id, f"[AÇÃO] Movimento INVÁLIDO (Vizinho {neighbor_index} vazio).")
                self.invalid_action_counts[agent_id] += 1
                reward = -5 

        # --- Ação 4: Equipar Item (Universal + Social) ---        
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
                self._log(agent_id, f"[AÇÃO] Equipou: '{best_item_to_equip}'.")
                
                # Flag de pick-up para barganha
                if best_item_type == 'Artifact':
                    self.social_flags[agent_id]['just_picked_up'] = True                

                if best_item_type == 'Artifact' and is_in_arena:
                     reward = 10 # Incentivo social
                else:
                     reward = 50 + (max_rarity_diff * 100) # Incentivo PvE
            else:
                if room_items:
                    reward = -2 # Itens ignorados
                else:
                    self.invalid_action_counts[agent_id] += 1                    
                    reward = -5 # Chão vazio

        # --- Ação 5: Dropar Artefato ---        
        elif action == 5:
            if not is_in_arena:
                self._log(agent_id, "[AÇÃO] Ação 5 inválida fora da Arena K.")
                self.invalid_action_counts[agent_id] += 1
                reward = -5
            else:
                equipped = self.agent_states[agent_id]['equipment'].get('Artifact')
                if equipped:
                    del self.agent_states[agent_id]['equipment']['Artifact']
                    room_items = current_graph.nodes[current_node_id]['content'].setdefault('items', [])
                    room_items.append(equipped)
                    
                    # Flag de drop para barganha
                    self.social_flags[agent_id]['just_dropped'] = True                    
                    
                    self._log(agent_id, f"[AÇÃO-ARENA] Dropou '{equipped}'.")                    
                else:
                    self.invalid_action_counts[agent_id] += 1
                    reward = -5
        
        # --- Outras Ações ---
        elif 0 <= action <= 3:                
            self.invalid_action_counts[agent_id] += 1
            reward = -5 
        else:
            self.invalid_action_counts[agent_id] += 1
            reward = -5

        return reward, terminated
    
    # Gerencia o movimento dentro da arena (Zona K)
    def _handle_arena_movement(self, agent_id: str, action: int) -> float:
        """
        Processa uma ação de movimento (6-9) dentro da Zona K (Arena).
        Usa o self.arena_graph (não-direcionado) e o MAX_NEIGHBORS.
        Retorna a recompensa pela ação.
        """
        
        # 1. Obter o índice do vizinho (Ação 6 -> 0, Ação 7 -> 1, etc.)
        neighbor_index = action - 6 # Mapeia 6-9 para 0-3

        # 2. Obter o estado atual da arena
        current_node_id = self.current_nodes[agent_id]
        
        # Na arena, usamos .neighbors() pois o grafo não é direcionado
        try:
            neighbors = list(self.arena_graph.neighbors(current_node_id))
        except nx.NetworkXError:
             # Fallback caso o nó não esteja no grafo (não deve acontecer)
            neighbors = []
            
        # 3. Ordenar vizinhos para consistência
        # Isso garante que a Ação 6 (Vizinho 0) sempre se refira ao mesmo nó
        # (ex: 'k_20_1' sempre virá antes de 'k_20_2')
        neighbors.sort() 
        
        # 4. Verificar se a ação é válida
        if 0 <= neighbor_index < len(neighbors):
            # --- Movimento VÁLIDO ---

            # Verifica se o nó escolhido é uma saída
            chosen_node = neighbors[neighbor_index]

            # Se o nó é uma saída...
            if node_data.get('is_exit', False):
                # mas o encontro AINDA NÃO ACONTECEU
                if not self.arena_meet_occurred:
                    self._log(agent_id, f"[AÇÃO-ARENA] A saída em '{chosen_node}' está TRANCADA. Encontre o outro agente primeiro!")
                    # Não penalizamos como inválida, mas impedimos o movimento (choque na porta)
                    return -1.0 
                
                # Se o encontro JÁ aconteceu, permite sair
                self._log(agent_id, f"[ZONA K] {self.agent_names[agent_id]} saiu da Arena!")
                self._end_arena_encounter(agent_id)
                return 50.0 # Recompensa por sair
            
            node_data = self.arena_graph.nodes[chosen_node]

            self.current_nodes[agent_id] = chosen_node # Move o agente
            
            # Atualiza o andar (embora na arena deva ser o mesmo)
            self.current_floors[agent_id] = self.arena_graph.nodes[chosen_node].get('floor', self.current_floors[agent_id])
            
            self._log(agent_id, f"[AÇÃO-ARENA] Movimento válido para '{chosen_node}'.")
            
            # Recompensa pequena por movimento na arena
            return -0.1
        
        else:
            # --- Movimento INVÁLIDO ---
            # O agente tentou mover para um slot de vizinho que não existe
            # (ex: Ação 8 (Vizinho 2), mas a sala só tem 2 vizinhos (Índice 0 e 1))
            self._log(agent_id, f"[AÇÃO-ARENA] Movimento INVÁLIDO. (Slot de Vizinho {neighbor_index} está vazio)")
            self.invalid_action_counts[agent_id] += 1
            return -5.0 # Penalidade por ação inválida
        
    def _handle_arena_social(self, agent_id: str) -> float:
        """
        Processa a Ação Social (Ação 5): APENAS Dropar o Artefato equipado.
        Ação 4 (Equipar) agora lida com o 'Pegar'.
        Retorna a recompensa pela ação.
        """
        self._log(agent_id, f"[AÇÃO] {self.agent_names[agent_id]} tentou Ação 5 (Drop Artifact).")

        # --- 1. Ação Social só é válida na Arena ---
        if self.env_state != 'ARENA_INTERACTION':
            self._log(agent_id, "[AÇÃO] Ação 5 (Drop Artifact) é inválida fora da Arena K.")
            self.invalid_action_counts[agent_id] += 1
            return -5.0

        # --- 2. Obter Estados Atuais ---
        agent_state = self.agent_states[agent_id]
        current_node_id = self.current_nodes[agent_id]
        
        # Garante que 'content' e 'items' existam no nó da arena
        if not self.arena_graph.has_node(current_node_id) or 'content' not in self.arena_graph.nodes[current_node_id]:
             self._log(agent_id, "[AÇÃO] Tentou dropar item em nó/sala inválida.")
             self.invalid_action_counts[agent_id] += 1
             return -5.0
             
        room_items = self.arena_graph.nodes[current_node_id]['content'].setdefault('items', [])
        
        # --- 3. Lógica da Ação (APENAS Dropar) ---

        # Verifica se o agente TEM um artefato no slot 'Artifact'
        equipped_artifact_name = agent_state['equipment'].get('Artifact')

        if equipped_artifact_name:
            # --- Ação VÁLIDA: Dropar o Artefato ---
            
            # Tira o artefato do agente (remove a chave do dicionário)
            del agent_state['equipment']['Artifact']
            
            # Coloca o artefato no chão da sala da ARENA
            room_items.append(equipped_artifact_name)
            
            # Seta a flag de 1 turno (para o outro agente ver)
            self.social_flags[agent_id]['just_dropped'] = True
            
            self._log(agent_id, f"[AÇÃO-ARENA] {self.agent_names[agent_id]} dropou '{equipped_artifact_name}' (iniciando barganha).")
            # Recompensa por iniciar interação social
            return 10.0 
        else:
            # --- Ação INVÁLIDA ---
            # (Agente tentou dropar, mas não tinha artefato equipado)
            self._log(agent_id, f"[AÇÃO-ARENA] Ação 5 (Drop Artifact) falhou. Nenhum artefato equipado para dropar.")
            self.invalid_action_counts[agent_id] += 1
            return -5.0 # Penalidade
    
    def _start_combat(self, agent_id: str, enemy_name: str):
        """
        Inicializa o estado de combate PvE para um agente específico.
        Cria cópias 'combatant' do agente e do inimigo.
        """
        
        # Pega o estado mestre do agente
        agent_main_state = self.agent_states[agent_id]
        
        # Inicializa o 'combatant' do agente para o combate
        agent_combatant = combat.initialize_combatant(
            name=self.agent_names[agent_id], # Usa o nome real do agente
            hp=agent_main_state['hp'], 
            equipment=list(agent_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, # Skills base são compartilhadas
            team=1, 
            catalogs=self.catalogs
        )
        # Copia os cooldowns atuais do estado mestre para a instância de combate
        agent_combatant['cooldowns'] = agent_main_state.get('cooldowns', {s: 0 for s in self.agent_skill_names}).copy()

        # --- Lógica de Escalonamento do Inimigo ---
        enemy_base = self.catalogs['enemies'][enemy_name]
        
        # Usa o andar do agente específico para o escalonamento
        agent_current_floor = self.current_floors[agent_id]
        
        # O escalonamento só começa a ter um efeito real a partir do andar 3
        effective_floor = max(0, agent_current_floor - 2)
        
        hp_scaling_factor = 1 + (effective_floor * 0.08)
        scaled_hp = int(enemy_base.get('hp', 50) * hp_scaling_factor)

        # Inicializa o 'combatant' do inimigo
        enemy_combatant = combat.initialize_combatant(
            name=enemy_name, hp=scaled_hp, equipment=enemy_base.get('equipment', []),
            skills=enemy_base.get('skills', []), team=2, catalogs=self.catalogs
        )

        # Aplica escalonamento de dano, exp e nível
        damage_scaling_factor = 1 + (effective_floor * 0.1)
        enemy_combatant['base_stats']['flat_damage_bonus'] *= damage_scaling_factor
        enemy_combatant['exp_yield'] = int(enemy_base.get('exp_yield', 20) * (1 + effective_floor * 0.15))
        enemy_combatant['level'] = agent_current_floor # Nível do inimigo = andar atual
        
        # Armazena o estado de combate no dicionário, usando o agent_id como chave
        self.combat_states[agent_id] = {
            'agent': agent_combatant,
            'enemy': enemy_combatant,
            'start_step': self.current_step
        }
        
        # Loga o início do combate, usando o agent_id
        self._log(
            agent_id,
            f"[COMBATE] {self.agent_names[agent_id]} (Nível {agent_main_state['level']}, HP {agent_combatant['hp']}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {enemy_combatant['hp']})"
        )  

    def _initiate_pvp_combat(self, attacker_id: str, defender_id: str):
        """
        Inicializa o estado de combate PvP entre dois agentes.
        Cria cópias 'combatant' de ambos e armazena em self.pvp_state.        
        """
        
        # 1. Pega os estados mestres de ambos os agentes
        attacker_main_state = self.agent_states[attacker_id]
        defender_main_state = self.agent_states[defender_id]

        # 2. Inicializa o 'combatant' para o Atacante (team=1)
        attacker_combatant = combat.initialize_combatant(
            name=self.agent_names[attacker_id], 
            hp=attacker_main_state['hp'], 
            equipment=list(attacker_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=1, 
            catalogs=self.catalogs
        )
        attacker_combatant['cooldowns'] = attacker_main_state.get('cooldowns', {s: 0 for s in self.agent_skill_names}).copy()

        # 3. Inicializa o 'combatant' para o Defensor (team=2)
        defender_combatant = combat.initialize_combatant(
            name=self.agent_names[defender_id], 
            hp=defender_main_state['hp'], 
            equipment=list(defender_main_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=2, 
            catalogs=self.catalogs
        )
        defender_combatant['cooldowns'] = defender_main_state.get('cooldowns', {s: 0 for s in self.agent_skill_names}).copy()

        # 4. Armazena os combatants no estado de PvP global
        # Garante que as chaves sejam 'a1' e 'a2' para consistência
        self.pvp_state = {
            'a1': attacker_combatant if attacker_id == 'a1' else defender_combatant,
            'a2': defender_combatant if attacker_id == 'a1' else attacker_combatant,            
            'start_step': self.current_step             
        }
                
        self.pvp_combats_this_episode[attacker_id] += 1
        self.pvp_combats_this_episode[defender_id] += 1        
        
        # Loga o início do combate
        log_message = (
            f"[PVP] Combate iniciado! "
            f"{attacker_combatant['name']} (Nível {attacker_main_state['level']}, HP {attacker_combatant['hp']}) "
            f"vs. {defender_combatant['name']} (Nível {defender_main_state['level']}, HP {defender_combatant['hp']})"
        )
        self._log(attacker_id, log_message)
        self._log(defender_id, log_message)

    def _resolve_pvp_end_karma(self, winner_id: str, loser_id: str):
        """
        Atualiza o karma do vencedor com base no contexto da luta (Regra de Covardia).
        Chamado pelo step() após o término de um combate PvP.        
        """
        
        # --- 1. Obter os Níveis dos Agentes (do estado mestre) ---
        try:
            winner_level = self.agent_states[winner_id]['level']
            loser_level = self.agent_states[loser_id]['level']
        except KeyError:
            self._log(winner_id, "[KARMA ERROR] Não foi possível encontrar os estados dos agentes para resolver o karma.")
            return

        level_difference = winner_level - loser_level
        
        # --- 2. Definir o Limiar de "Covardia" ---
        # (Ajuste este valor conforme necessário para o balanceamento)
        COWARDICE_THRESHOLD = 10 

        # --- 3. Aplicar Atualização de Karma ao Vencedor ---
        action_type = 'neutral' # Padrão para uma luta justa

        if level_difference > COWARDICE_THRESHOLD:
            # --- Regra de Covardia ---
            # O vencedor estava 10+ níveis *acima* do perdedor.
            self._log(winner_id, f"[KARMA] Covardia! (Nível {winner_level} vs {loser_level}). Karma (-)")
            action_type = 'bad'

            # Registra a covardia nas estatísticas do episódio
            self.cowardice_kills_this_episode[winner_id] += 1
            
        elif level_difference < -COWARDICE_THRESHOLD:
            # --- Regra "Davi vs. Golias" ---
            # O vencedor estava 10+ níveis *abaixo* do perdedor.
            self._log(winner_id, f"[KARMA] Vitória Heroica! (Nível {winner_level} vs {loser_level}). Karma (+)")
            action_type = 'good'
            
        else:
            # --- Luta Justa ---
            # A diferença de nível era pequena.
            self._log(winner_id, f"[KARMA] Luta Justa. (Nível {winner_level} vs {loser_level}). Karma (Neutro)")
            action_type = 'neutral' # O karma decairá lentamente para o centro
            
        # Chama o motor de reputação para atualizar o karma do VENCEDOR
        self.reputation_system.update_karma(winner_id, action_type)
        
        # (O karma do perdedor não é atualizado, pois seu episódio terminou)

    def _end_arena_encounter(self, agent_id: str):
        """
        Finaliza a participação de um agente na Arena K e o move para a próxima P-Zone.
        
        Chamado após uma vitória em PvP ou uma barganha bem-sucedida.
        """
        
        # 1. Verifica se o agente está na arena
        if agent_id not in self.agents_in_arena:
            self._log(agent_id, f"[WARN] _end_arena_encounter chamado para {agent_id}, mas ele não estava na arena.")
            return

        self._log(agent_id, f"[ZONA K] {self.agent_names[agent_id]} concluiu a arena.")
        
        # 2. Remove o agente da arena
        self.agents_in_arena.remove(agent_id)
        
        # 3. Determina o próximo andar de progressão
        # O andar atual é o da arena (ex: 20). O próximo será 21.
        current_arena_floor = self.current_floors[agent_id]
        next_p_floor = current_arena_floor + 1
        
        # 4. Encontra/Cria o nó inicial da próxima P-Zone
        # (Ex: "p_21_0")
        
        # Garante que o contador para este novo andar exista no grafo do agente
        if next_p_floor not in self.nodes_per_floor_counters[agent_id]:
            self.nodes_per_floor_counters[agent_id][next_p_floor] = 0
            
        next_node_index = self.nodes_per_floor_counters[agent_id][next_p_floor]
        next_p_node_id = f"p_{next_p_floor}_{next_node_index}"
        
        # Adiciona este novo nó ao grafo de progressão individual do agente
        agent_graph = self.graphs[agent_id]
        agent_graph.add_node(next_p_node_id, floor=next_p_floor)
        # (Opcional: adicionar uma aresta do nó da arena? Por enquanto, apenas o movemos)
        
        # Atualiza o contador de nós para este andar
        self.nodes_per_floor_counters[agent_id][next_p_floor] += 1

        # 5. Move o agente para este novo nó
        self.current_nodes[agent_id] = next_p_node_id
        self.current_floors[agent_id] = next_p_floor
        
        # 6. Popula o novo nó com conteúdo (vazio, pois é um "hub" de entrada)
        start_content = content_generation.generate_room_content(
            self.catalogs, budget=0, current_floor=next_p_floor, guarantee_enemy=False
        )
        agent_graph.nodes[next_p_node_id]['content'] = start_content
        
        # 7. Gera os primeiros sucessores da P-Zone (ex: p_22_0, p_22_1)
        self._generate_and_populate_successors(agent_id, next_p_node_id)
        self._log(agent_id, f"[PROGRESSION] {self.agent_names[agent_id]} entrou na P-Zone do Andar {next_p_floor} no nó '{next_p_node_id}'.")

        # 8. Se a arena estiver vazia, reseta o estado global
        if not self.agents_in_arena:
            self._log(agent_id, "[ZONA K] Arena vazia. Retornando ao modo de Progressão global.")
            self.env_state = 'PROGRESSION'
            self.arena_graph = None # Limpa o grafo da arena

    def _respawn_agent(self, agent_id: str):
        """
        Processa a "morte" de um agente.
        Reseta o progresso (Nível, P-Zone, Equipamentos) do agente,
        mas PRESERVA sua identidade e seu Karma.
        O agente é movido de volta para o início de uma P-Zone totalmente nova.
        """
        
        # 1. Loga a Morte
        self._log(agent_id, f"[MORTE] {self.agent_names[agent_id]} foi derrotado! Resetando o progresso da run.")

        # 2. Preservar Identidade e Karma
        agent_name = self.agent_names[agent_id]
        # Pega o estado de karma complexo (z) ATUAL do agente no motor de reputação
        preserved_karma_z = self.reputation_system.get_karma_state(agent_id)
        
        # 3. Resetar o Estado Mestre do Agente
        # Chama create_initial_agent para resetar Nível, HP, EXP, Equip, Artefato
        self.agent_states[agent_id] = agent_rules.create_initial_agent(agent_name)
        
        # Re-aplica as skills base
        self.agent_states[agent_id]['skills'] = self.agent_skill_names
        # Atribui um artefato comum inicial para validar H3
        self.agent_states[agent_id]['equipment']['Artifact'] = 'Amulet of Vigor'
        
        # 4. RE-APLICAR o Karma Preservado
        # O estado local do agente (para observação) é atualizado com o karma persistente
        self.agent_states[agent_id]['karma']['real'] = preserved_karma_z.real
        self.agent_states[agent_id]['karma']['imag'] = preserved_karma_z.imag
        
        # 5. Resetar Métricas da Run
        # Apenas adicionamos um marcador visual no log
        self.current_episode_logs[agent_id].append(f"\n{'='*20} RESPAWN {'='*20}\n")
        
        # Resetamos APENAS o que é específico da "vida" atual e afetaria a lógica
        self.last_milestone_floors[agent_id] = 0

        # 6. Limpar Estados Ativos
        self.combat_states[agent_id] = None # Limpa qualquer estado de combate (PvE ou PvP)
        if self.pvp_state and agent_id in self.pvp_state:
             self.pvp_state = None # Limpa o combate PvP se o agente estava nele
        if agent_id in self.agents_in_arena:
            self.agents_in_arena.remove(agent_id) # Remove da arena
            
            # Se a arena ficar vazia, reseta o estado global
            if not self.agents_in_arena:
                self._log(agent_id, "[ZONA K] Arena vazia. Retornando ao modo de Progressão global.")
                self.env_state = 'PROGRESSION'
                self.arena_graph = None # Limpa o grafo da arena

        # 7. Criar uma Nova P-Zone
        self.current_floors[agent_id] = 0
        self.current_nodes[agent_id] = "start"
        self.nodes_per_floor_counters[agent_id] = {0: 1} # Reseta o contador de nós
        self.graphs[agent_id] = nx.DiGraph() # Cria um grafo novo e vazio
        self.graphs[agent_id].add_node("start", floor=0)
        
        # 8. Popular e Gerar a Nova P-Zone
        start_content = content_generation.generate_room_content(
            self.catalogs, 
            budget=0,
            current_floor=0,
            guarantee_enemy=False
        )
        self.graphs[agent_id].nodes["start"]['content'] = start_content
        
        # Gera os sucessores do "start" (Andar 1)
        self._generate_and_populate_successors(agent_id, "start")
        
        self._log(agent_id, f"[RESPAWN] {agent_name} (Nível 1) iniciou em uma nova Zona de Progressão.")