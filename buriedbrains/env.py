# buriedbrains/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
import networkx as nx
import time

# Importando todos os módulos que criamos
from . import agent_rules
from . import combat
from . import content_generation
# from . import map_generation
from . import reputation

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
                 verbose: int = 0):
        super().__init__()        
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

        self.max_episode_steps = max_episode_steps # Define um limite de passos por episódio
        self.current_step = 0
        self.verbose = verbose 
        self.max_floors = max_floors # Limite máximo de andares para evitar loops infinitos
        self.max_level = max_level # Nível máximo do agente         
        self.budget_multiplier = budget_multiplier # Multiplicador de orçamento para geração de conteúdo
        self.enemies_defeated_this_episode = 0 # Conta inimigos derrotados
        self.invalid_action_count = 0 # Conta ações inválidas
        self.last_milestone_floor = 0 # Último andar que concedeu recompensa de marco
        self.nodes_per_floor = {} # Dicionário para contar nós por andar

        self.agent_name = "Agent_000" # Um nome padrão
        self.current_episode_log = []  # Lista para guardar a história do episódio

        # Pré-processamento: Injeta a chave 'name' em cada item dos catálogos
        for catalog_name, catalog_data in self.catalogs.items():
            if isinstance(catalog_data, dict):
                for name, data in catalog_data.items():
                    if isinstance(data, dict):
                        data['name'] = name
                        
        self.pool_costs = content_generation._calculate_costs(enemy_data['pools'])
        self.rarity_map = {'Common': 0.25, 'Rare': 0.5, 'Epic': 0.75, 'Legendary': 1.0}
        self.action_space = spaces.Discrete(8) # sem a ação de soltar item (social) 
        # self.action_space = spaces.Discrete(9) 
        observation_shape = (14,) # PvE 
        # observation_shape = (24,) # PvE + Social + Itens no Chão
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=observation_shape, dtype=np.float32)
        self.agent_state = None
        self.graph = None
        self.current_node = None
        self.current_floor = 0
        self.combat_state = None
        reputation_params = {'z_saint': 0.95, 'z_villain': -0.95, 'attraction': 0.5}
        self.reputation_system = reputation.HyperbolicReputationSystem(
            potential_func=reputation.saint_villain_potential,
            potential_params=reputation_params
        )
        self.agent_skill_names = [] # Para mapear ações a nomes de habilidades

        self.guarantee_enemy = guarantee_enemy

    def _generate_successors_for_node(self, parent_node: str):
        """
        Gera e popula os nós sucessores (filhos) para um determinado nó.
        """
        parent_floor = self.graph.nodes[parent_node].get('floor', 0)
        next_floor = parent_floor + 1
        branching_factor = 2

        # Gera as salas sucessoras
        for i in range(branching_factor):
                            
            # Garante que a chave do andar existe no contador
            if next_floor not in self.nodes_per_floor:
                self.nodes_per_floor[next_floor] = 0
                
            # Pega o índice atual para este andar
            current_index = self.nodes_per_floor[next_floor]
            
            # Cria o novo nome no formato desejado
            new_node = f"Sala {next_floor}_{current_index}" 
            
            # Incrementa o contador para o próximo nó neste andar
            self.nodes_per_floor[next_floor] += 1
            # --- FIM DA LÓGICA DE NOMENCLATURA ---
            
            self.graph.add_node(new_node, floor=next_floor)
            self.graph.add_edge(parent_node, new_node)
            
            # Popula este novo nó com conteúdo (como antes)
            budget = ( 100 + (next_floor * 10) ) * self.budget_multiplier
            content = content_generation.generate_room_content(
                self.catalogs,                 
                budget, 
                next_floor,
                guarantee_enemy=self.guarantee_enemy 
            )
            self.graph.nodes[new_node]['content'] = content

            # Pega os nomes das entidades geradas, ou 'Nenhum' se a lista estiver vazia
            enemy_log = content.get('enemies', []) or ['Nenhum']
            # Pega o nome do EVENTO que aconteceu (ex: Trap, Fountain, ou Nenhum se Treasure/None)
            event_outcome_log = content.get('events', []) or ['Nenhum']
            # Pega o nome do ITEM gerado pelo evento Treasure (se houver)
            item_generated_log = content.get('items', []) or ['Nenhum']
            # Pega o nome do EFEITO de sala
            effect_log = content.get('room_effects', []) or ['Nenhum']

            # Monta o log final com todas as informações CORRETAS
            self._log(f"[MAPA] Sala '{new_node}' (Andar {next_floor}) gerada com: "
                        f"Inimigo: {enemy_log[0]}, "
                        f"Evento: {event_outcome_log[0]}, "
                        f"Item Gerado: {item_generated_log[0]}, "
                        f"Efeito: {effect_log[0]}")

    # VERSÃO MODIFICADA DA FUNÇÃO DE OBSERVAÇÃO, REMOVENDO BLOCOS SOCIAIS E ITENS SOCIAIS
    def _get_observation(self) -> np.ndarray:
        # O shape agora é (14,)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        agent = self.agent_state

        # --- Bloco Próprio (7 valores) ---
        # (Antigo obs[7] (has_artifact) foi removido)
        obs[0] = (agent['hp'] / agent['max_hp']) * 2 - 1 if agent['max_hp'] > 0 else -1.0
        obs[1] = (agent['level'] / self.max_level) * 2 - 1
        obs[2] = (agent['exp'] / agent['exp_to_level_up']) if agent['exp_to_level_up'] > 0 else 0.0
        for i in range(4):
            skill_name = self.agent_skill_names[i] if i < len(self.agent_skill_names) else None
            if skill_name:
                max_cd = self.catalogs['skills'].get(skill_name, {}).get('cd', 1)
                current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
                obs[3 + i] = (current_cd / max_cd) if max_cd > 0 else 0.0

        # --- Bloco Ambiente e Contexto ---
        room_content = self.graph.nodes[self.current_node].get('content', {})
        enemy_in_combat = self.combat_state.get('enemy') if self.combat_state else None
        
        # --- Flags de Sala (3 valores) ---
        items_events = room_content.get('items', []) or room_content.get('room_effects', [])
        if self.combat_state:
            obs[7] = 1.0 # (Antigo obs[8])
        elif items_events:
            obs[8] = 1.0 # (Antigo obs[9])
        else:
            obs[9] = 1.0 # (Antigo obs[10])

        # --- Informações de Combate (2 valores) ---
        if enemy_in_combat:
            obs[10] = (enemy_in_combat['hp'] / enemy_in_combat['max_hp']) * 2 - 1 if enemy_in_combat['max_hp'] > 0 else -1.0 # (Antigo obs[11])
            obs[11] = enemy_in_combat.get('level', 1) / 50.0 * 2 - 1 # (Antigo obs[12])

        # --- Bloco Itens no Chão (2 valores) ---
        # (Antigos obs[13] e obs[14] (artifact) foram removidos)
        items_in_room = room_content.get('items', [])
        if items_in_room:
            found_equip = False
            for item_name in items_in_room:
                details = self.catalogs['equipment'].get(item_name, {})
                item_type = details.get('type')
                
                # Procura APENAS por 'Weapon' ou 'Armor'
                if item_type in ['Weapon', 'Armor'] and not found_equip:
                    rarity = details.get('rarity')
                    obs[12] = 1.0 # (Antigo obs[15])
                    obs[13] = self.rarity_map.get(rarity, 0.0) # (Antigo obs[16])
                    found_equip = True
                    break # Encontrou o único item de poder que importa
        
        return obs
    
    # VERSÃO COMPLETA DA FUNÇÃO DE OBSERVAÇÃO, INCLUINDO BLOCOS SOCIAIS E ITENS NO CHÃO
    # def _get_observation(self) -> np.ndarray:
    #     obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    #     agent = self.agent_state

    #     # --- Bloco Próprio (8 valores) ---
    #     obs[0] = (agent['hp'] / agent['max_hp']) * 2 - 1 if agent['max_hp'] > 0 else -1.0
    #     obs[1] = (agent['level'] / self.max_level) * 2 - 1
    #     # >> Normalização de ratio para [0, 1] <<
    #     obs[2] = (agent['exp'] / agent['exp_to_level_up']) if agent['exp_to_level_up'] > 0 else 0.0
    #     for i in range(4):
    #         skill_name = self.agent_skill_names[i] if i < len(self.agent_skill_names) else None
    #         if skill_name:
    #             max_cd = self.catalogs['skills'].get(skill_name, {}).get('cd', 1)
    #             current_cd = agent.get('cooldowns', {}).get(skill_name, 0)
    #             obs[3 + i] = (current_cd / max_cd) if max_cd > 0 else 0.0
    #     obs[7] = 1.0 if 'Artifact' in agent and agent['Artifact'] is not None else -1.0

    #     # --- Bloco Ambiente e Contexto ---
    #     room_content = self.graph.nodes[self.current_node].get('content', {})
    #     enemy_in_combat = self.combat_state.get('enemy') if self.combat_state else None
        
    #     # Flags de Sala (4 valores)
    #     other_agents = room_content.get('agents', [])
    #     items_events = room_content.get('items', []) or room_content.get('room_effects', [])
    #     if self.combat_state:
    #         obs[8] = 1.0
    #     elif other_agents:
    #         obs[9] = 1.0
    #     elif items_events:
    #         obs[10] = 1.0
    #     else:
    #         obs[11] = 1.0

    #     # Informações de Combate (2 valores)
    #     if enemy_in_combat:
    #         obs[12] = (enemy_in_combat['hp'] / enemy_in_combat['max_hp']) * 2 - 1 if enemy_in_combat['max_hp'] > 0 else -1.0
    #         obs[13] = enemy_in_combat.get('level', 1) / 50.0 * 2 - 1

    #     # Bloco Social (6 valores)
    #     if other_agents:
    #         other_agent = other_agents[0]
    #         obs[14] = 1.0
    #         obs[15] = (other_agent.get('hp', 0) / other_agent.get('max_hp', 1)) * 2 - 1
    #         other_karma_z = self.reputation_system.get_karma_state(other_agent.get('id', 'other'))
    #         obs[18] = other_karma_z.real
    #         obs[19] = other_karma_z.imag
    #     my_karma_z = self.reputation_system.get_karma_state(self.agent_state.get('id', 'player'))
    #     obs[16] = my_karma_z.real
    #     obs[17] = my_karma_z.imag

    #     # >> Lógica de Itens no Chão em um único loop <<
    #     # Bloco Itens no Chão (4 valores)
    #     items_in_room = room_content.get('items', [])
    #     if items_in_room:
    #         found_Artifact = False
    #         found_equip = False
    #         for item_name in items_in_room:
    #             details = self.catalogs['equipment'].get(item_name, {})
    #             item_type = details.get('type')
    #             rarity = details.get('rarity')
                
    #             if item_type == 'Artifact' and not found_Artifact:
    #                 obs[20] = 1.0
    #                 obs[21] = self.rarity_map.get(rarity, 0.0)
    #                 found_Artifact = True
    #             elif item_type in ['Weapon', 'Armor'] and not found_equip:
    #                 obs[22] = 1.0
    #                 obs[23] = self.rarity_map.get(rarity, 0.0)
    #                 found_equip = True
                
    #             if found_Artifact and found_equip:
    #                 break
        
    #     return obs

# env.py - Substitua sua função reset por esta

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Semeia self.np_random
        if seed is not None:
            random.seed(seed) # Garante que o 'random' global também seja semeado

        # Hall da Fama: Gera um nome único para o agente
        gerador = GeradorNomes()
        self.agent_name = gerador.gerar_nome()
        self.current_episode_log = [] # Reinicia o gravador de história

        self.enemies_defeated_this_episode = 0
        self.invalid_action_count = 0
        self.last_milestone_floor = 0

        self.agent_state = agent_rules.create_initial_agent(self.agent_name)
        self.agent_state['name'] = self.agent_name

        self.agent_skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]
        self.agent_state['skills'] = self.agent_skill_names        
        self.reputation_system.add_agent(self.agent_name)
                
        # --- LÓGICA DE GERAÇÃO DO MAPA ---
        self.current_floor = 0
        self.graph = nx.DiGraph()
        self.current_node = "start"
        self.nodes_per_floor = {0: 1}
        
        self.graph.add_node("start", floor=0)
        
        start_content = content_generation.generate_room_content(
            self.catalogs, 
            budget=0,
            current_floor=0,
            guarantee_enemy=False
        )
        self.graph.nodes["start"]['content'] = start_content
        
        self._generate_successors_for_node("start") # Pré-gera o primeiro nível
        # --- FIM DA GERAÇÃO DO MAPA ---

        self.combat_state = None
        self.current_step = 0
        
        self._log(f"[RESET] Novo episódio iniciado. {self.agent_name} (Nível {self.agent_state['level']}).\n"
                  f"[RESET] Mapa gerado. Sala inicial: '{self.current_node}'.")
                
        successors = list(self.graph.successors(self.current_node))
        if successors:
            self._log(f"[RESET] Próximas salas disponíveis: {', '.join(successors)}")        
        
        return self._get_observation(), {}

    def _handle_combat_turn(self, action: int, info: dict) -> tuple[float, bool]:
        """Orquestra um único turno de combate e retorna (recompensa, combate_terminou)."""
        agent = self.combat_state['agent']
        enemy = self.combat_state['enemy']
        reward = 0
        combat_over = False

        # 1. AGENTE AGE
        hp_before_enemy = enemy['hp']
        action_name = self.agent_skill_names[action] if action < len(self.agent_skill_names) else "Wait"
        
        combat.execute_action(agent, [enemy], action_name, self.catalogs)

        # Recompensa por dano causado
        damage_dealt = hp_before_enemy - enemy['hp']
        reward += damage_dealt * 0.6

        # 2. VERIFICA MORTE DO INIMIGO, ADICIONA EXP E FAZ O LEVEL UP
        if combat.check_for_death_and_revive(enemy, self.catalogs):
            # PASSO 1: Dar a recompensa pela vitória e adicionar a EXP
            reward += 100 # Recompensa grande por vencer
            self.enemies_defeated_this_episode += 1
            self.agent_state['exp'] += enemy.get('exp_yield', 50) # Ganha XP
            
            # PASSO 2: Chamar a lógica de level up IMEDIATAMENTE
            
            self._log(
                f"[DIAGNÓSTICO] Inimigo '{enemy.get('name')}' derrotado. "
                f"EXP Ganhada: {enemy.get('exp_yield', 50)}. "
                f"EXP Total: {self.agent_state['exp']}. "
                f"EXP Nec: {self.agent_state['exp_to_level_up']}."
            )
            leveled_up = agent_rules.check_for_level_up(self.agent_state)

            if leveled_up:                
                self._log(f"[DEBUG] {self.agent_name} subiu para o nível {self.agent_state['level']} durante o combate.")                

                reward += 50  # Recompensa extra por subir de nível                
                info['level_up'] = True

                # SINCRONIZAÇÃO CRÍTICA: Atualiza o agente em combate com os novos status                
                agent_in_combat = self.combat_state['agent']
                agent_in_combat['hp'] = self.agent_state['hp']
                agent_in_combat['max_hp'] = self.agent_state['max_hp']
                agent_in_combat['base_stats'] = self.agent_state['base_stats'].copy()
                
                # Reseta os cooldowns no estado principal
                for skill in self.agent_state.get('cooldowns', {}):
                    self.agent_state['cooldowns'][skill] = 0

            # PASSO 3: AGORA combate pode ser encerrado
            self.combat_state = None
            combat_over = True
        
        # 3. INIMIGO AGE (se o combate não terminou)
        if not combat_over:
            hp_before_agent = agent['hp']
            
            # IA Simples: escolhe uma habilidade aleatória que não esteja em cooldown
            available_skills = [s for s, cd in enemy['cooldowns'].items() if cd == 0]
            enemy_action = random.choice(available_skills) if available_skills else "Wait"            
            
            combat.execute_action(enemy, [agent], enemy_action, self.catalogs)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
            reward -= damage_taken * 0.5

            # 4. VERIFICA MORTE DO AGENTE
            if combat.check_for_death_and_revive(agent, self.catalogs):
                combat_over = True # O loop principal do step() aplicará a penalidade final

        # 5. FIM DO TURNO: RESOLVE EFEITOS E COOLDOWNS
        # A sincronização de HP é a última coisa a acontecer no turno
        if self.combat_state:
            combat.resolve_turn_effects_and_cooldowns(agent, self.catalogs)
            if not combat_over: # Garante que não tentemos resolver efeitos de um inimigo morto
                combat.resolve_turn_effects_and_cooldowns(enemy, self.catalogs)
        
        # Sincroniza o HP do agente principal
        self.agent_state['hp'] = agent['hp']
        
        return reward, combat_over
    
    def _log(self, message: str):
            """
            Função auxiliar para salvar na história do episódio E
            printar no console se self.verbose > 0.
            """
            if self.verbose > 0:
                print(message) # Printa no console se verbose=1 ou mais
            self.current_episode_log.append(message + "\n") # Sempre salva na lista

    def step(self, action: int):
            reward = 0
            terminated = False
            info = {}
            game_won = False # <-- Variável para rastrear a vitória

            self.current_step += 1 # Incrementa o contador a cada passo
            truncated = self.current_step >= self.max_episode_steps

            # Penalidade de tempo: o agente perde um pouco a cada passo.
            reward -= 0.5

            # Verifica se o agente está no último andar E não tem para onde ir
            is_on_last_floor = (self.current_floor == self.max_floors)
            has_no_successors = not list(self.graph.successors(self.current_node))

            if is_on_last_floor and has_no_successors and not self.combat_state:
                terminated = True
                reward += 400  # Recompensa final por chegar ao fim
                game_won = True # <-- Define a vitória
                self._log(f"[EPISÓDIO] FIM: {self.agent_name} VENCEU! (Chegou ao fim do labirinto no Andar {self.current_floor})")                

            if self.combat_state:
                # Em combate: usa o handler de combate
                reward_combat, combat_over = self._handle_combat_turn(action, info)
                reward += reward_combat # Adiciona a recompensa do combate
                
                if combat_over and self.agent_state['hp'] > 0:
                    # Se o combate terminou com vitória, remove o inimigo do conteúdo da sala
                    room_content = self.graph.nodes[self.current_node]['content']
                    if room_content.get('enemies'):
                        room_content['enemies'].pop(0)
            else:
                # Fora de combate: usa o handler de exploração
                reward_explore, terminated_explore = self._handle_exploration_turn(action)
                                
                # Adiciona a recompensa da exploração ao total
                reward += reward_explore
                # Se a exploração retornou 'True' para terminado, atualiza
                if terminated_explore:
                    terminated = True                

            # Se o agente morrer, é 'terminated', não 'truncated'
            if self.agent_state['hp'] <= 0:
                terminated = True
                truncated = False 
                reward = -300  # Penalidade de morte aumentada para ser mais significativa
                self._log(f"[EPISÓDIO] FIM: {self.agent_name} MORREU no Andar {self.current_floor}.")                

            # Recompensa por vencer o jogo (condição de andar máximo)
            if self.current_floor > self.max_floors:
                terminated = True
                reward += 400  # Recompensa final
                game_won = True # <-- Define a vitória
                self._log(f"[EPISÓDIO] FIM: {self.agent_name} VENCEU! (Chegou ao andar {self.current_floor})")                
                # REMOVEMOS o return imediato daqui

            # Recompensa a cada 10 andares concluídos
            if self.current_floor % 10 == 0 and self.current_floor > self.last_milestone_floor:
                reward += 400                
                self._log(f"[MARCO] {self.agent_name} alcançou o Andar {self.current_floor}! Bônus de +400.")                
                self.last_milestone_floor = self.current_floor  # Atualiza o último marco alcançado
                # REMOVEMOS o bloco "if terminated or truncated" que estava aninhado aqui

            # Log se for truncado (limite de tempo)
            if truncated and not terminated:                        
                self._log(f"[EPISÓDIO] Encerrado por tempo limite no andar {self.current_floor}.")

            # --- BLOCO DE INFO FINAL UNIFICADO ---
            # Se o episódio terminou por QUALQUER motivo (morte, vitória, tempo),
            # nós populamos o dicionário 'info' aqui.
            if terminated or truncated:
                info['final_status'] = {
                    'level': self.agent_state['level'],
                    'hp': self.agent_state['hp'],
                    'floor': self.current_floor,
                    'win': game_won and self.agent_state['hp'] > 0, # Vitória SÓ é verdade se 'game_won' foi marcado
                    'steps': self.current_step,
                    'enemies_defeated': self.enemies_defeated_this_episode,
                    'invalid_actions': self.invalid_action_count,
                    'agent_name': self.agent_name,
                    'full_log': self.current_episode_log # Passa a história inteira
                }
                
            observation = self._get_observation()        

            # Este é agora o ÚNICO ponto de retorno da função
            return observation, reward, terminated, truncated, info

    # VERSÃO SIMPLIFICADA DA FUNÇÃO DE EXPLORAÇÃO, SEM AÇÃO DE SOLTAR/PEGAR ARTEFATO
    def _handle_exploration_turn(self, action: int):
            """
            Processa uma ação em modo de exploração, incluindo a poda do mapa.
            Retorna (recompensa, episodio_terminou).
            """
            reward = 0
            terminated = False
            
            # Ações 5 e 6: Movimento
            if 4 < action < 7:
                successors = list(self.graph.successors(self.current_node))
                action_index = action - 5  # Mapeia ação 5 para índice 0, 6 para 1
                
                self._log(f"[AÇÃO] {self.agent_name} na sala '{self.current_node}'. Tentando Ação de Movimento {action_index}.")
                if len(successors) > action_index:
                    # --- LÓGICA DE PODA E GERAÇÃO ---
                    # 1. Agente escolhe um caminho
                    chosen_node = successors[action_index]
                    
                    self._log(f"[AÇÃO] Movimento VÁLIDO para '{chosen_node}'.")
                                        # 2. PODA: Identifica e remove os caminhos não escolhidos
                    nodes_to_remove = [succ for i, succ in enumerate(successors) if i != action_index]
                    self.graph.remove_nodes_from(nodes_to_remove) # Isso remove os nós e todas as suas sub-árvores

                    # 3. GERAÇÃO: Move o agente e gera os próximos caminhos
                    self.current_node = chosen_node
                    
                    self.current_floor = self.graph.nodes[self.current_node].get('floor', self.current_floor)
                    self._generate_successors_for_node(self.current_node) # Gera os filhos do novo nó
                    
                    reward = 5  # Recompensa por explorar

                    # --- FIM DA LÓGICA DE PODA ---

                    # Após mover, verifica se a nova sala inicia um combate
                    room_content = self.graph.nodes[self.current_node].get('content', {})
                    enemy_names = room_content.get('enemies', [])
                    if enemy_names:                        
                        self._log(f"[AÇÃO] >>> COMBATE INICIADO com {enemy_names[0]} <<<")
                        self._start_combat(enemy_names[0])
                        reward += 10  # Recompensa bônus por encontrar um desafio

                else:                    
                    self._log(f"[AÇÃO] Movimento INVÁLIDO. (Sem sala no índice {action_index})")
                    self.invalid_action_count += 1
                    reward = -5
            
            # Ação 7: Equipar Item do Chão
            elif action == 7:                
                self._log(f"[AÇÃO] {self.agent_name} tentou Ação 7 (Equipar Item).")
                room_items = self.graph.nodes[self.current_node].get('content', {}).setdefault('items', [])
                equip_on_floor = next(
                    (item for item in room_items if self.catalogs['equipment'].get(item, {}).get('type') in ['Weapon', 'Armor']),
                    None
                )

                if equip_on_floor:
                    details = self.catalogs['equipment'][equip_on_floor]
                    item_type = details['type']
                    self.agent_state['equipment'][item_type] = equip_on_floor
                    room_items.remove(equip_on_floor)
                    
                    self._log(f"[AÇÃO] {self.agent_name} equipou '{equip_on_floor}' ({item_type}).")
                    reward = 50
                else:
                    self.invalid_action_count += 1                    
                    self._log(f"[AÇÃO] {self.agent_name} tentou equipar, mas não havia itens (Weapon/Armor) no chão.")
                    reward = -1

            
            # Ações de Combate (0 a 4) ou Ação Inválida (antiga 7)
            else:                
                self._log(f"[AÇÃO] {self.agent_name} usou Ação {action} (Combate/Inválida) fora de combate.")
                self.invalid_action_count += 1
                reward = -5 

            return reward, terminated
    
    # # VERSÃO COMPLETA DA FUNÇÃO DE EXPLORAÇÃO, INCLUINDO AÇÃO DE SOLTAR/PEGAR ARTEFATO
    # def _handle_exploration_turn(self, action: int):
    #     """
    #     Processa uma ação tomada pelo agente em modo de exploração.
    #     Retorna (recompensa, episodio_terminou).
    #     """
    #     reward = 0
    #     terminated = False
        
    #     # Ações 5 e 6: Movimento
    #     if 4 < action < 7:
    #         successors = list(self.graph.successors(self.current_node))
    #         action_index = action - 5  # Mapeia ação 5 para índice 0, 6 para 1

    #         if len(successors) > action_index:
    #             # Movimento válido: atualiza a posição do agente
    #             self.current_node = successors[action_index]
    #             reward = 5  # Recompensa por explorar

    #             # Após mover, verifica se a nova sala inicia um combate
    #             room_content = self.graph.nodes[self.current_node].get('content', {})
    #             enemy_names = room_content.get('enemies', [])
    #             if enemy_names:
    #                 self._start_combat(enemy_names[0])
    #                 reward += 10 # Recompensa bônus por encontrar um desafio
    #         else:
    #             # Movimento inválido (tentou ir para uma sala que não existe)
    #             reward = -5
        
    #     # Ação 7: Soltar/Pegar Artifact
    #     elif action == 7:
    #         room_items = self.graph.nodes[self.current_node].get('content', {}).setdefault('items', [])
    #         agent_has_Artifact = 'Artifact' in self.agent_state and self.agent_state['Artifact'] is not None
    #         artifact_on_floor = next((item for item in room_items if self.catalogs['equipment'].get(item, {}).get('type') == 'Artifact'), None)

    #         if agent_has_Artifact:
    #             # Ação: Soltar a Artifact que possui
    #             room_items.append(self.agent_state['Artifact'])
    #             self.agent_state['Artifact'] = None
    #             reward = 1 # Recompensa pequena por uma ação social (uma "oferta")
    #         elif artifact_on_floor:
    #             # Ação: Pegar a Artifact do chão
    #             self.agent_state['Artifact'] = artifact_on_floor
    #             room_items.remove(artifact_on_floor)
    #             reward = 20 # Recompensa maior por adquirir um item de valor
    #         else:
    #             # Ação inválida (sem Artifact para soltar ou pegar)
    #             reward = -1

    #     # Ação 8: Equipar Item do Chão
    #     elif action == 8:
    #         room_items = self.graph.nodes[self.current_node].get('content', {}).setdefault('items', [])
    #         equip_on_floor = next((item for item in room_items if self.catalogs['equipment'].get(item, {}).get('type') in ['Weapon', 'Armor', 'Artifact']), None)
            
    #         if equip_on_floor:
    #             # Ação: Equipa o item, destruindo o antigo no mesmo slot
    #             details = self.catalogs['equipment'][equip_on_floor]
    #             item_type = details['type']
    #             self.agent_state['equipment'][item_type] = equip_on_floor
    #             room_items.remove(equip_on_floor)
    #             reward = 15 # Recompensa alta por um upgrade de poder
    #         else:
    #             # Ação inválida (nenhum equipamento no chão)
    #             reward = -1
        
    #     # Ações de Combate (0 a 4) usadas fora de combate
    #     else:
    #         reward = -5 # Penalidade por usar uma ação de combate quando não há inimigos

    #     return reward, terminated
    
    def _start_combat(self, enemy_name: str):
        """Inicializa o estado de combate."""
        # Inicializa o agente para o combate
        agent_combatant = combat.initialize_combatant(
            name="Player 1", 
            hp=self.agent_state['hp'], 
            equipment=list(self.agent_state.get('equipment', {}).values()), 
            skills=self.agent_skill_names, 
            team=1, 
            catalogs=self.catalogs
        )
        agent_combatant['cooldowns'] = self.agent_state.get('cooldowns', {s: 0 for s in self.agent_skill_names})

        # >> ALTERAÇÃO 4: INIMIGOS "SACO DE BOXE" NOS PRIMEIROS ANDARES <<
        enemy_base = self.catalogs['enemies'][enemy_name]
        
        # O escalonamento só começa a ter um efeito real a partir do andar 3
        effective_floor = max(0, self.current_floor - 2)
        
        hp_scaling_factor = 1 + (effective_floor * 0.08) # HP escala mais lentamente no início
        scaled_hp = int(enemy_base.get('hp', 50) * hp_scaling_factor)

        enemy_combatant = combat.initialize_combatant(
            name=enemy_name, hp=scaled_hp, equipment=enemy_base.get('equipment', []),
            skills=enemy_base.get('skills', []), team=2, catalogs=self.catalogs
        )

        damage_scaling_factor = 1 + (effective_floor * 0.1)
        enemy_combatant['base_stats']['flat_damage_bonus'] *= damage_scaling_factor
        enemy_combatant['exp_yield'] = int(enemy_base.get('exp_yield', 20) * (1 + effective_floor * 0.15))
        enemy_combatant['level'] = self.current_floor
        
        self.combat_state = {
            'agent': agent_combatant,
            'enemy': enemy_combatant,
        }
        
        self._log(
            f"[COMBATE] {self.agent_name} (Nível {self.agent_state['level']}, HP {agent_combatant['hp']}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {enemy_combatant['hp']})"
        )        

class GeradorNomes:
    """
    Uma classe para gerar nomes de agentes únicos e variados para um roguelike.
    
    Ela armazena os nomes já gerados para garantir que nunca haja duplicatas
    dentro da mesma instância.
    """
    
    def __init__(self, seed=None):
        """
        Inicializa o gerador.
        
        :param seed: (Opcional) Uma seed para o gerador aleatório. 
                     Útil se você quiser gerar os mesmos nomes para fins de teste
                     ou para recriar um "mundo" específico.
        """
        # Usamos uma instância de Random interna para não afetar o 'random' global
        self.random = random.Random()
        if seed:
            self.random.seed(seed)
        else:
            # Se nenhuma seed for dada, usa o tempo atual, como no seu original
            self.random.seed(time.time())
            
        # Conjunto (set) para armazenar nomes já usados. É muito rápido
        # para verificar se um nome já existe.
        self.nomes_gerados = set()

        # --- Listas de Partes de Nomes Expandidas ---
        
        self.nomes_pessoas = [
            "Lucas", "Mateus", "Joao", "Pedro", "Rafael", "Bruno", "Thiago", "Gustavo", "Felipe", "Daniel",
            "Ana", "Julia", "Marina", "Beatriz", "Camila", "Lara", "Isabela", "Carla", "Luana", "Fernanda",
            # Adicionados:
            "Kael", "Zara", "Milo", "Orion", "Sora", "Elara", "Jax", "Kai", "Ren", "Anya",
            "Ryu", "Kenji", "Akira", "Yuki", "Hana", "Talia", "Niko", "Leon", "Max", "Eva"
        ]
        
        self.apelidos_gamer = [
            "Shadow", "Neo", "Dark", "Cyber", "Ghost", "Iron", "Alpha", "Omega", "Blade", "Storm",
            "Sniper", "Hunter", "Rogue", "Wizard", "Titan", "Viper", "Ninja", "Specter", "Drifter", "Blitz",
            # Adicionados:
            "Reaper", "Slayer", "Warden", "Echo", "Vector", "Havoc", "Fury", "Razor", "Psyche", "Wraith",
            "Jester", "Zero", "Bolt", "Spike", "Hex"
        ]
        
        self.criaturas = [
            "Wolf", "Dragon", "Phoenix", "Kraken", "Griffin", "Hydra", "Leviathan", "Falcon", "Cobra", "Bear",
            # Adicionadas:
            "Wyvern", "Golem", "Behemoth", "Manticore", "Basilisk", "Juggernaut", "Serpent", "Gorgon", "Chimera", "Droid"
        ]
        
        self.sufixos_tech = [
            "X", "Z", "99", "777", "Prime", "Zero", "One", "MK", "RX", "EXE", "VX", "Ultra", "Core", "Void",
            # Adicionados:
            "Matrix", "Data", "Net", "Sys", "Bot", "Log", "Unit"
        ]
        
        # --- Novas Listas para Mais Variedade ---
        
        self.prefixos_tech = [
            "Cyber", "Robo", "Mecha", "Nano", "Bio", "Psy", "Gen", "Xeno", "Proto", "Hyper", "Giga", "Auto"
        ]

        self.titulos_fantasy = [
            "the_Silent", "the_Brave", "of_the_Void", "the_Wanderer", "the_Chosen", "Bloodhand", "Ironfist", 
            "Shadowstep", "Stormcaller", "Fireheart", "Ghostblade"
        ]
        
        self.sufixos_fantasy = [
            "bane", "heart", "soul", "reaver", "shard", "wind", "fury", "shade"
        ]

    def _get_parts(self):
        """Função auxiliar interna para pegar um conjunto de partes aleatórias."""
        return {
            "nome": self.random.choice(self.nomes_pessoas),
            "apelido": self.random.choice(self.apelidos_gamer),
            "criatura": self.random.choice(self.criaturas),
            "sufixo_t": self.random.choice(self.sufixos_tech),
            "prefixo_t": self.random.choice(self.prefixos_tech),
            "titulo_f": self.random.choice(self.titulos_fantasy),
            "sufixo_f": self.random.choice(self.sufixos_fantasy),
            "num": self.random.choice(["7", "9", "X", "01", "007", "66", "88", "42"])
        }

    def gerar_nome(self):
        """
        Gera um nome de agente com base nos formatos e listas definidos.
        
        :param garantir_unicidade: Se True (padrão), garante que o nome nunca 
                                 foi gerado por esta instância, adicionando um 
                                 sufixo numérico (ex: _2, _3) se necessário.
        :return: Uma string com o nome do agente.
        """
        
        parts = self._get_parts()
        
        # --- Lista de Formatos (Receitas de Nomes) ---
        # Usamos 'lambda' para definir mini-funções que montam os nomes.
        # A função vai escolher uma dessas aleatoriamente.
        formatos = [
            # Formatos Tech/Gamer
            lambda p: f"{p['apelido']}{p['criatura']}",                          # Ex: ShadowWolf
            lambda p: f"{p['prefixo_t']}{p['apelido']}",                        # Ex: CyberGhost
            lambda p: f"{p['prefixo_t']}{p['criatura']}",                       # Ex: MechaDragon
            lambda p: f"{p['apelido']}{p['num']}",                              # Ex: Sniper7
            lambda p: f"{p['apelido']}_{p['sufixo_t']}",                        # Ex: Viper_Prime
            lambda p: f"{p['criatura']}{p['sufixo_t']}",                        # Ex: GriffinCore
            lambda p: f"{p['apelido']}{p['criatura']}{p['sufixo_t']}",          # O seu formato original!
            
            # Formatos com Nomes de Pessoas
            lambda p: f"{p['nome']}_{p['apelido']}",                            # Ex: Lucas_Shadow
            lambda p: f"{p['nome']}_{p['titulo_f']}",                           # Ex: Ana_the_Silent
            lambda p: f"{p['nome']}{p['sufixo_f']}",                            # Ex: Kaelbane
            
            # Formatos Simples
            lambda p: f"{p['apelido']}",                                        # Ex: Wraith
            lambda p: f"{p['nome']}"                                            # Ex: Zara
        ]
        
        # 1. Escolhe um formato e gera o nome base
        gerador_formato = self.random.choice(formatos)
        base_name = gerador_formato(parts)
        unique_id = random.randint(1, 999)
        final_name = base_name + f"_{unique_id}"

        return final_name
              