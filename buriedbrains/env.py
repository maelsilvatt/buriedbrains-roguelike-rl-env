# buriedbrains/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import random
import networkx as nx

# Importando todos os módulos que criamos
from . import agent_rules
from . import combat
from . import content_generation
from . import map_generation
from . import reputation

class BuriedBrainsEnv(gym.Env):
    """
    Ambiente principal do BuriedBrains, compatível com a interface Gymnasium.
    """
    def __init__(self, guarantee_enemy: bool = False):
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
            'room_effects': enemy_data['pools']['room_effects'],
        }

        self.max_episode_steps = 10000 # Define um limite, por exemplo, 1000 passos
        self.current_step = 0
        self.max_floors = 5000 # Limite máximo de andares para evitar loops infinitos
        self.max_level = 400
        self.branching_factor = 2 # Número de caminhos possíveis por andar
        self.last_milestone_floor = 0 # Último andar que concedeu recompensa de marco
        self.nodes_per_floor = {} # Dicionário para contar nós por andar

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

        # Gera as salas sucessoras
        for i in range(self.branching_factor):
                            
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
            budget = 100 + (next_floor * 10)
            content = content_generation.generate_room_content(
                self.catalogs, 
                self.pool_costs, 
                budget, 
                next_floor,
                guarantee_enemy=self.guarantee_enemy 
            )
            self.graph.nodes[new_node]['content'] = content

            # --- ADICIONE ESTES LOGS ---
            enemy_log = content.get('enemies', ['Nenhum'])
            item_log = content.get('items', ['Nenhum'])
            print(f"[MAPA] Sala '{new_node}' (Andar {next_floor}) gerada com: "
                f"Inimigos: {enemy_log}, Itens: {item_log}")
            # --- FIM DOS LOGS ---

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_state = agent_rules.create_initial_agent("Player 1")

        # Define as habilidades iniciais do agente
        # self.agent_skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield"]
        self.agent_skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]
        self.agent_state['skills'] = self.agent_skill_names
        self.agent_state['id'] = "Player 1"
        self.reputation_system.add_agent("Player 1")
                
        # --- LÓGICA DE GERAÇÃO DO MAPA CORRIGIDA ---
        self.current_floor = 0
        self.graph = nx.DiGraph()
        self.current_node = "start"
        self.nodes_per_floor = {0: 1} # Reseta o contador (andar 0 tem 1 nó: "start")
        
        # 1. Cria apenas o nó inicial
        self.graph.add_node("start", floor=0)
        
        # 2. Popula o nó inicial (geralmente uma sala vazia)
        start_content = content_generation.generate_room_content(
            self.catalogs, self.pool_costs, 
            budget=0, # Orçamento zero para a sala inicial
            current_floor=0,
            guarantee_enemy=False # Garante que a sala inicial seja segura
        )
        self.graph.nodes["start"]['content'] = start_content

        # 3. Pré-gera APENAS o primeiro nível de escolhas
        self._generate_successors_for_node("start")
        # --- FIM DA LÓGICA DO MAPA ---

        self.combat_state = None
        self.current_step = 0

        # --- ADICIONE ESTES LOGS ---
        print("\n" + "="*50)
        print(f"[RESET] Novo episódio iniciado. Agente Nível {self.agent_state['level']}.")
        print(f"[RESET] Mapa gerado. Sala inicial: '{self.current_node}'.")
        
        successors = list(self.graph.successors(self.current_node))
        if successors:
            print(f"[RESET] Próximas salas disponíveis: {', '.join(successors)}")
        # --- FIM DOS LOGS ---
        
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

        # print(f"  [TURNO] Agente usou: '{action_name}'")
        
        combat.execute_action(agent, [enemy], action_name, self.catalogs)

        # Recompensa por dano causado
        damage_dealt = hp_before_enemy - enemy['hp']
        reward += damage_dealt * 0.6

        # 2. VERIFICA MORTE DO INIMIGO, ADICIONA EXP E FAZ O LEVEL UP
        if combat.check_for_death_and_revive(enemy):
            # PASSO 1: Dar a recompensa pela vitória e adicionar a EXP
            reward += 100 # Recompensa grande por vencer
            self.agent_state['exp'] += enemy.get('exp_yield', 50) # Ganha XP
            
            # PASSO 2: Chamar a lógica de level up IMEDIATAMENTE

            # ----> PRINT DE DIAGNÓSTICO <----
            print(f"[DIAGNÓSTICO] Inimigo '{enemy.get('name')}' derrotado. "
                f"EXP Ganhada: {enemy.get('exp_yield', 50)}. "
                f"EXP Total: {self.agent_state['exp']}. "
                f"EXP Nec: {self.agent_state['exp_to_level_up']}.")
            
            leveled_up = agent_rules.check_for_level_up(self.agent_state) 
            if leveled_up:
                print(f"[DEBUG] Agente subiu para o nível {self.agent_state['level']} durante o combate.")
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

            # print(f"  [TURNO] Inimigo usou: '{enemy_action}'")
            
            combat.execute_action(enemy, [agent], enemy_action, self.catalogs)

            # Penalidade por dano sofrido
            damage_taken = hp_before_agent - agent['hp']
            reward -= damage_taken * 0.5

            # 4. VERIFICA MORTE DO AGENTE
            if combat.check_for_death_and_revive(agent):
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

    def step(self, action: int):
            reward = 0
            terminated = False
            info = {}

            self.current_step += 1 # Incrementa o contador a cada passo
            truncated = self.current_step >= self.max_episode_steps

            # Penalidade de tempo: o agente perde um pouco a cada passo.
            reward -= 0.5

            # Verifica se o agente está no último andar E não tem para onde ir
            is_on_last_floor = (self.current_floor == self.max_floors)
            has_no_successors = not list(self.graph.successors(self.current_node))

            if is_on_last_floor and has_no_successors and not self.combat_state:
                terminated = True
                reward += 400 # Recompensa final por chegar ao fim
                print(f"[EPISÓDIO] FIM: Agente VENCEU! (Chegou ao fim do labirinto no Andar {self.current_floor})")            

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
                # Penalidade de morte aumentada para ser mais significativa
                reward = -300 
                print(f"[EPISÓDIO] FIM: Agente MORREU no Andar {self.current_floor}.") # LOG
                
            # Adiciona uma recompensa grande por vencer o jogo E TERMINA IMEDIATAMENTE
            if self.current_floor > self.max_floors:
                terminated = True
                reward += 400 # Pode adicionar a recompensa final aqui
                print(f"[EPISÓDIO] FIM: Agente VENCEU! (Chegou ao andar {self.current_floor})")
                # Prepara o info final AQUI e retorna
                info['final_status'] = {
                    'level': self.agent_state['level'],
                    'hp': self.agent_state['hp'],
                    'floor': self.current_floor,
                    'win': self.agent_state['hp'] > 0 and not terminated
                }
                observation = self._get_observation()
                return observation, reward, terminated, False, info # Retorna imediatamente

            # Recompensa a cada 10 andares concluídos
            if self.current_floor % 10 == 0 and self.current_floor > self.last_milestone_floor:
                reward += 400
                print(f"[MARCO] Agente alcançou o Andar {self.current_floor}! Bônus de +400.")
                self.last_milestone_floor = self.current_floor # Atualiza o último marco alcançado

            if terminated or truncated:
                if truncated and not terminated:
                    print(f"[EPISÓDIO] FIM: TEMPO ESGOTADO (Truncated) no Andar {self.current_floor}.") # LOG
                info['final_status'] = {
                    'level': self.agent_state['level'],
                    'hp': self.agent_state['hp'],
                    'floor': self.current_floor,
                    'win': self.agent_state['hp'] > 0 and not terminated
                }

            observation = self._get_observation()        

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

                print(f"[AÇÃO] Agente na sala '{self.current_node}'. Tentando Ação de Movimento {action_index}.") # LOG

                if len(successors) > action_index:
                    # --- LÓGICA DE PODA E GERAÇÃO ---
                    # 1. Agente escolhe um caminho
                    chosen_node = successors[action_index]
                    print(f"[AÇÃO] Movimento VÁLIDO para '{chosen_node}'.") # LOG
                    
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
                        print(f"[AÇÃO] >>> COMBATE INICIADO com {enemy_names[0]} <<<") # LOG
                        self._start_combat(enemy_names[0])
                        reward += 10 # Recompensa bônus por encontrar um desafio
                else:
                    # Movimento inválido (tentou ir para uma sala que não existe)
                    print(f"[AÇÃO] Movimento INVÁLIDO. (Sem sala no índice {action_index})") # LOG
                    reward = -5
            
            # Ação 7: (Antiga Ação 8) Equipar Item de Poder (Weapon/Armor)
            elif action == 7: 
                print(f"[AÇÃO] Agente tentou Ação 7 (Equipar Item).") # LOG
                room_items = self.graph.nodes[self.current_node].get('content', {}).setdefault('items', [])
                equip_on_floor = next((item for item in room_items if self.catalogs['equipment'].get(item, {}).get('type') in ['Weapon', 'Armor']), None)
                
                if equip_on_floor:
                    details = self.catalogs['equipment'][equip_on_floor]
                    item_type = details['type']
                    self.agent_state['equipment'][item_type] = equip_on_floor
                    room_items.remove(equip_on_floor)
                    print(f"[AÇÃO] Agente tentou Ação 7 (Equipar Item).") # LOG
                    reward = 15 
                else:
                    print(f"[AÇÃO] Agente tentou equipar, mas não havia itens (Weapon/Armor) no chão.") # LOG            
                    reward = -1
            
            # Ações de Combate (0 a 4) ou Ação Inválida (antiga 7)
            else:
                print(f"[AÇÃO] Agente usou Ação {action} (Combate/Inválida) fora de combate.") # LOG
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

        # --- ADICIONE ESTES LOGS ---
        print(f"[COMBATE] Agente (Nível {self.agent_state['level']}, HP {agent_combatant['hp']}) "
            f"vs. {enemy_combatant['name']} (Nível {enemy_combatant['level']}, HP {enemy_combatant['hp']})")
        # --- FIM DOS LOGS ---