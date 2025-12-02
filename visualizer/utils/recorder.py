import json
import numpy as np
from datetime import datetime

class BuriedBrainsRecorder:
    def __init__(self, vec_env):
        """
        vec_env: Instância do SharedPolicyVecEnv
        """
        self.vec_env = vec_env
        self.base_env = vec_env.env  # O BuriedBrainsEnv real
        self.episode_data = []

    def record_step(self, step_count, actions_dict=None, rewards_dict=None):
        """
        Captura um snapshot do estado atual de todos os agentes.
        """
        frame_snapshot = {
            "turn": step_count,
            "timestamp": datetime.now().isoformat(),
            "agents": {}
        }

        # Itera sobre os agentes reais no ambiente base
        for agent_id in self.base_env.agent_ids:
            # Pega o estado mestre (God Mode)
            state = self.base_env.agent_states.get(agent_id, {})
            if not state: continue 

            # Pega logs recentes
            full_log = self.base_env.current_episode_logs.get(agent_id, [])
            last_logs = full_log[-2:] if full_log else []

            # Determina o modo da cena
            in_combat = self.base_env.combat_states.get(agent_id) is not None
            in_arena = agent_id in self.base_env.arena_instances
            
            scene_mode = "EXPLORATION"
            if in_combat: scene_mode = "COMBAT_PVE"
            elif in_arena: scene_mode = "ARENA"

            # Dados de Localização
            current_node = self.base_env.current_nodes.get(agent_id)
            current_floor = self.base_env.current_floors.get(agent_id, 0)
            
            # --- Captura Vizinhos e Efeito Atual ---
            neighbors_data, current_effect = self._get_context_view(agent_id, in_arena, current_node)

            # --- Captura Dados de Combate (HP Inimigo) ---
            enemy_stats = {}
            if in_combat:
                combat_data = self.base_env.combat_states[agent_id]['enemy']
                enemy_stats = {
                    "name": combat_data['name'],
                    "hp": float(combat_data['hp']),
                    "max_hp": float(combat_data.get('max_hp', combat_data['hp']))
                }

            # --- Ação Tomada (Para o highlight da skill) ---
            # Tenta converter numpy.int64 para int nativo do Python para o JSON não quebrar
            action_taken = actions_dict.get(agent_id) if actions_dict else None
            if hasattr(action_taken, "item"): 
                action_taken = action_taken.item()
            
            # Mapeamento de índice para nome da skill (opcional, ajuda no debug visual)
            skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]
            action_name = None
            if action_taken is not None and 0 <= action_taken < len(skill_names):
                action_name = skill_names[action_taken]

            # Monta o objeto do agente
            frame_snapshot["agents"][agent_id] = {
                "name": self.base_env.agent_names.get(agent_id, "Unknown"),
                "hp": float(state.get('hp', 0)),
                "max_hp": float(state.get('max_hp', 100)),
                "level": int(state.get('level', 1)),
                # Evita divisão por zero no XP
                "exp_percent": int((state.get('exp', 0) / max(1, state.get('exp_to_level_up', 100))) * 100),
                
                # Campos Essenciais para o Visualizer
                "karma": {
                    "real": float(state['karma']['real']),
                    "imag": float(state['karma']['imag'])
                },
                "equipment": list(state.get('equipment', {}).values()),
                "cooldowns": state.get('cooldowns', {}).copy(),
                "action_taken": action_name, # Manda o nome da skill ("Heavy Blow")

                # Contexto
                "location_node": current_node,
                "floor": current_floor,
                "scene_mode": scene_mode,
                "neighbors": neighbors_data,
                "current_effect": current_effect,
                "combat_data": enemy_stats if in_combat else None,
                "logs": [l.strip() for l in last_logs]
            }

        self.episode_data.append(frame_snapshot)

    def _get_context_view(self, agent_id, in_arena, current_node):
        """Retorna (vizinhos, efeito_da_sala_atual)"""
        neighbors_data = []
        current_effect = "None"
        
        graph = None
        if in_arena:
            graph = self.base_env.arena_instances.get(agent_id)
        else:
            graph = self.base_env.graphs.get(agent_id)
        
        if graph and graph.has_node(current_node):
            # 1. Pega efeito da sala atual
            curr_content = graph.nodes[current_node].get('content', {})
            effects = curr_content.get('room_effects', [])
            if effects: current_effect = effects[0]

            # 2. Pega vizinhos
            if in_arena:
                succ = list(graph.neighbors(current_node))
                succ.sort()
            else:
                succ = list(graph.successors(current_node))
            
            for n_node in succ:
                content = graph.nodes[n_node].get('content', {})
                summary = {
                    "id": n_node,
                    "has_enemy": len(content.get('enemies', [])) > 0,
                    "enemy_type": content.get('enemies', [''])[0] if content.get('enemies') else None,
                    "has_treasure": 'Treasure' in content.get('events', []) or len(content.get('items', [])) > 0,
                    "is_exit": graph.nodes[n_node].get('is_exit', False),
                    "effect": content.get('room_effects', [None])[0]
                }
                neighbors_data.append(summary)
        
        return neighbors_data, current_effect

    def save_to_json(self, filename="replay.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.episode_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Replay salvo com sucesso: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")