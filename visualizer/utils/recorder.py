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
        frame_snapshot = {
            "turn": step_count,
            "timestamp": datetime.now().isoformat(),
            "agents": {}
        }

        for agent_id in self.base_env.agent_ids:
            state = self.base_env.agent_states.get(agent_id, {})
            if not state: continue 

            # --- 1. CAPTURA DE LOGS (Corrigido: Adicionado aqui) ---
            full_log = self.base_env.current_episode_logs.get(agent_id, [])
            last_logs = full_log[-2:] if full_log else []

            # --- 2. Captura o contexto do estado
            # A Arena tem prioridade sobre PvE. Se ele tem uma instância de arena, ele está lá.            
            in_pvp = agent_id in self.base_env.pvp_sessions
            in_queue = agent_id in self.base_env.matchmaking_queue
            in_arena = agent_id in self.base_env.arena_instances
            
            # Só considera PvE se NÃO estiver na arena
            combat_state = self.base_env.combat_states.get(agent_id)
            in_pve = (combat_state is not None) and (not in_arena)

            scene_mode = "EXPLORATION"
            
            if in_pvp: 
                scene_mode = "COMBAT_PVP"
            elif in_queue:
                scene_mode = "WAITING"
            elif in_arena: 
                # Se está na arena e não está em PvP nem Fila, é exploração da arena (Grid)
                scene_mode = "ARENA"
            elif in_pve: 
                scene_mode = "COMBAT_PVE"

            # --- QUEM É O OPONENTE? ---
            opponent_id = None
            if in_pvp:
                # Se está em combate, pega da sessão
                session = self.base_env.pvp_sessions.get(agent_id)
                if session:
                    opponent_id = session['a2_id'] if session['a1_id'] == agent_id else session['a1_id']
            elif in_arena:
                # Se está andando na arena, pega do matchmaking
                opponent_id = self.base_env.active_matches.get(agent_id)

            # --- 3. DADOS DO MAPA DA ARENA ---
            arena_edges = []
            if scene_mode == "ARENA" or scene_mode == "COMBAT_PVP":
                arena_graph = self.base_env.arena_instances.get(agent_id)
                if arena_graph:
                    for u, v in arena_graph.edges():
                        # Extrai ID numérico (k_20_5 -> 5)
                        try:
                            u_idx = int(u.split('_')[-1])
                            v_idx = int(v.split('_')[-1])
                            arena_edges.append([u_idx, v_idx])
                        except: pass

            # --- 4. CONTEXTO E VIZINHOS ---
            current_node = self.base_env.current_nodes.get(agent_id)
            current_floor = self.base_env.current_floors.get(agent_id, 0)
            neighbors_data, current_effect, room_items = self._get_context_view(agent_id, in_arena, current_node)

            # --- 5. DADOS DE COMBATE ---
            combat_info = None
            if in_pve:
                c_data = self.base_env.combat_states[agent_id]['enemy']
                combat_info = {"name": c_data['name'], "hp": float(c_data['hp']), "max_hp": float(c_data.get('max_hp', c_data['hp']))}
            elif in_pvp:
                session = self.base_env.pvp_sessions[agent_id]
                opp_id = session['a2_id'] if session['a1_id'] == agent_id else session['a1_id']
                opp_state = self.base_env.agent_states.get(opp_id, {})
                combat_info = {
                    "name": self.base_env.agent_names.get(opp_id, "Oponente"),
                    "hp": float(opp_state.get('hp', 0)),
                    "max_hp": float(opp_state.get('max_hp', 100))
                }

            # --- 6. AÇÃO E OBSERVAÇÃO BRUTA ---
            action_taken = actions_dict.get(agent_id) if actions_dict else None
            if hasattr(action_taken, "item"): action_taken = action_taken.item()
            
            skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait", "Equip", "Social", "Move N", "Move S", "Move E", "Move W"]
            action_name = skill_names[action_taken] if action_taken is not None and 0 <= action_taken < len(skill_names) else None

            # Captura Raw Obs para o Brain View
            raw_obs = self.base_env._get_observation(agent_id)
            raw_obs_list = raw_obs.tolist() if hasattr(raw_obs, 'tolist') else list(raw_obs)

            # --- 7. MONTAGEM FINAL ---
            frame_snapshot["agents"][agent_id] = {
                "name": self.base_env.agent_names.get(agent_id, "Unknown"),
                "hp": float(state.get('hp', 0)),
                "max_hp": float(state.get('max_hp', 100)),
                "level": int(state.get('level', 1)),
                "exp_percent": int((state.get('exp', 0) / max(1, state.get('exp_to_level_up', 100))) * 100),
                "karma": {"real": float(state['karma']['real']), "imag": float(state['karma']['imag'])},
                "equipment": list(state.get('equipment', {}).values()),
                "cooldowns": state.get('cooldowns', {}).copy(),
                "action_taken": action_name,
                "raw_obs": raw_obs_list,
                "opponent_id": opponent_id,            
                "location_node": current_node,
                "floor": current_floor,
                "scene_mode": scene_mode,
                "neighbors": neighbors_data,
                "current_effect": current_effect,
                "room_items": room_items,
                "arena_edges": arena_edges,                
                "combat_data": combat_info,
                "logs": [l.strip() for l in last_logs] 
            }

        self.episode_data.append(frame_snapshot)

    def _get_context_view(self, agent_id, in_arena, current_node):
        neighbors_data = []
        current_effect = "None"
        room_items = [] # Lista de nomes de itens no chão
        
        graph = None
        if in_arena:
            graph = self.base_env.arena_instances.get(agent_id)
        else:
            graph = self.base_env.graphs.get(agent_id)
        
        if graph and graph.has_node(current_node):
            node_data = graph.nodes[current_node]
            content = node_data.get('content', {})
            
            # Efeitos e Itens da sala ATUAL
            effects = content.get('room_effects', [])
            if effects: current_effect = effects[0]
            room_items = content.get('items', [])

            # Vizinhos
            if in_arena:
                succ = list(graph.neighbors(current_node))
                succ.sort()
            else:
                succ = list(graph.successors(current_node))
            
            for n_node in succ:
                n_content = graph.nodes[n_node].get('content', {})
                neighbors_data.append({
                    "id": n_node,
                    "has_enemy": len(n_content.get('enemies', [])) > 0,
                    "enemy_type": n_content.get('enemies', [''])[0] if n_content.get('enemies') else None,
                    "has_treasure": 'Treasure' in n_content.get('events', []) or len(n_content.get('items', [])) > 0,
                    "is_exit": graph.nodes[n_node].get('is_exit', False),
                    "effect": n_content.get('room_effects', [None])[0]
                })
        
        return neighbors_data, current_effect, room_items

    def save_to_json(self, filename="replay.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.episode_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Replay salvo com sucesso: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")