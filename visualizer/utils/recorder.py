import json
import numpy as np
from datetime import datetime

class BuriedBrainsRecorder:
    def __init__(self, vec_env, save_path="replay_data.js"):
        """
        vec_env: Instância do SharedPolicyVecEnv
        """
        self.vec_env = vec_env
        self.base_env = vec_env.env  
        self.save_path = save_path
        self.frames = []

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

            # Karma
            k = state.get('karma', {'real': 0.0, 'imag': 0.0})
            if isinstance(k, dict):
                k_real = k.get('real', 0.0)
                k_imag = k.get('imag', 0.0)
            else:
                k_real = k.real
                k_imag = k.imag
            
            # Captura o contexto de cena            
            in_pvp_session = agent_id in self.base_env.pvp_sessions
            in_queue = agent_id in self.base_env.matchmaking_queue
            in_arena = agent_id in self.base_env.arena_instances
            
            # Só considera PvE se tiver estado de combate E NÃO estiver na Arena (evita falsos positivos)
            combat_state = self.base_env.combat_states.get(agent_id)
            in_pve = (combat_state is not None)

            scene_mode = "EXPLORATION"
            
            if in_pvp_session: 
                scene_mode = "COMBAT_PVP"  # PvP tem prioridade máxima na Arena
            elif in_queue:
                scene_mode = "WAITING"    
            elif in_arena: 
                scene_mode = "ARENA"      
            elif in_pve: 
                scene_mode = "COMBAT_PVE" 

            # Dados de Localização
            current_node = self.base_env.current_nodes.get(agent_id, "Unknown")
            current_floor = self.base_env.current_floors.get(agent_id, 0)
            
            # Captura Vizinhos, Efeito Atual e Itens na Sala
            neighbors_data, current_effect, room_items = self._get_context_view(agent_id, in_arena, current_node)

            # Dados de combate
            enemy_stats = None
            
            if scene_mode == "COMBAT_PVE" and combat_state:
                combat_data = combat_state['enemy']
                enemy_stats = {
                    "name": combat_data['name'],
                    "hp": float(combat_data['hp']),
                    "max_hp": float(combat_data.get('max_hp', combat_data['hp']))
                }
            elif scene_mode == "COMBAT_PVP":
                # Recupera dados do oponente PvP
                session = self.base_env.pvp_sessions.get(agent_id)
                if session:
                    opp_id = session['a2_id'] if session['a1_id'] == agent_id else session['a1_id']
                    opp_state = self.base_env.agent_states.get(opp_id, {})
                    enemy_stats = {
                        "name": self.base_env.agent_names.get(opp_id, "Oponente"),
                        "hp": float(opp_state.get('hp', 0)),
                        "max_hp": float(opp_state.get('max_hp', 100))
                    }

            # Ação Tomada
            action_taken = actions_dict.get(agent_id) if actions_dict else None
            if hasattr(action_taken, "item"): 
                action_taken = action_taken.item()
            
            skill_names = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait", "Equip", "Drop", "North", "South", "East", "West"]
            action_name = "Wait"
            if action_taken is not None and 0 <= action_taken < len(skill_names):
                action_name = skill_names[action_taken]
            elif action_taken is not None:
                action_name = f"Action {action_taken}"

            # Raw Obs
            raw_obs_array = self.base_env._get_observation(agent_id)
            if hasattr(raw_obs_array, 'tolist'):
                raw_obs_list = raw_obs_array.tolist()
            else:
                raw_obs_list = list(raw_obs_array)

            # Topologia da Arena (Se estiver nela ou em PvP nela)
            arena_edges = []
            if in_arena: # Envia o mapa sempre que estiver na arena, mesmo lutando
                current_graph = self.base_env.arena_instances.get(agent_id)
                if current_graph:
                    for u, v in current_graph.edges():
                        try:
                            u_idx = int(u.split('_')[-1])
                            v_idx = int(v.split('_')[-1])
                            arena_edges.append([u_idx, v_idx])
                        except: pass
            
            # Pega o oponente
            opponent_id = None
            if in_pvp_session:
                session = self.base_env.pvp_sessions.get(agent_id)
                if session:
                    opponent_id = session['a2_id'] if session['a1_id'] == agent_id else session['a1_id']
            elif in_arena:
                opponent_id = self.base_env.active_matches.get(agent_id)

            # Monta o objeto do agente
            frame_snapshot["agents"][agent_id] = {
                "name": self.base_env.agent_names.get(agent_id, "Unknown"),
                "hp": float(state.get('hp', 0)),
                "max_hp": float(state.get('max_hp', 100)),
                "level": int(state.get('level', 1)),
                "raw_obs": raw_obs_list,
                "exp_percent": int((state.get('exp', 0) / max(1, state.get('exp_to_level_up', 100))) * 100),
                
                "karma": {
                    "real": float(k_real),
                    "imag": float(k_imag)
                },
                "equipment": list(state.get('equipment', {}).values()),
                "cooldowns": state.get('cooldowns', {}).copy(),
                "action_taken": action_name,

                "location_node": current_node,
                "floor": current_floor,
                "scene_mode": scene_mode,
                
                "arena_edges": arena_edges, 
                "opponent_id": opponent_id, 
                
                "neighbors": neighbors_data,
                "current_effect": current_effect,
                "room_items": room_items,
                "combat_data": enemy_stats,
                
                "social": {
                    "offered_peace": self.base_env.arena_interaction_state[agent_id]['offered_peace'],
                    "just_dropped": self.base_env.social_flags[agent_id].get('just_dropped', False),
                    "skipped_attack": self.base_env.social_flags[agent_id].get('skipped_attack', False)
                },

                "logs": [l.strip() for l in last_logs]
            }

        self.frames.append(frame_snapshot)

    def _get_context_view(self, agent_id, in_arena, current_node):
        """Retorna (vizinhos, efeito_da_sala_atual, itens_da_sala_atual)"""
        neighbors_data = []
        current_effect = "None"
        room_items = []
        
        graph = None
        if in_arena:
            graph = self.base_env.arena_instances.get(agent_id)
        else:
            graph = self.base_env.graphs.get(agent_id)
        
        if graph and graph.has_node(current_node):
            # Dados da Sala Atual
            curr_content = graph.nodes[current_node].get('content', {})
            
            # Verificação de Efeitos
            effs = curr_content.get('room_effects', [])
            if effs and len(effs) > 0:
                current_effect = effs[0]
            
            room_items = curr_content.get('items', [])

            # Pega vizinhos
            succ = []
            if in_arena:
                try: 
                    succ = list(graph.neighbors(current_node))
                    succ.sort()
                except: succ = []
            else:
                try:
                    succ = list(graph.successors(current_node))
                except: succ = []
            
            for n_node in succ:
                n_content = graph.nodes[n_node].get('content', {})
                
                # Verificação de Efeitos do Vizinho                
                n_effs = n_content.get('room_effects', [])
                n_effect_val = n_effs[0] if n_effs and len(n_effs) > 0 else None

                summary = {
                    "id": n_node,
                    "has_enemy": len(n_content.get('enemies', [])) > 0,
                    "enemy_type": n_content.get('enemies', [''])[0] if n_content.get('enemies') else None,
                    "has_treasure": 'Treasure' in n_content.get('events', []) or len(n_content.get('items', [])) > 0,
                    "is_exit": graph.nodes[n_node].get('is_exit', False),
                    "effect": n_effect_val 
                }
                neighbors_data.append(summary)
        
        return neighbors_data, current_effect, room_items

    def save_to_json(self, filename=None):
        """
        Salva o replay. 
        """
        target_file = filename if filename else self.save_path
        
        print(f"Salvando replay com {len(self.frames)} frames em {target_file}...")
        
        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
            
        try:
            with open(target_file, "w", encoding='utf-8') as f:
                if target_file.endswith('.js'):
                    f.write("const REPLAY_DATA = ") 
                    json.dump(self.frames, f, default=np_encoder) 
                    f.write(";") 
                else:
                    json.dump(self.frames, f, default=np_encoder, indent=2)
                    
            print(f"✅ Replay salvo com sucesso: {target_file}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar arquivo: {e}")