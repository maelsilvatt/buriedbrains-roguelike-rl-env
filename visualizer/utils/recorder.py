import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import json
import numpy as np
from datetime import datetime

class BuriedBrainsRecorder:
    def __init__(self, vec_env):
        """
        vec_env: Instância do SharedPolicyVecEnv
        """
        self.vec_env = vec_env
        self.base_env = vec_env.env  # O BuriedBrainsEnv está aqui dentro
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
            if not state: continue # Agente pode não ter sido inicializado

            # Pega logs recentes (últimos 2 para não poluir a tela)
            full_log = self.base_env.current_episode_logs.get(agent_id, [])
            last_logs = full_log[-2:] if full_log else []

            # Pega cooldowns atuais (Para mostrar ícones de habilidades)
            current_cooldowns = state.get('cooldowns', {})

            # Determina o modo da cena
            in_combat = self.base_env.combat_states.get(agent_id) is not None
            in_arena = agent_id in self.base_env.arena_instances
            
            scene_mode = "EXPLORATION"
            if in_combat: scene_mode = "COMBAT_PVE"
            elif in_arena: scene_mode = "ARENA" # Pode ser Sanctum ou PvP

            # Dados de Localização e Vizinhos (Para desenhar o Grid)
            current_node = self.base_env.current_nodes.get(agent_id)
            current_floor = self.base_env.current_floors.get(agent_id, 0)
            neighbors_data = self._get_neighbors_view(agent_id, in_arena, current_node)

            # Coleta dados extras de combate se houver
            enemy_stats = {}
            if in_combat:
                combat_data = self.base_env.combat_states[agent_id]['enemy']
                enemy_stats = {
                    "name": combat_data['name'],
                    "hp": combat_data['hp'],
                    "max_hp": combat_data.get('max_hp', combat_data['hp']) # Fallback se não tiver max_hp
                }

            # --- PEGAR EFEITO DA SALA ATUAL ---
            current_node = self.base_env.current_nodes.get(agent_id)
            current_effect = "None"
            
            # Descobre o grafo correto (Arena ou P-Zone)
            graph = None
            if in_arena:
                graph = self.base_env.arena_instances.get(agent_id)
            else:
                graph = self.base_env.graphs.get(agent_id)
            
            # Lê o conteúdo
            if graph and graph.has_node(current_node):
                content = graph.nodes[current_node].get('content', {})
                effects = content.get('room_effects', [])
                if effects:
                    current_effect = effects[0] # Pega o primeiro efeito            

            # Monta o objeto do agente para este frame
            frame_snapshot["agents"][agent_id] = {
                # Status Básicos
                "name": self.base_env.agent_names.get(agent_id, "Unknown"),
                "hp": float(state.get('hp', 0)),
                "max_hp": float(state.get('max_hp', 100)),
                "level": int(state.get('level', 1)),
                "exp_percent": int((state.get('exp', 0) / state.get('exp_to_level_up', 100)) * 100),
                
                "cooldowns": state.get('cooldowns', {}).copy(),                
                "combat_data": enemy_stats if in_combat else None,
                
                # Karma (Crucial para o Disco de Poincaré)
                "karma": {
                    "real": float(state['karma']['real']),
                    "imag": float(state['karma']['imag'])
                },

                # Equipamento (Para mostrar ícones no inventário)
                "equipment": list(state.get('equipment', {}).values()),

                # Ações e Recompensas
                "cooldowns": current_cooldowns.copy(),

                # Contexto Visual
                "location_node": current_node,
                "floor": current_floor,
                "scene_mode": scene_mode,
                "neighbors": neighbors_data,
                "logs": [l.strip() for l in last_logs],
                "current_effect": current_effect,
                
                # Combate
                "combat_enemy": self.base_env.combat_states[agent_id]['enemy']['name'] if in_combat else None
            }

        self.episode_data.append(frame_snapshot)

    def _get_neighbors_view(self, agent_id, in_arena, current_node):
        """Extrai o que tem nas salas vizinhas (Inimigos, Itens, Eventos)."""
        neighbors_data = []
        
        graph = None
        if in_arena:
            graph = self.base_env.arena_instances.get(agent_id)
        else:
            graph = self.base_env.graphs.get(agent_id)
        
        if graph and graph.has_node(current_node):
            if in_arena:
                succ = list(graph.neighbors(current_node))
                succ.sort() # Ordena para manter consistência visual (Norte, Sul, Leste, Oeste)
            else:
                succ = list(graph.successors(current_node))
            
            for i, n_node in enumerate(succ):
                content = graph.nodes[n_node].get('content', {})
                
                # Simplifica o conteúdo para o JSON
                summary = {
                    "id": n_node,
                    "has_enemy": len(content.get('enemies', [])) > 0,
                    "enemy_type": content.get('enemies', [''])[0] if content.get('enemies') else None,
                    "has_treasure": 'Treasure' in content.get('events', []) or len(content.get('items', [])) > 0,
                    "is_exit": graph.nodes[n_node].get('is_exit', False), # Para saber se é porta trancada
                    "effect": content.get('room_effects', [None])[0]
                }
                neighbors_data.append(summary)
        
        return neighbors_data

    def save_to_json(self, filename="replay.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.episode_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Replay salvo com sucesso: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")