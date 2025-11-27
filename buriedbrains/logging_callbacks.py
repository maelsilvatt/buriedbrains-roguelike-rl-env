import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LoggingCallback(BaseCallback):
    """
    Callback avançado para registrar métricas detalhadas do BuriedBrains (SAE/MAE)
    e salvar as "melhores histórias" (Hall da Fama).
    """
    def __init__(self, log_interval: int = 10, verbose: int = 1, top_n: int = 10):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.top_n = top_n 

        # --- 1. Métricas de Performance (O Básico) ---
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        
        # --- 2. Métricas de Progressão ---
        self.episode_levels = []
        self.episode_floors = []
        self.episode_total_exp = [] 
        
        # --- 3. Métricas de Estilo de Jogo (Novas!) ---
        self.episode_damage_dealt = []      # Agressividade
        self.episode_equipment_swaps = []   # Inteligência Estratégica
        self.episode_exploration_steps = [] # Passos gastos explorando vs lutando
        
        # --- 4. Métricas de Combate/Erro ---
        self.episode_enemies_defeated = []
        self.episode_invalid_actions = []
        self.episode_total_actions = []
        
        # --- 5. Métricas Sociais (Karma - Preparação Fase 2) ---
        self.episode_final_karma_real = []
        
        # --- Hall da Fama ---
        self.hall_of_fame_level = [] 
        self.hall_of_fame_floor = [] 
        self.hall_of_fame_enemies = [] 
        
        self.episode_count = 0
        self.max_floor_ever = 0

    def _update_hall_of_fame(self, story: dict, hall_of_fame: list, metric_key: str):
        """Mantém a lista de Hall da Fama atualizada e ordenada."""
        new_score = story.get(metric_key, 0)
        
        if len(hall_of_fame) < self.top_n:
            hall_of_fame.append(story)
            hall_of_fame.sort(key=lambda s: s[metric_key], reverse=True)
            return 
        
        worst_score = hall_of_fame[-1].get(metric_key, 0)
        if new_score > worst_score:
            hall_of_fame.pop() 
            hall_of_fame.append(story) 
            hall_of_fame.sort(key=lambda s: s[metric_key], reverse=True) 

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        
        for i, done in enumerate(dones):
            if not done:
                continue
            
            self.episode_count += 1
            info = self.locals["infos"][i]
            final_info = info.get("final_info", info)
            
            if not isinstance(final_info, dict): continue 
            final_status = final_info.get("final_status")
            if not final_status: continue 
            
            # --- Coleta de Dados (Lendo do final_status) ---
            self.episode_rewards.append(self.locals["rewards"][i])
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))
            self.episode_floors.append(final_status.get("floor", 0))
            self.episode_lengths.append(final_status.get("steps", 0))
            self.episode_total_exp.append(final_status.get("exp", 0))
            
            self.episode_enemies_defeated.append(final_status.get("enemies_defeated", 0))
            self.episode_invalid_actions.append(final_status.get("invalid_actions", 0))
            self.episode_total_actions.append(final_status.get("steps", 1)) # Total steps = total actions                    
            self.episode_damage_dealt.append(final_status.get("damage_dealt", 0))
            self.episode_equipment_swaps.append(final_status.get("equipment_swaps", 0))

            pve_durs = final_status.get('pve_durations', [])
            pvp_durs = final_status.get('pvp_durations', [])
            
            avg_pve_dur = np.mean(pve_durs) if pve_durs else 0
            avg_pvp_dur = np.mean(pvp_durs) if pvp_durs else 0
            
            # Salva em listas temporárias do callback para fazer média do intervalo
            if not hasattr(self, 'episode_avg_pve_duration'): self.episode_avg_pve_duration = []
            if not hasattr(self, 'episode_avg_pvp_duration'): self.episode_avg_pvp_duration = []
            
            self.episode_avg_pve_duration.append(avg_pve_dur)
            self.episode_avg_pvp_duration.append(avg_pvp_dur)
            
            # --- Métricas Sociais ---
            self.episode_arena_encounters.append(final_status.get('arena_encounters', 0))
            self.episode_pvp_combats.append(final_status.get('pvp_combats', 0))
            self.episode_bargains.append(final_status.get('bargains_succeeded', 0))
            self.episode_cowardice_kills.append(final_status.get('cowardice_kills', 0))
            
            # Karma
            karma = final_status.get("karma", {'real': 0.0})
            self.episode_final_karma_real.append(karma.get('real', 0.0))

            current_floor = final_status.get("floor", 0)
            self.max_floor_ever = max(self.max_floor_ever, current_floor)

            # --- Montagem da História para o Hall da Fama ---
            agent_name = final_status.get('agent_name', 'Agente_Desconhecido')
            full_log = final_status.get('full_log', ['Log não capturado.'])
            equipment = final_status.get('equipment', {})
            death_cause = final_status.get('death_cause', 'Desconhecida') # Vamos adicionar isso!
            
            story = {
                'agent_name': agent_name,
                'level': final_status.get('level', 0),
                'floor': current_floor,
                'enemies_defeated': final_status.get('enemies_defeated', 0),
                'damage_dealt': final_status.get("damage_dealt", 0),
                'death_cause': death_cause,
                'equipment': equipment,
                'log_content': full_log 
            }
            
            self._update_hall_of_fame(story, self.hall_of_fame_level, 'level')
            self._update_hall_of_fame(story, self.hall_of_fame_floor, 'floor')
            self._update_hall_of_fame(story, self.hall_of_fame_enemies, 'enemies_defeated')

            # --- Logging no TensorBoard ---
            if self.episode_count % self.log_interval == 0:
                # Cálculos de Médias
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                win_rate = np.mean(self.episode_wins) if self.episode_wins else 0
                avg_level = np.mean(self.episode_levels) if self.episode_levels else 0
                avg_floor = np.mean(self.episode_floors) if self.episode_floors else 0
                avg_damage = np.mean(self.episode_damage_dealt) if self.episode_damage_dealt else 0
                avg_swaps = np.mean(self.episode_equipment_swaps) if self.episode_equipment_swaps else 0
                
                total_invalid = sum(self.episode_invalid_actions)
                total_actions = sum(self.episode_total_actions)
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0

                # Calcula médias de duração de combate
                avg_pve_dur_interval = np.mean(self.episode_avg_pve_duration) if self.episode_avg_pve_duration else 0
                avg_pvp_dur_interval = np.mean(self.episode_avg_pvp_duration) if self.episode_avg_pvp_duration else 0

                # Registro
                self.logger.record("rollout/ep_rew_mean", mean_reward)
                self.logger.record("custom/win_rate", win_rate)
                self.logger.record("custom/avg_level", avg_level)
                self.logger.record("custom/avg_floor_reached", avg_floor)
                self.logger.record("custom/avg_damage_dealt", avg_damage)     
                self.logger.record("custom/avg_equipment_swaps", avg_swaps)   
                self.logger.record("custom/rate_invalid_actions", invalid_rate)
                self.logger.record("combat/avg_pve_duration", avg_pve_dur_interval)
                self.logger.record("combat/avg_pvp_duration", avg_pvp_dur_interval)            
                self.logger.record("social/total_arena_encounters", np.sum(self.episode_arena_encounters))
                self.logger.record("social/total_pvp_combats", np.sum(self.episode_pvp_combats))
                self.logger.record("social/total_bargains", np.sum(self.episode_bargains))
                
                self.logger.dump(step=self.num_timesteps)

                # Limpeza das listas
                self.episode_rewards.clear()
                self.episode_wins.clear()
                self.episode_levels.clear()
                self.episode_floors.clear()
                self.episode_lengths.clear()
                self.episode_total_exp.clear()
                self.episode_damage_dealt.clear()
                self.episode_equipment_swaps.clear()
                self.episode_enemies_defeated.clear()
                self.episode_invalid_actions.clear()
                self.episode_total_actions.clear()
                self.episode_final_karma_real.clear()
                self.episode_avg_pve_duration.clear()
                self.episode_avg_pvp_duration.clear()

        return True

    def save_hall_of_fame(self, save_dir: str):
        """Salva as histórias com detalhes extras (Equipamento, Causa da Morte)."""
        if self.verbose > 0:
            print(f"\nSalvando Hall da Fama em {save_dir}...")
        
        def _save_list(hall_list: list, sub_folder: str, metric_key: str):
            path = os.path.join(save_dir, sub_folder)
            os.makedirs(path, exist_ok=True)
            
            for i, story in enumerate(hall_list):
                metric_val = story[metric_key]
                name = story['agent_name']
                filename = f"Rank_{i+1:02d}__{metric_key}_{metric_val}__{name}.txt"
                
                try:
                    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
                        f.write(f"AGENTE: {name}\n")
                        f.write(f"RANK: {i+1}\n")
                        f.write(f"MÉTRICA: {metric_key.upper()} = {metric_val}\n")
                        f.write(f"ANDAR FINAL: {story['floor']}\n")
                        f.write(f"NÍVEL FINAL: {story['level']}\n")
                        f.write(f"INIMIGOS DERROTADOS: {story['enemies_defeated']}\n")
                        f.write(f"DANO TOTAL CAUSADO: {story['damage_dealt']:.1f}\n") # Novo
                        f.write(f"CAUSA DA MORTE: {story['death_cause']}\n")       # Novo
                        
                        f.write("\n--- EQUIPAMENTOS FINAIS ---\n")
                        equipment = story.get('equipment', {})
                        if equipment:
                            for slot, item in equipment.items():
                                f.write(f"  {slot}: {item}\n")
                        else:
                            f.write("  (Nenhum)\n")

                        f.write("\n" + "="*50 + "\n\nHISTÓRIA DO AGENTE:\n" + "="*50 + "\n")
                        f.writelines(story['log_content'])
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  [Callback ERROR] Falha ao salvar: {e}")
        
        _save_list(self.hall_of_fame_level, "top_por_nivel", "level")
        _save_list(self.hall_of_fame_floor, "top_por_andar", "floor")
        _save_list(self.hall_of_fame_enemies, "top_por_inimigos", "enemies_defeated")
        
        if self.verbose > 0:
            print("Hall da Fama salvo.")