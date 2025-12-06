# buriedbrains/logging_callbacks.py
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from buriedbrains.plot_utils import save_poincare_plot

class LoggingCallback(BaseCallback):
    """
    Callback avançado para registrar métricas detalhadas do BuriedBrains (MAE)
    e salvar as "melhores histórias" (Hall da Fama) periodicamente.
    """
    def __init__(self, log_interval: int = 10, verbose: int = 1, top_n: int = 10, enable_hall_of_fame: bool = True):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.top_n = top_n

        # Buffers
        self._reset_interval_buffers()

        # Hall da Fama (Listas separadas por critério)
        self.hall_of_fame_level = []
        self.hall_of_fame_floor = []
        self.hall_of_fame_enemies = []

        self.max_floor_ever = 0
        self.episode_count = 0
        self.enable_hall_of_fame = enable_hall_of_fame

    def _reset_interval_buffers(self):
        """Zera os buffers do intervalo."""
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_levels = []
        self.episode_floors = []
        self.episode_lengths = []
        self.episode_total_exp = []

        self.episode_damage_dealt = []
        self.episode_equipment_swaps = [] 
        self.episode_skill_upgrades = []  
        
        self.episode_enemies_defeated = []
        self.episode_invalid_actions = []
        self.episode_total_actions = []        

        self.episode_avg_pve_duration = []
        self.episode_avg_pvp_duration = []

        self.episode_arena_encounters = []
        self.episode_pvp_combats = []
        self.episode_bargains = []
        self.episode_bargains_trade = []
        self.episode_bargains_toll = []
        self.episode_cowardice_kills = []
        self.episode_betrayals = []
        self.episode_final_karma_real = []

    def _extract_final_info(self, info):
        raw = info.get("final_info", None)
        if isinstance(raw, list) and len(raw) > 0:
            return raw[0]
        if isinstance(raw, dict):
            return raw
        return info 

    def _update_hall_of_fame(self, story, hall_list, key):
        """Gerencia a lista de melhores agentes."""
        # Assinatura para evitar duplicatas exatas
        signature = (
            story.get("agent_name"),
            story.get("level"),
            story.get("floor"),
            story.get("enemies_defeated"),
            tuple(story.get("active_skills", [])), # Adiciona skills na assinatura
            story.get("death_cause")
        )

        existing_signatures = {
            (
                s.get("agent_name"),
                s.get("level"),
                s.get("floor"),
                s.get("enemies_defeated"),
                tuple(s.get("active_skills", [])),
                s.get("death_cause")
            )
            for s in hall_list
        }

        if signature in existing_signatures:
            return 

        new_score = story.get(key, 0)

        if len(hall_list) < self.top_n:
            hall_list.append(story)
        else:
            if new_score > hall_list[-1].get(key, 0):
                hall_list[-1] = story
            else:
                return 

        # Reordena
        hall_list.sort(key=lambda s: s[key], reverse=True)
        hall_list[:] = hall_list[:self.top_n]
        
    def _on_step(self) -> bool:
        # Recupera as listas do VecEnv
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for i in range(len(infos)):
            info = infos[i]
            final_status = info.get("final_status", None)

            if not isinstance(final_status, dict):
                continue
            
            self.episode_count += 1
            
            # Coleta métricas
            self.episode_rewards.append(rewards[i])
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))
            self.episode_floors.append(final_status.get("floor", 0))
            self.episode_lengths.append(final_status.get("steps", 0))
            self.episode_total_exp.append(final_status.get("exp", 0))

            self.episode_damage_dealt.append(final_status.get("damage_dealt", 0))
            self.episode_equipment_swaps.append(final_status.get("equipment_swaps", 0))
            
            # Coleta Skill Upgrades
            self.episode_skill_upgrades.append(final_status.get("skill_upgrades", 0))
            
            self.episode_enemies_defeated.append(final_status.get("enemies_defeated", 0))
            self.episode_invalid_actions.append(final_status.get("invalid_actions", 0))
            self.episode_total_actions.append(final_status.get("steps", 1))

            # Durations
            pve_durs = final_status.get("pve_durations", [])
            pvp_durs = final_status.get("pvp_durations", [])
            self.episode_avg_pve_duration.append(np.mean(pve_durs) if pve_durs else 0)            
            self.episode_avg_pvp_duration.append(np.mean(pvp_durs) if pvp_durs else 0)

            # Social
            self.episode_arena_encounters.append(final_status.get("arena_encounters", 0))
            self.episode_pvp_combats.append(final_status.get("pvp_combats", 0))
            self.episode_bargains.append(final_status.get("bargains_succeeded", 0))
            self.episode_bargains_trade.append(final_status.get("bargains_trade", 0))
            self.episode_bargains_toll.append(final_status.get("bargains_toll", 0))
            self.episode_cowardice_kills.append(final_status.get("cowardice_kills", 0))
            self.episode_betrayals.append(final_status.get("betrayals", 0))
            karma = final_status.get("karma", {'real': 0})
            self.episode_final_karma_real.append(karma.get("real", 0))

            # Atualiza máximo global
            cp_floor = final_status.get("floor", 0)
            self.max_floor_ever = max(self.max_floor_ever, cp_floor)

            # Monta a história para o Hall da Fama
            story = {
                'agent_name': final_status.get("agent_name", "Agente"),
                'level': final_status.get("level", 0),
                'floor': cp_floor,
                'enemies_defeated': final_status.get("enemies_defeated", 0),
                'damage_dealt': final_status.get("damage_dealt", 0),
                'death_cause': final_status.get("death_cause", "Desconhecida"),
                'equipment': final_status.get("equipment", {}),
                'active_skills': final_status.get("active_skills", []), # Lista de Skills
                'skill_upgrades': final_status.get("skill_upgrades", 0), # Qtd de upgrades
                'log_content': final_status.get("full_log", []),
                'karma_history': final_status.get("karma_history", [])
            }

            self._update_hall_of_fame(story, self.hall_of_fame_level, 'level')
            self._update_hall_of_fame(story, self.hall_of_fame_floor, 'floor')
            self._update_hall_of_fame(story, self.hall_of_fame_enemies, 'enemies_defeated')

            # Logging
            if self.episode_count % self.log_interval == 0:                
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
                self.logger.record("custom/win_rate", np.mean(self.episode_wins))
                self.logger.record("custom/avg_level", np.mean(self.episode_levels))
                self.logger.record("custom/avg_floor_reached", np.mean(self.episode_floors))
                self.logger.record("custom/avg_damage_dealt", np.mean(self.episode_damage_dealt))
                
                self.logger.record("custom/avg_equipment_swaps", np.mean(self.episode_equipment_swaps))                
                self.logger.record("custom/avg_skill_upgrades", np.mean(self.episode_skill_upgrades))
                
                total_invalid = sum(self.episode_invalid_actions)
                total_actions = sum(self.episode_total_actions)
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0
                self.logger.record("custom/rate_invalid_actions", invalid_rate)

                self.logger.record("combat/avg_pve_duration", np.mean(self.episode_avg_pve_duration))
                self.logger.record("combat/avg_pvp_duration", np.mean(self.episode_avg_pvp_duration))

                self.logger.record("social/total_arena_encounters", np.sum(self.episode_arena_encounters))
                self.logger.record("social/total_pvp_combats", np.sum(self.episode_pvp_combats))
                self.logger.record("social/total_bargains", np.sum(self.episode_bargains))
                self.logger.record("social/total_bargains_trade", np.sum(self.episode_bargains_trade))
                self.logger.record("social/total_bargains_toll", np.sum(self.episode_bargains_toll))
                self.logger.record("social/total_cowardice_kills", np.sum(self.episode_cowardice_kills))
                self.logger.record("social/avg_final_karma", np.mean(self.episode_final_karma_real))
                self.logger.record("social/total_betrayals", np.mean(self.episode_betrayals))

                self.logger.dump(step=self.num_timesteps)

                if self.enable_hall_of_fame:
                    try:
                        log_dir = self.logger.get_dir()
                        if log_dir:
                            hof_path = os.path.join(log_dir, "hall_of_fame")
                            self.save_hall_of_fame(hof_path)
                    except Exception as e:
                        if self.verbose: print(f"[Logger WARN] Falha ao salvar Hall da Fama: {e}")                  

                self._reset_interval_buffers()                

        return True

    def _sanitize_filename(self, name):
        return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()

    def save_hall_of_fame(self, save_dir: str):
        def save_list(hlist, sub, key):
            path = os.path.join(save_dir, sub)
            os.makedirs(path, exist_ok=True)
            
            # Limpa arquivos antigos
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.png')):
                        os.remove(file_path)
                except: pass

            for i, story in enumerate(hlist):
                safe_agent_name = self._sanitize_filename(story['agent_name'])
                fname = f"Rank_{i+1:02d}__{key}_{story[key]}__{safe_agent_name}.txt"
                fpath = os.path.join(path, fname)

                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(f"AGENTE: {story['agent_name']}\n")
                        f.write(f"RANK: {i+1}\n")
                        f.write(f"{key.upper()}: {story[key]}\n")
                        f.write(f"FLOOR: {story['floor']}\n")
                        f.write(f"LEVEL: {story['level']}\n")
                        f.write(f"INIMIGOS DERROTADOS: {story['enemies_defeated']}\n")
                        f.write(f"SKILLS APRENDIDAS: {story.get('skill_upgrades', 0)}\n")
                        f.write(f"CAUSA DA MORTE: {story['death_cause']}\n")
                        
                        # Seção de Skills
                        f.write("\n--- SKILLS (BUILD) ---\n")
                        skills = story.get('active_skills', [])
                        if skills:
                            for idx, s in enumerate(skills):
                                f.write(f"  Slot {idx}: {s}\n")
                        else:
                            f.write("  (Padrão)\n")

                        f.write("\n--- EQUIPAMENTO FINAL ---\n")
                        equipment = story.get('equipment', {})
                        if equipment:
                            for slot, item in equipment.items():
                                f.write(f"  {slot:<10}: {item}\n")
                        else:
                            f.write("  (Nenhum)\n")

                        f.write("\n--- LOG (Últimas Ações) ---\n")
                        logs = story.get('log_content', [])
                        
                        if logs:                            
                            f.writelines(logs)
                        else:
                            f.write("  (Vazio)\n")

                    if story.get("karma_history"):
                        plot_name = fname.replace(".txt", "_Karma.png")
                        plot_path = os.path.join(path, plot_name)
                        save_poincare_plot(story["karma_history"], story["agent_name"], plot_path)

                except Exception as e:
                    if self.verbose: print(f"[Logger ERROR] {e}")

        save_list(self.hall_of_fame_level, "top_por_nivel", "level")
        save_list(self.hall_of_fame_floor, "top_por_andar", "floor")
        save_list(self.hall_of_fame_enemies, "top_por_inimigos", "enemies_defeated")