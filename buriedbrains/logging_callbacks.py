# buriedbrains/logging_callbacks.py
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from buriedbrains.plot_utils import save_poincare_plot

class LoggingCallback(BaseCallback):
    """
    Callback avançado para registrar métricas detalhadas do BuriedBrains (SAE/MAE)
    e salvar as "melhores histórias" (Hall da Fama) periodicamente.
    """
    def __init__(self, log_interval: int = 10, verbose: int = 1, top_n: int = 10):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.top_n = top_n

        # Buffers
        self._reset_interval_buffers()

        # Hall da Fama
        self.hall_of_fame_level = []
        self.hall_of_fame_floor = []
        self.hall_of_fame_enemies = []

        self.max_floor_ever = 0
        self.episode_count = 0

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

        self.episode_artifacts_equiped = []
        self.episode_sanctum_steps = []
        self.episode_sanctum_deaths = [] # Mortes por profanação

    def _extract_final_info(self, info):
        """Retorna o final_info real do SB3, sem quebrar."""
        raw = info.get("final_info", None)
        if isinstance(raw, list) and len(raw) > 0:
            return raw[0]
        if isinstance(raw, dict):
            return raw
        return info  # fallback seguro

    def _update_hall_of_fame(self, story, hall_list, key):
        """
        Insere a história no Hall da Fama garantindo:
        - no máximo top_n entradas
        - ordenação estável por score
        - remoção de duplicatas
        """

        # --- 1. Evita duplicatas reais ---
        signature = (
            story.get("agent_name"),
            story.get("level"),
            story.get("floor"),
            story.get("enemies_defeated"),
            story.get("damage_dealt"),
            story.get("death_cause")
        )

        existing_signatures = {
            (
                s.get("agent_name"),
                s.get("level"),
                s.get("floor"),
                s.get("enemies_defeated"),
                s.get("damage_dealt"),
                s.get("death_cause")
            )
            for s in hall_list
        }

        if signature in existing_signatures:
            return  # Já existe, não adiciona outra cópia

        # --- 2. Insere se houver espaço ou se score for maior ---
        new_score = story.get(key, 0)

        if len(hall_list) < self.top_n:
            hall_list.append(story)
        else:
            # Se for melhor que o pior, substitui
            if new_score > hall_list[-1].get(key, 0):
                hall_list[-1] = story
            else:
                return  # Não entra no top N

        # --- 3. Reordena ---
        hall_list.sort(key=lambda s: s[key], reverse=True)

        # --- 4. Garante tamanho máximo (pode cortar empates excessivos) ---
        hall_list[:] = hall_list[:self.top_n]
        
    def _on_step(self) -> bool:
        # Recupera as listas do VecEnv
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        # Itera sobre todos os agentes (índices)
        for i in range(len(infos)):
            
            info = infos[i]
            
            # Extrai o relatório final real            
            final_status = info.get("final_status", None)

            # Se não tem relatório final, pula este agente neste frame
            if not isinstance(final_status, dict):
                continue
            
            # Coleta dados do episódio finalizado
            self.episode_count += 1
            if self.verbose > 1:
                 print(f"[LoggingCallback] Capturado episódio de {final_status.get('agent_name')} (Morte/Vitória)")

            # Coleta métricas
            self.episode_rewards.append(rewards[i])
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))
            self.episode_floors.append(final_status.get("floor", 0))
            self.episode_lengths.append(final_status.get("steps", 0))
            self.episode_total_exp.append(final_status.get("exp", 0))

            self.episode_damage_dealt.append(final_status.get("damage_dealt", 0))
            self.episode_equipment_swaps.append(final_status.get("equipment_swaps", 0))
            self.episode_enemies_defeated.append(final_status.get("enemies_defeated", 0))
            self.episode_invalid_actions.append(final_status.get("invalid_actions", 0))
            self.episode_total_actions.append(final_status.get("steps", 1))
            self.episode_artifacts_equiped.append(final_status.get("artifacts_equiped", 0))
            self.episode_sanctum_steps.append(final_status.get("sanctum_steps", 0))            

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

            # Detecta Morte por Profanação
            # Verificamos se a string da causa da morte contém a palavra chave que determina morte por profanar o Templo
            cause = final_status.get("death_cause", "")
            if "Colapso" in cause or "Demora" in cause:
                self.episode_sanctum_deaths.append(1)
            else:
                self.episode_sanctum_deaths.append(0)

            # Atualiza máximo global
            cp_floor = final_status.get("floor", 0)
            self.max_floor_ever = max(self.max_floor_ever, cp_floor)

            # Monta a história para o Hall da Fama
            story = {
                'agent_name': final_status.get("agent_name", "Agente_Desconhecido"),
                'level': final_status.get("level", 0),
                'floor': cp_floor,
                'enemies_defeated': final_status.get("enemies_defeated", 0),
                'damage_dealt': final_status.get("damage_dealt", 0),
                'death_cause': final_status.get("death_cause", "Desconhecida"),
                'equipment': final_status.get("equipment", {}),
                'log_content': final_status.get("full_log", []),
                'karma_history': final_status.get("karma_history", [])
            }

            self._update_hall_of_fame(story, self.hall_of_fame_level, 'level')
            self._update_hall_of_fame(story, self.hall_of_fame_floor, 'floor')
            self._update_hall_of_fame(story, self.hall_of_fame_enemies, 'enemies_defeated')

            # Logging do TensorBoard E SALVAMENTO PERIÓDICO
            if self.episode_count % self.log_interval == 0:

                # básicos
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
                self.logger.record("custom/win_rate", np.mean(self.episode_wins))
                self.logger.record("custom/avg_level", np.mean(self.episode_levels))
                self.logger.record("custom/avg_floor_reached", np.mean(self.episode_floors))

                # comportamento
                self.logger.record("custom/avg_damage_dealt", np.mean(self.episode_damage_dealt))
                self.logger.record("custom/avg_equipment_swaps", np.mean(self.episode_equipment_swaps))
                total_invalid = sum(self.episode_invalid_actions)
                total_actions = sum(self.episode_total_actions)
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0
                self.logger.record("custom/rate_invalid_actions", invalid_rate)

                # duração
                self.logger.record("combat/avg_pve_duration", np.mean(self.episode_avg_pve_duration))
                self.logger.record("combat/avg_pvp_duration", np.mean(self.episode_avg_pvp_duration))

                # social
                self.logger.record("social/total_arena_encounters", np.sum(self.episode_arena_encounters))
                self.logger.record("social/total_pvp_combats", np.sum(self.episode_pvp_combats))
                self.logger.record("social/total_bargains", np.sum(self.episode_bargains))
                self.logger.record("social/total_bargains_trade", np.sum(self.episode_bargains_trade))
                self.logger.record("social/total_bargains_toll", np.sum(self.episode_bargains_toll))
                self.logger.record("social/total_cowardice_kills", np.sum(self.episode_cowardice_kills))
                self.logger.record("social/avg_final_karma", np.mean(self.episode_final_karma_real))
                self.logger.record("social/total_betrayals", np.mean(self.episode_betrayals))
                
                # Média de artefatos equipados por episódio (indica procura por barganha)
                self.logger.record("custom/avg_artifacts_equiped", np.mean(self.episode_artifacts_equiped))
                
                # Média de passos gastos dentro do santuário (indica eficiência da negociação)
                self.logger.record("social/avg_steps_in_sanctum", np.mean(self.episode_sanctum_steps))

                self.logger.dump(step=self.num_timesteps)

                if self.verbose:
                    print(f"--- Intervalo Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count} (Timestep {self.num_timesteps}) ---")
                    print(f"  Avg Floor: {np.mean(self.episode_floors):.2f} | Max Ever: {self.max_floor_ever}")
                    print(f"  Social: {np.sum(self.episode_arena_encounters)} Encounters, {np.sum(self.episode_bargains)} Bargains")
                    print("-" * 40)
                
                # --- SALVAMENTO AUTOMÁTICO DO HALL DA FAMA ---
                # Salva sempre que logar no TensorBoard, para garantir que não percamos dados
                try:
                    log_dir = self.logger.get_dir()
                    if log_dir:
                        hof_path = os.path.join(log_dir, "hall_of_fame")
                        self.save_hall_of_fame(hof_path)
                except Exception as e:
                    if self.verbose: print(f"[Logger WARN] Falha ao salvar Hall da Fama automático: {e}")                

                self._reset_interval_buffers()                

        return True

    # Função auxiliar para garantir que o nome do arquivo seja válido no SO
    def _sanitize_filename(self, name):
        return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()

    def save_hall_of_fame(self, save_dir: str):
        if self.verbose:
            print(f"\nSalvando Hall da Fama em {save_dir}...")

        def save_list(hlist, sub, key):
            path = os.path.join(save_dir, sub)
            os.makedirs(path, exist_ok=True)
            
            # Remove arquivos .txt e .png antigos para evitar duplicidade de Ranks
            # (ex: Rank 01 antigo vs Rank 01 novo)
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) and (filename.endswith('.txt') or filename.endswith('.png')):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Erro ao limpar arquivo antigo {filename}: {e}")            

            for i, story in enumerate(hlist):
                # Sanitiza o nome para evitar erros de caminho inválido
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
                        f.write(f"DANO TOTAL: {story['damage_dealt']}\n")
                        f.write(f"CAUSA DA MORTE: {story['death_cause']}\n")
                        f.write("\n--- EQUIPAMENTO FINAL ---\n")
                        
                        equipment = story.get('equipment', {})
                        if equipment:
                            for slot, item in equipment.items():
                                f.write(f"  {slot:<10}: {item}\n")
                        else:
                            f.write("  (Nenhum equipamento registrado)\n")

                        f.write("\n--- LOG ---\n")
                        # Verifica se log_content existe e é lista
                        logs = story.get('log_content', [])
                        if logs:
                            f.writelines(logs)
                        else:
                            f.write("  (Log vazio)\n")

                    # Karma plot
                    if story.get("karma_history"):
                        # Nome do plot vinculado ao arquivo de texto para facilitar identificação
                        plot_name = fname.replace(".txt", "_Karma.png")
                        plot_path = os.path.join(path, plot_name)
                        save_poincare_plot(story["karma_history"], story["agent_name"], plot_path)

                except Exception as e:
                    if self.verbose: print(f"[Logger ERROR] Erro ao salvar arquivo {fname}: {e}")

        # Chama a função interna para cada categoria
        save_list(self.hall_of_fame_level, "top_por_nivel", "level")
        save_list(self.hall_of_fame_floor, "top_por_andar", "floor")
        save_list(self.hall_of_fame_enemies, "top_por_inimigos", "enemies_defeated")

        if self.verbose:
            print("Hall da Fama salvo com sucesso.\n")