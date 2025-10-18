# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silencia logs chatos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # evita warning do oneDNN
os.environ["TF_USE_LEGACY_FILESYSTEM"] = "1"  # evita bug de verificação no Windows

import time
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Importa o ambiente personalizado
from buriedbrains.env import BuriedBrainsEnv

class LoggingCallback(BaseCallback):
    """
    Callback customizado para registrar métricas detalhadas do BuriedBrains.
    Inclui: win_rate, avg_level, avg_floor, max_floor, episode_length,
            avg_enemies_defeated, rate_invalid_actions.
    """
    def __init__(self, log_interval: int = 50, verbose: int = 1): # Aumentei o log_interval padrão
        super().__init__(verbose)
        self.log_interval = log_interval
        # Listas para guardar dados do intervalo
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_levels = []
        self.episode_floors = []
        self.episode_lengths = []
        self.episode_enemies_defeated = []
        self.episode_invalid_actions = []
        self.episode_total_actions = [] # Para calcular a taxa de ações inválidas
        # Contadores gerais
        self.episode_count = 0
        self.max_floor_ever = 0 # Rastreia o recorde de andar

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            # --- Coleta de Dados do Episódio Finalizado ---
            self.episode_count += 1
            info = self.locals["infos"][i]
            # final_info agora é garantido em ambientes Gymnasium/SB3 quando done=True
            final_info = info.get("final_info")
            if not isinstance(final_info, dict):
                # Fallback caso 'final_info' não exista (menos provável com gym recente)
                final_info = info
                if not isinstance(final_info, dict):
                     if self.verbose > 1: print("[Callback WARN] Não foi possível encontrar final_info.")
                     continue # Pula se não conseguir ler os dados

            # Pega os dados que adicionamos no env.py
            final_status = final_info.get("final_status")
            if not final_status:
                 if self.verbose > 1: print("[Callback WARN] 'final_status' não encontrado em final_info.")
                 continue # Pula se 'final_status' estiver faltando

            # Adiciona dados às listas do intervalo
            self.episode_rewards.append(self.locals["rewards"][i]) # Recompensa total pode ser útil
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))
            current_floor = final_status.get("floor", 0)
            self.episode_floors.append(current_floor)
            self.episode_lengths.append(final_status.get("steps", 0))
            self.episode_enemies_defeated.append(final_status.get("enemies_defeated", 0))
            self.episode_invalid_actions.append(final_status.get("invalid_actions", 0))
            self.episode_total_actions.append(final_status.get("steps", 1)) # Usa steps como total de ações

            # Atualiza o recorde de andar
            self.max_floor_ever = max(self.max_floor_ever, current_floor)

            # --- Loga Médias a Cada `log_interval` Episódios ---
            if self.episode_count % self.log_interval == 0:
                # Calcula médias e outras estatísticas
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                win_rate = np.mean(self.episode_wins) if self.episode_wins else 0
                avg_level = np.mean(self.episode_levels) if self.episode_levels else 0
                avg_floor = np.mean(self.episode_floors) if self.episode_floors else 0
                max_floor_interval = np.max(self.episode_floors) if self.episode_floors else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                avg_enemies = np.mean(self.episode_enemies_defeated) if self.episode_enemies_defeated else 0
                
                total_invalid = sum(self.episode_invalid_actions)
                total_actions = sum(self.episode_total_actions)
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0

                # Loga para o TensorBoard
                self.logger.record("rollout/ep_rew_mean", mean_reward) # Recompensa média padrão SB3
                self.logger.record("custom/win_rate", win_rate)
                self.logger.record("custom/avg_level", avg_level)
                self.logger.record("custom/avg_floor_reached", avg_floor)
                self.logger.record("custom/max_floor_reached_interval", max_floor_interval) # Max no intervalo
                self.logger.record("custom/max_floor_reached_ever", self.max_floor_ever)   # Max geral
                self.logger.record("custom/avg_episode_length", avg_length)
                self.logger.record("custom/avg_enemies_defeated", avg_enemies)
                self.logger.record("custom/rate_invalid_actions", invalid_rate)

                # Força a escrita dos logs no TensorBoard
                self.logger.dump(step=self.num_timesteps)

                # Printa no console se verbose > 0
                if self.verbose > 0:
                    print(f"--- Intervalo Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count} (Timestep {self.num_timesteps}) ---")
                    print(f"  Avg Reward: {mean_reward:.2f}")
                    print(f"  Win Rate: {win_rate:.2f}")
                    print(f"  Avg Level: {avg_level:.2f}")
                    print(f"  Avg Floor Reached: {avg_floor:.2f}")
                    print(f"  Max Floor (Interval): {max_floor_interval}")
                    print(f"  Max Floor (Ever): {self.max_floor_ever}")
                    print(f"  Avg Ep Length: {avg_length:.1f}")
                    print(f"  Avg Enemies Defeated: {avg_enemies:.2f}")
                    print(f"  Invalid Action Rate: {invalid_rate:.3f}")
                    print("-" * (len(f"--- Intervalo Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count} (Timestep {self.num_timesteps}) ---")))


                # Limpa as listas para o próximo intervalo
                self.episode_rewards.clear()
                self.episode_wins.clear()
                self.episode_levels.clear()
                self.episode_floors.clear()
                self.episode_lengths.clear()
                self.episode_enemies_defeated.clear()
                self.episode_invalid_actions.clear()
                self.episode_total_actions.clear()

        return True

def main():
    """
    Script principal para treinar um agente PPO no ambiente BuriedBrains.
    """
    # --- Escolha da política ---
    use_lstm = True  # Mude para False se quiser treinar sem memória (PPO padrão)

    if use_lstm:
        model_class = RecurrentPPO
        policy_class = "MlpLstmPolicy"
    else:
        model_class = PPO
        policy_class = "MlpPolicy"

    print(f"Usando a política: {policy_class}")

    # Dispositivo (GPU se disponível)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando o dispositivo: {device}")

    # --- 1. Verificação do ambiente base ---
    print("Verificando o ambiente BuriedBrainsEnv base...")
    temp_env = BuriedBrainsEnv()
    try:
        check_env(temp_env)
        print("Verificação do ambiente bem-sucedida!")
    except Exception as e:
        print(f"Erro na verificação do ambiente: {e}")
        temp_env.close()
        return
    temp_env.close()

    # --- 2. Criação do ambiente vetorizado ---
    print("Criando ambiente vetorizado para treinamento...")
    env = DummyVecEnv([lambda: BuriedBrainsEnv()])

    # --- 3. Configuração de logs e diretórios ---
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_logdir = os.path.normpath("logs").replace("\\", "/")
    base_models_dir = os.path.normpath("models").replace("\\", "/")


    os.makedirs(base_logdir, exist_ok=True)
    os.makedirs(base_models_dir, exist_ok=True)

    run_name = f"{model_class.__name__}_{policy_class}_{timestamp}"
    run_models_dir = os.path.join(base_models_dir, run_name)
    os.makedirs(run_models_dir, exist_ok=True)

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=run_models_dir, name_prefix="bb_model")
    logging_callback = LoggingCallback()
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    # --- 5. Criação e configuração do modelo ---
    model = model_class(
        policy_class,
        env,
        verbose=0,
        tensorboard_log=base_logdir,  
        device=device,
        ent_coef=0.01,
    )

    TIMESTEPS = 5_000_000
    print(f"Iniciando treinamento por {TIMESTEPS} passos...")
    print(f"Logs do TensorBoard serão salvos em: {os.path.join(base_logdir, run_name)}")

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback_list,
        tb_log_name=run_name,  
        progress_bar=True,
    )

    # --- 6. Salvamento do modelo final ---
    model.save(os.path.join(run_models_dir, f"final_model_{TIMESTEPS}"))
    print(f"Treinamento concluído. Modelo final salvo em {run_models_dir}")

    env.close()

if __name__ == "__main__":
    main()
 