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
    Callback personalizado para logar métricas específicas do BuriedBrains.
    """
    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_wins = []
        self.episode_levels = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 'dones' sinaliza fim de episódio em cada ambiente vetorizado
        dones = self.locals.get("dones", [])        

        for i, done in enumerate(dones):
            if not done:                                
                continue

            self.episode_count += 1
            print(f"[DEBUG] Episódio {self.episode_count} finalizado no ambiente {i}.")            

            # Obtém info do ambiente atual
            info = self.locals["infos"][i]

            # Usa final_info se existir
            final_info = info.get("final_info", info)
            if not isinstance(final_info, dict):
                continue

            final_status = final_info.get("final_status")
            if not final_status:
                continue

            # Coleta dados do episódio
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))

            # Loga médias a cada N episódios
            if self.episode_count % self.log_interval == 0:
                if self.episode_wins:
                    win_rate = float(np.mean(self.episode_wins))
                    self.logger.record("custom/win_rate_last_episodes", win_rate)
                    self.logger.dump(step=self.num_timesteps)
                    if self.verbose > 0:
                        print(f"[LOG] Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count}: "
                              f"win_rate={win_rate:.2f}")
                    self.episode_wins.clear()

                if self.episode_levels:
                    avg_level = float(np.mean(self.episode_levels))
                    self.logger.record("custom/avg_level_last_episodes", avg_level)
                    self.logger.dump(step=self.num_timesteps)
                    if self.verbose > 0:
                        print(f"[LOG] avg_level={avg_level:.2f}")
                    self.episode_levels.clear()

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
    logging_callback = LoggingCallback(verbose=1)
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    # --- 5. Criação e configuração do modelo ---
    model = model_class(
        policy_class,
        env,
        verbose=1,
        tensorboard_log=base_logdir,  
        device=device,
        ent_coef=0.01,
    )

    TIMESTEPS = 1_000_000
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
