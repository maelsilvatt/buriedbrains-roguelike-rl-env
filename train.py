# train.py
import os
import time
import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Importa a clase do ambiente 
from buriedbrains.env import BuriedBrainsEnv

class LoggingCallback(BaseCallback):
    """
    Um callback personalizado para logar métricas específicas do BuriedBrains.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_wins = []
        self.episode_levels = []
        self.episode_count = 0

    def _on_step(self) -> bool:        
        # Acessa 'dones' através do dicionário self.locals
        dones = self.locals.get("dones", [])
        
        # O loop agora verifica cada ambiente no vetor (mesmo que seja só um)
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                
                info = self.locals['infos'][i]
                final_status = info.get('final_status')
                
                if final_status:
                    self.episode_wins.append(1 if final_status.get('win') else 0)
                    self.episode_levels.append(final_status.get('level'))
                    
                    if self.episode_count % 10 == 0:
                        win_rate = np.mean(self.episode_wins)
                        avg_level = np.mean(self.episode_levels)
                        
                        self.logger.record("custom/win_rate_ (last_10_eps)", win_rate)
                        self.logger.record("custom/average_level (last_10_eps)", avg_level)
                        
                        # Limpa as listas para a próxima média de 10 episódios
                        self.episode_wins = []
                        self.episode_levels = []

        return True
    
def main():
    """
    Script principal para treinar um agente PPO no ambiente BuriedBrains.
    """

    # Verifica se uma GPU com CUDA está disponível, caso contrário, usa a CPU
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Usando o dispositivo: {device}")

    # --- 2. Criação do Ambiente ---
    print("Criando o ambiente BuriedBrainsEnv...")
    env = DummyVecEnv([lambda: BuriedBrainsEnv()])

    # --- 3. Configuração dos Logs e Modelos ---
    log_name = f"PPO_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    models_dir = f"models/{log_name}"
    logdir = "logs"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # --- 4. Definição dos Callbacks ---
    checkpoint_callback = CheckpointCallback(
      save_freq=20000,
      save_path=models_dir,
      name_prefix="bb_model"
    )
    logging_callback = LoggingCallback()
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    # --- 5. Definição e Treinamento do Modelo ---
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=logdir,
        device=device
    )

    TIMESTEPS = 1_000_000
    print(f"Iniciando treinamento por {TIMESTEPS} passos...")
    print(f"Logs do TensorBoard serão salvos em: {logdir}/{log_name}_1")
    
    model.learn(
        total_timesteps=TIMESTEPS, 
        callback=callback_list, 
        tb_log_name=log_name,
        progress_bar=True
    )

    # --- 6. Salvar o Modelo Final ---
    model.save(f"{models_dir}/final_model_{TIMESTEPS}")
    print(f"Treinamento concluído. Modelo final salvo em {models_dir}")

    env.close()

if __name__ == "__main__":
    main()