# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_FILESYSTEM"] = "1"

import time
import gymnasium as gym
import numpy as np
import torch
import argparse

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Importa o ambiente personalizado
from buriedbrains.env import BuriedBrainsEnv
# Importa o callback de logging personalizado
from logging_callbacks import LoggingCallback

def main():
    """
    Script principal para treinar um agente PPO no ambiente BuriedBrains.
    """
    
    # --- 0. PARSER DE ARGUMENTOS ---
    parser = argparse.ArgumentParser(description="Script de Treinamento BuriedBrains")
    
    parser.add_argument('--no_lstm', action='store_true', help="Usar PPO padrão em vez de RecurrentPPO (LSTM)")
    parser.add_argument('--total_timesteps', type=int, default=5_000_000, help="Total de passos de treino")
    parser.add_argument('--suffix', type=str, default="", help="Sufixo para o nome da run")
    parser.add_argument('--max_episode_steps', type=int, default=30000, help="Limite de passos por episódio")
    parser.add_argument('--budget_multiplier', type=float, default=1.0, help="Multiplicador de dificuldade")
    parser.add_argument('--load_path', type=str, default=None, help="Caminho para carregar modelo .zip")
    
    args = parser.parse_args()

    # --- Escolha da política ---    
    use_lstm = not args.no_lstm 

    if use_lstm:
        model_class = RecurrentPPO
        policy_class = "MlpLstmPolicy"
    else:
        model_class = PPO
        policy_class = "MlpPolicy"

    print(f"Usando a política: {policy_class}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando o dispositivo: {device}")

    # --- 1. Verificação do ambiente base ---
    # (Opcional/Pulada para velocidade em maratona)
    # print("Verificando o ambiente BuriedBrainsEnv base...")
    
    # --- 2. Criação do ambiente vetorizado ---
    print("Criando ambiente vetorizado para treinamento...")
    env = DummyVecEnv([lambda: BuriedBrainsEnv(
        verbose=0, # Verbose 0 para velocidade máxima
        max_episode_steps=args.max_episode_steps,
        budget_multiplier=args.budget_multiplier
    )])

    # --- 3. Configuração de logs e diretórios ---
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_logdir = os.path.normpath("logs").replace("\\", "/")
    base_models_dir = os.path.normpath("models").replace("\\", "/")

    os.makedirs(base_logdir, exist_ok=True)
    os.makedirs(base_models_dir, exist_ok=True)

    if args.suffix:
        run_name = f"{model_class.__name__}_{args.suffix}"
    else:
        run_name = f"{model_class.__name__}_{policy_class}_{timestamp}"
    
    run_models_dir = os.path.join(base_models_dir, run_name)
    os.makedirs(run_models_dir, exist_ok=True)

    tb_path = os.path.join(base_logdir, run_name)
    os.makedirs(tb_path, exist_ok=True)

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=500_000, save_path=run_models_dir, name_prefix="bb_model")
    # Usando o novo LoggingCallback importado
    logging_callback = LoggingCallback(verbose=0, log_interval=2)
    callback_list = CallbackList([checkpoint_callback, logging_callback])

    # --- 5. Criação e configuração do modelo (HPO) ---
    # Parâmetros otimizados (PPO)
    ppo_params = {
        "learning_rate": 0.0009880458115502663,
        "n_steps": 512,
        "batch_size": 128,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "ent_coef": 0.01814275260474149,
        "vf_coef": 0.5,
        "clip_range": 0.2,
        "n_epochs": 20,
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]}
        }
    }

    # Parâmetros otimizados (LSTM)
    lstm_params = {
        "learning_rate": 0.0001653881849494385,
        "n_steps": 512,
        "batch_size": 128,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "ent_coef": 0.00943339989056226,
        "vf_coef": 0.4,
        "clip_range": 0.2,
        "n_epochs": 20,
        "policy_kwargs": {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "lstm_hidden_size": 128,
            "enable_critic_lstm": False
        }
    }

    if use_lstm:
        print("Carregando modelo RecurrentPPO com hiperparâmetros otimizados (LSTM)...")
        model_kwargs = lstm_params
    else:
        print("Carregando modelo PPO com hiperparâmetros otimizados (MLP)...")
        model_kwargs = ppo_params
    
    # Lógica de Carregamento vs Criação
    if args.load_path and os.path.exists(args.load_path):
        print(f"Carregando modelo existente de: {args.load_path}")
        model = model_class.load(
            args.load_path,
            env=env,
            device=device,
            custom_objects={"policy_kwargs": model_kwargs.get("policy_kwargs", {})} 
        )
        model.set_parameters(model_kwargs)
        print("Parâmetros otimizados re-aplicados ao modelo carregado.")
    else:
        if args.load_path:
            print(f"Aviso: --load_path '{args.load_path}' não encontrado.")
        print("Criando novo modelo do zero.")
        model = model_class(
            policy_class,
            env,
            verbose=0,
            tensorboard_log=base_logdir, 
            device=device,
            **model_kwargs
        )

    TIMESTEPS = args.total_timesteps
    print(f"Iniciando treinamento por {TIMESTEPS} passos...")
    print(f"Logs do TensorBoard: {tb_path}")

    # --- INÍCIO DO TREINO ---
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback_list,
        tb_log_name=run_name,  
        progress_bar=False, # Sem barra para velocidade
        reset_num_timesteps=False # Continua logs
    )

    # --- 6. Salvamento Final ---
    final_save_path = os.path.join(run_models_dir, f"model_{model.num_timesteps}_steps.zip")
    model.save(final_save_path)
    print(f"Modelo final salvo em {final_save_path}")
    
    # --- 7. Salvar o Hall da Fama ---
    print("Salvando histórias do Hall da Fama...")
    if len(callback_list.callbacks) > 1 and isinstance(callback_list.callbacks[1], LoggingCallback):
        logging_callback_instance = callback_list.callbacks[1]
        hof_save_path = os.path.join(tb_path, "hall_of_fame")
        logging_callback_instance.save_hall_of_fame(hof_save_path)
        print(f"Histórias salvas em: {hof_save_path}")
    else:
        print("[WARN] LoggingCallback não encontrado.")    

    print(f"Treinamento concluído.")
    env.close()

if __name__ == "__main__":
    main()