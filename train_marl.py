# train_marl.py
import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import SymmetricSelfPlayWrapper

def transfer_weights(old_model_path, new_model, verbose=0):
    """
    Transplanta os pesos do modelo PvE para o modelo MARL (38 inputs).
    Suporta modelos antigos (14 inputs) e modelos revamp (16 inputs).

    verbose:
        0 = silencioso
        1 = logs essenciais
        2 = logs detalhados (parâmetro por parâmetro)
    """
    if verbose >= 1:
        print(f"\n--- INICIANDO TRANSFER LEARNING ---")
        print(f"Carregando pesos de: {old_model_path}")

    temp_model = RecurrentPPO.load(old_model_path, device='cpu')
    old_state_dict = temp_model.policy.state_dict()
    new_state_dict = new_model.policy.state_dict()

    with torch.no_grad():
        for key in new_state_dict.keys():
            if key in old_state_dict:
                old_param = old_state_dict[key]
                new_param = new_state_dict[key]

                # Caso 1: shapes iguais (cópia direta)
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)

                    if verbose == 2:
                        print(f"[OK] Copiado: {key}")

                # Caso 2: camada de entrada com tamanhos diferentes (transferência parcial)
                elif len(old_param.shape) == 2 and new_param.shape[1] == 38:
                    input_size_old = old_param.shape[1]

                    if input_size_old in [14, 16]:
                        # Copia colunas existentes
                        new_param[:, :input_size_old].copy_(old_param)

                        if verbose >= 1:
                            print(f"[PARCIAL] {key}: Copiado {input_size_old} → 38 colunas")

                        if verbose == 2:
                            print(f"         Old shape: {old_param.shape}, New shape: {new_param.shape}")

                    else:
                        if verbose >= 1:
                            print(f"[IGNORADO] {key}: Input antigo desconhecido ({old_param.shape})")

                # Caso 3: shapes totalmente incompatíveis
                else:
                    if verbose >= 1:
                        print(f"[SKIP] {key}: shape incompatível {old_param.shape} → {new_param.shape}")

            else:
                if verbose == 2:
                    print(f"[NOVO PARÂMETRO] {key}: não existe no modelo antigo")

    if verbose >= 1:
        print("--- TRANSFERÊNCIA CONCLUÍDA ---\n")

    del temp_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    parser.add_argument('--pretrained_path', type=str, required=True, help="Caminho do modelo PvE Expert (16 inputs)")
    parser.add_argument('--suffix', type=str, default="MARL_Transfer")
    parser.add_argument('--max_episode_steps', type=int, default=10_000)
    args = parser.parse_args()

    # --- Configuração ---
    base_logdir = "logs_marl"
    base_models_dir = "models_marl"
    run_name = f"RecurrentPPO_{args.suffix}"
    
    tb_path = os.path.join(base_logdir, run_name)
    model_path = os.path.join(base_models_dir, run_name)
    
    # --- 1. Cria o Ambiente MAE com Wrapper ---
    # Função lambda para criar o env envelopado
    max_episode_steps = args.max_episode_steps
    def make_env():
        env = BuriedBrainsEnv(max_episode_steps) # Dificuldade normal
        env = SymmetricSelfPlayWrapper(env) # Transforma em Single-Agent para o SB3
        return env

    vec_env = DummyVecEnv([make_env])

    # --- 2. Cria o Novo Modelo MARL (Input 38) ---    
    lstm_params = {
        "learning_rate": 0.000165, 
        "n_steps": 512, 
        "batch_size": 128, 
        "n_epochs": 20, 
        "gamma": 0.98, 
        "gae_lambda": 0.92, 
        "ent_coef": 0.009, 
        "vf_coef": 0.4,
        "policy_kwargs": {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "lstm_hidden_size": 128,
            "enable_critic_lstm": False
        }
    }

    new_model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=0,
        tensorboard_log=base_logdir,
        **lstm_params
    )

    # --- 3. Aplica Transfer Learning ---
    transfer_weights(args.pretrained_path, new_model)

    # --- 4. Vincula o Modelo ao Wrapper ---
    # O Wrapper precisa conhecer o modelo para auto-jogar contra si mesmo
    vec_env.envs[0].set_model(new_model) 

    # --- 5. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=model_path, name_prefix="marl_model")
    logging_callback = LoggingCallback(verbose=1, log_interval=1)

    # --- 6. Treino ---
    print(f"Iniciando Treino MARL (Self-Play) por {args.total_timesteps} passos...")
    new_model.learn(
        total_timesteps=args.total_timesteps,
        callback = CallbackList([checkpoint_callback, logging_callback]),
        tb_log_name=run_name,
        progress_bar=True
    )
    
    new_model.save(f"{model_path}/final_marl_model_{args.total_timesteps}_steps")
    print("Treinamento Concluído!")

if __name__ == "__main__":
    main()