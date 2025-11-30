# train_marl.py
import os
import argparse
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import SharedPolicyVecEnv 

def transfer_weights(old_model_path, new_model, verbose=0):
    """
    Transplanta os pesos do modelo PvE para o modelo MARL.
    Suporta transferência parcial se o input shape mudou (ex: 14 -> 42).
    """
    if verbose >= 1:
        print(f"\n--- INICIANDO TRANSFER LEARNING ---")
        print(f"Carregando pesos de: {old_model_path}")

    # Carrega na CPU para evitar conflito de memória GPU
    temp_model = RecurrentPPO.load(old_model_path, device='cpu')
    old_state_dict = temp_model.policy.state_dict()
    new_state_dict = new_model.policy.state_dict()

    with torch.no_grad():
        for key in new_state_dict.keys():
            if key in old_state_dict:
                old_param = old_state_dict[key]
                new_param = new_state_dict[key]

                # Caso 1: Shapes idênticos (Cópia total)
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)

                # Caso 2: Camada de Entrada mudou (ex: Obs 14 -> 42)
                # Verifica se é uma matriz de pesos (len=2) e se o output bate
                elif len(old_param.shape) == 2 and old_param.shape[0] == new_param.shape[0]:
                    # Copia apenas as colunas que existiam antes
                    input_size_old = old_param.shape[1]
                    input_size_new = new_param.shape[1]
                    
                    if input_size_old < input_size_new:
                        new_param[:, :input_size_old].copy_(old_param)
                        if verbose >= 1:
                            print(f"[PARCIAL] {key}: Copiado {input_size_old} colunas para as primeiras {input_size_old} de {input_size_new}")

    if verbose >= 1:
        print("--- TRANSFERÊNCIA CONCLUÍDA ---\n")
    del temp_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    
    # Argumentos de Modelo
    parser.add_argument('--pretrained_path', type=str, default=None, help="Caminho do modelo PvE Expert (para começar do zero)")
    parser.add_argument('--resume_path', type=str, default=None, help="Caminho de um checkpoint MARL (para continuar treino)")
    
    parser.add_argument('--suffix', type=str, default="MARL_Shared")
    parser.add_argument('--max_episode_steps', type=int, default=10_000)
    parser.add_argument('--sanctum_floor', type=int, default=20)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_agents', type=int, default=2)
    args = parser.parse_args()

    # Validação básica
    if not args.resume_path and not args.pretrained_path:
        raise ValueError("Você deve fornecer --pretrained_path (começar novo) OU --resume_path (continuar).")

    # --- Configuração ---
    base_logdir = "logs_marl"
    base_models_dir = "models_marl"
    run_name = f"RecurrentPPO_{args.suffix}"
    
    tb_path = os.path.join(base_logdir, run_name)
    model_path = os.path.join(base_models_dir, run_name)
    
    # --- 1. Cria o Ambiente MAE com SharedPolicyVecEnv ---            
    # Instancia o ambiente base
    base_env = BuriedBrainsEnv(
        max_episode_steps=args.max_episode_steps, 
        sanctum_floor=args.sanctum_floor,
        verbose=args.verbose,
        num_agents=args.num_agents,
        seed=args.seed
    ) 
    
    # Aplica o wrapper para Shared Policy
    env = SharedPolicyVecEnv(base_env) 

    # Parâmetros do LSTM (Ajuste automático baseado no número de agentes)
    if args.num_agents <= 4:
        batch_size = 128
        n_steps = 512
        lstm_hidden_size = 128
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
        enable_critic_lstm = False

    elif args.num_agents <= 8:
        batch_size = 256
        n_steps = 512
        lstm_hidden_size = 128
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
        enable_critic_lstm = False

    elif args.num_agents <= 16:
        batch_size = 512
        n_steps = 1024
        lstm_hidden_size = 256
        net_arch = {"pi": [256, 256], "vf": [256, 256]}
        enable_critic_lstm = True

    elif args.num_agents >= 32:
        batch_size = 512
        n_steps = 1024
        lstm_hidden_size = 256
        net_arch = {"pi": [256, 256], "vf": [256, 256]}
        enable_critic_lstm = True  # desativa critic LSTM para aliviar

    else:
        batch_size = 256
        n_steps = 512
        lstm_hidden_size = 128
        net_arch = {"pi": [128, 128], "vf": [128, 128]}
        enable_critic_lstm = True

    lstm_params = {
        "learning_rate": 0.0001,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 10,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "ent_coef": 0.03,
        "vf_coef": 0.4,
        "policy_kwargs": {
            "net_arch": net_arch,
            "lstm_hidden_size": lstm_hidden_size,
            "enable_critic_lstm": enable_critic_lstm
        }
    }

    # --- 2. Carregar ou Criar Modelo ---
    
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"\n>>> RESUMINDO TREINAMENTO MARL <<<")
        print(f"Carregando checkpoint: {args.resume_path}")
        
        # Carrega o modelo completo
        model = RecurrentPPO.load(
            args.resume_path, 
            env=env, # Passa o VecEnv direto
            device='cuda',
            custom_objects={'policy_kwargs': lstm_params['policy_kwargs']} 
        )
        model.set_parameters(lstm_params) 
        
    else:
        print(f"\n>>> INICIANDO NOVO TREINO MARL (TRANSFER LEARNING) <<<")        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env, # Passa o VecEnv direto
            verbose=0,
            tensorboard_log=base_logdir,
            **lstm_params
        )
        # Transplanta o cérebro do PvE
        transfer_weights(args.pretrained_path, model, verbose=1)    

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=model_path, name_prefix="marl_model")
    logging_callback = LoggingCallback(verbose=0, log_interval=1)

    # --- 5. Treino ---
    print(f"Iniciando Treino por {args.total_timesteps} passos...")
    print(f"Logs: {tb_path}")
    print(f"Ambiente: {env.num_envs} agentes treinando em paralelo (Parameter Sharing).")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([checkpoint_callback, logging_callback]),
            tb_log_name=run_name,
            progress_bar=True,
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário. Salvando...")

    # Salva o modelo final
    final_steps = model.num_timesteps
    save_name = f"{model_path}/final_marl_model_{final_steps}_steps.zip"
    model.save(save_name)
    print(f"Modelo final salvo em: {save_name}")
    
    # Salva o Hall da Fama
    logging_callback.save_hall_of_fame(os.path.join(tb_path, "hall_of_fame"))

    print("Encerrado.")

if __name__ == "__main__":
    main()