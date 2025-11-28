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
    Apenas para o INÍCIO do treinamento.
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

                # Caso 2: camada de entrada com tamanhos diferentes (transferência parcial)
                elif len(old_param.shape) == 2 and new_param.shape[1] == 38:
                    input_size_old = old_param.shape[1]
                    if input_size_old in [14, 16]:
                        new_param[:, :input_size_old].copy_(old_param)
                        if verbose >= 1:
                            print(f"[PARCIAL] {key}: Copiado {input_size_old} -> 38 colunas")

    if verbose >= 1:
        print("--- TRANSFERÊNCIA CONCLUÍDA ---\n")
    del temp_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    
    # Argumentos de Modelo
    parser.add_argument('--pretrained_path', type=str, default=None, help="Caminho do modelo PvE Expert (para começar do zero)")
    parser.add_argument('--resume_path', type=str, default=None, help="Caminho de um checkpoint MARL (para continuar treino)")
    
    parser.add_argument('--suffix', type=str, default="MARL_Transfer")
    parser.add_argument('--max_episode_steps', type=int, default=10_000)
    parser.add_argument('--sanctum_floor', type=int, default=20)
    parser.add_argument('--verbose', type=int, default=0)
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
    
    # --- 1. Cria o Ambiente MAE com Wrapper ---            
    def make_env():        
        env = BuriedBrainsEnv(
            max_episode_steps=args.max_episode_steps, 
            sanctum_floor=args.sanctum_floor,
            verbose=args.verbose
        ) 
        env = SymmetricSelfPlayWrapper(env) 
        return env

    vec_env = DummyVecEnv([make_env])

    # Parâmetros do LSTM (Otimizados)
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

    # --- 2. Carregar ou Criar Modelo ---
    
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"\n>>> RESUMINDO TREINAMENTO MARL <<<")
        print(f"Carregando checkpoint: {args.resume_path}")
        
        # Carrega o modelo completo (pesos + otimizador + estado interno)
        model = RecurrentPPO.load(
            args.resume_path, 
            env=vec_env, 
            device='cuda',
            # Garante que policy_kwargs sejam passados se necessário recriar partes
            custom_objects={'policy_kwargs': lstm_params['policy_kwargs']} 
        )
        # Força os hiperparâmetros (caso você queira mudar LR no meio do caminho)
        model.set_parameters(lstm_params) 
        
    else:
        print(f"\n>>> INICIANDO NOVO TREINO MARL (TRANSFER LEARNING) <<<")
        # Cria do zero
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            verbose=0,
            tensorboard_log=base_logdir,
            **lstm_params
        )
        # Transplanta o cérebro do PvE
        transfer_weights(args.pretrained_path, model, verbose=1)

    # --- 3. Vincula o Modelo ao Wrapper ---
    # O Wrapper precisa conhecer o modelo ATUAL para auto-jogar
    vec_env.envs[0].set_model(model) 

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=model_path, name_prefix="marl_model")
    logging_callback = LoggingCallback(verbose=1, log_interval=1)

    # --- 5. Treino ---
    print(f"Iniciando Treino por {args.total_timesteps} passos...")
    print(f"Logs: {tb_path}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([checkpoint_callback, logging_callback]),
            tb_log_name=run_name,
            progress_bar=True,
            reset_num_timesteps=False # Mantém a contagem do Tensorboard correta
        )
    except KeyboardInterrupt:
        print("Treinamento interrompido pelo usuário. Salvando...")

    # Salva o modelo final com o número de steps atual
    final_steps = model.num_timesteps
    save_name = f"{model_path}/final_marl_model_{final_steps}_steps.zip"
    model.save(save_name)
    print(f"Modelo final salvo em: {save_name}")
    
    # Salva o Hall da Fama
    logging_callback.save_hall_of_fame(os.path.join(tb_path, "hall_of_fame"))

    print("Encerrado.")

if __name__ == "__main__":
    main()