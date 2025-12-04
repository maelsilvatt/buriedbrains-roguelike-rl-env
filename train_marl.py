# train_marl.py
import os
import argparse
import torch
import torch.nn as nn
import random
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import SharedPolicyVecEnv 
from stable_baselines3.common.utils import set_random_seed

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
        
    # Argumentos de Modelo
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    parser.add_argument('--pretrained_path', type=str, default=None, help="Caminho do modelo PvE Expert (para começar do zero)")
    parser.add_argument('--resume_path', type=str, default=None, help="Caminho de um checkpoint MARL (para continuar treino)")

    parser.add_argument('--suffix', type=str, default="MARL_Shared")
    parser.add_argument('--max_episode_steps', type=int, default=10_000)
    parser.add_argument('--sanctum_floor', type=int, default=20)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--seed', nargs='?', const='random', default='random', help="Seed fixa ou 'random'")
    parser.add_argument('--num_agents', type=int, default=2)
    args = parser.parse_args()

    # Lógica da Seed
    seed_val = None  

    if args.seed is not None and str(args.seed).lower() != "random":
        try:
            seed_val = int(args.seed)
            print(f"\n>>> MODO DETERMINÍSTICO: Seed Global fixada em {seed_val} <<<")
            
            # Fixa a seed globalmente
            set_random_seed(seed_val) 
            
        except ValueError:
            print(f"Aviso: Valor de seed '{args.seed}' inválido. Usando modo aleatório.")
    else:
        print("\nMODO ALEATÓRIO: Nenhuma seed fixa definida.")

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
        seed=seed_val
    ) 
    
    # Aplica o wrapper para Shared Policy
    env = SharedPolicyVecEnv(base_env) 

    # Ajustes dinâmicos baseados no número de agentes
    na = args.num_agents

    # Defaults seguros
    lstm_hidden_size = 128
    n_steps = 512
    batch_size = 256
    enable_critic_lstm = False
    net_arch = {"pi": [128, 128], "vf": [128, 128]}

    # 1 - Quase nenhum agente → pode usar rede maior
    if na <= 4:
        lstm_hidden_size = 128
        n_steps = 512
        batch_size = 128
        net_arch = {"pi": [64, 64], "vf": [64, 64]}

    elif na <= 16:
        lstm_hidden_size = 128
        n_steps = 512
        batch_size = 256
        net_arch = {"pi": [128, 128], "vf": [128, 128]}

    # 2 - Média escala (32–128 agentes)
    elif na <= 128:
        lstm_hidden_size = 64
        n_steps = 256
        batch_size = 256
        net_arch = {"pi": [128, 128], "vf": [128, 128]}

    # 3 - Grande escala (128–1000 agentes)
    elif na <= 1000:
        lstm_hidden_size = 48
        n_steps = 128
        batch_size = 128
        net_arch = {"pi": [64, 64], "vf": [64, 64]}

    # 4 - Enorme escala (1000+ agentes)
    else:
        lstm_hidden_size = 32
        n_steps = 64
        batch_size = 64
        enable_critic_lstm = False
        net_arch = {"pi": [32, 32], "vf": [32, 32]}

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
    model = None

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"\nResumindo treinamento MARL...")
        print(f"Carregando checkpoint: {args.resume_path}")

        model = RecurrentPPO.load(
            args.resume_path, 
            env=env,
            device='cuda'
        )

    else:
        if not args.pretrained_path:
            raise ValueError("Para iniciar um treino novo você deve fornecer --pretrained_path.")

        print(f"\nIniciando novo treino MARL")

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=0,
            tensorboard_log=base_logdir,
            seed=args.seed,
            **lstm_params
        )

        # Transferência dos pesos PvE → PvP Social
        transfer_weights(args.pretrained_path, model, verbose=1)

    # --- 4. Callbacks ---
    save_freq = max(1, 200_000 // env.num_envs) # Como estamos usando múltiplos ambientes paralelos
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_path, name_prefix="marl_model")
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

    # 1. Garante que a pasta existe antes de tentar salvar
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print(f"Diretório criado: {model_path}")
    
    final_steps = model.num_timesteps
    save_filename = f"final_marl_model_{final_steps}_steps.zip"
    full_save_path = os.path.join(model_path, save_filename)
    
    # Salva
    model.save(full_save_path)
    print(f"Modelo final salvo com sucesso em: {full_save_path}")
    
    # Salva o Hall da Fama
    hof_path = os.path.join(tb_path, "hall_of_fame")
    if not os.path.exists(hof_path):
        os.makedirs(hof_path, exist_ok=True)
        
    logging_callback.save_hall_of_fame(hof_path)

    print("Encerrado.")

if __name__ == "__main__":
    main()