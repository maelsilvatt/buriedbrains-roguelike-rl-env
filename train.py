# train.py
import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

# Imports locais
from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import SharedPolicyVecEnv

# Arquitetura do AutoEncoder (Self-Attention)
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Extrator de características que usa Self-Attention para filtrar e
    relacionar os 172 inputs antes de passá-los para a política (PPO ou LSTM).
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0] # 172
        
        # Projeção Linear
        self.linear_in = nn.Linear(input_dim, 256)
        self.act = nn.ReLU()
        
        # Configuração do Self-Attention
        # Transformamos o vetor de 256 em 16 tokens de dimensão 16
        self.num_heads = 4
        self.embed_dim = 16
        self.seq_len = 16
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            batch_first=True
        )
        
        # Normalização e saída
        self.layer_norm = nn.LayerNorm(256)
        self.linear_out = nn.Linear(256, features_dim)

    def forward(self, observations):
        # Projeção linear das features
        x = self.act(self.linear_in(observations))
        
        # Tokenização virtual (reshape para sequência)
        batch_size = x.shape[0]
        x_seq = x.view(batch_size, self.seq_len, self.embed_dim)
        
        # Self-Attention 
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        
        # Flatten e Conexão Residual
        x_flat = attn_output.reshape(batch_size, -1)
        x_final = self.layer_norm(x + x_flat)
        
        # Saída
        return self.linear_out(x_final)

def transfer_weights(old_model_path, new_model, verbose=0):
    """
    Transfere pesos de um modelo para outro (ex: PvE Expert -> MARL).
    Funciona melhor se as arquiteturas forem compatíveis.
    """
    if verbose >= 1:
        print(f"\n--- INICIANDO TRANSFER LEARNING ---")
        print(f"Carregando pesos de: {old_model_path}")

    # Tenta carregar genericamente (PPO ou RecurrentPPO)
    try:
        temp_model = RecurrentPPO.load(old_model_path, device='cpu')
    except:
        temp_model = PPO.load(old_model_path, device='cpu')

    old_state_dict = temp_model.policy.state_dict()
    new_state_dict = new_model.policy.state_dict()

    with torch.no_grad():
        for key in new_state_dict.keys():
            if key in old_state_dict:
                old_param = old_state_dict[key]
                new_param = new_state_dict[key]
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                elif len(old_param.shape) == 2 and old_param.shape[0] == new_param.shape[0]:
                    # Lógica para shapes parciais (se necessário)
                    input_size_old = old_param.shape[1]
                    input_size_new = new_param.shape[1]
                    if input_size_old < input_size_new:
                        new_param[:, :input_size_old].copy_(old_param)

    if verbose >= 1: print("--- TRANSFERÊNCIA CONCLUÍDA ---\n")
    del temp_model

def main():
    parser = argparse.ArgumentParser(description="Script Unificado de Treinamento (PPO vs LSTM)")
    
    # Seletor de algoritmo
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'lstm'], 
                        help="Escolha o algoritmo: 'ppo' (Baseline sem memória) ou 'lstm' (Com memória)")
    
    # Argumentos do parser
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    parser.add_argument('--suffix', type=str, default="Experiment")
    parser.add_argument('--max_episode_steps', type=int, default=50_000)
    parser.add_argument('--sanctum_floor', type=int, default=20)
    parser.add_argument('--num_agents', type=int, default=16)
    parser.add_argument('--seed', nargs='?', const='random', default='random')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--no_hall_of_fame', action='store_true')
    parser.add_argument('--verbose', type=int, default=0)
    
    args = parser.parse_args()

    # Configuração de Seed
    seed_val = None
    if args.seed is not None and str(args.seed).lower() != "random":
        try:
            seed_val = int(args.seed)
            print(f"\nModo determinístico: seed global {seed_val}")
            set_random_seed(seed_val)
        except ValueError:
            print("Seed inválida. Usando aleatória.")
    
    # Configuração de caminhos
    algo_name = args.algo.upper() # "PPO" ou "LSTM"
    run_name = f"{algo_name}_{args.suffix}"
    
    base_logdir = "logs_marl"
    base_models_dir = "models_marl"
    
    tb_path = os.path.join(base_logdir, run_name)
    model_path = os.path.join(base_models_dir, run_name)

    # Inicialização do ambiente
    print(f"\nInicializando BuriedBrains (com {args.num_agents} Agentes)...")
    base_env = BuriedBrainsEnv(
        max_episode_steps=args.max_episode_steps, 
        sanctum_floor=args.sanctum_floor,
        verbose=args.verbose,
        num_agents=args.num_agents,
        seed=seed_val
    )
    env = SharedPolicyVecEnv(base_env)

    # Hiperparâmetros 
    # Ambos os algoritmos usarão a mesma base para comparação justa
    common_params = {
        "learning_rate": 3e-4,
        "n_steps": 512,         # Horizonte de coleta
        "batch_size": 2048,     # Tamanho do lote de treino
        "n_epochs": 10,
        "gamma": 0.995,         # Foco no longo prazo (social)
        "gae_lambda": 0.95,
        "ent_coef": 0.03,       # Exploração alta
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # Definição da Arquitetura Neural com self attention
    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 256], vf=[256, 256]), # MLP pós-atenção
    }

    # Ajustes específicos do LSTM
    if args.algo == 'lstm':
        policy_kwargs["lstm_hidden_size"] = 256
        policy_kwargs["enable_critic_lstm"] = False
        policy_name = "MlpLstmPolicy"
        ModelClass = RecurrentPPO
    else:
        # PPO Padrão (Sem LSTM)
        policy_name = "MlpPolicy"
        ModelClass = PPO

    # Injeta kwargs finais
    common_params["policy_kwargs"] = policy_kwargs

    # Inicialização do modelo
    model = None
    
    # Resumir treino anterior
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"\n[LOAD] Carregando checkpoint: {args.resume_path}")
        model = ModelClass.load(
            args.resume_path, 
            env=env,
            device='cuda',
            custom_objects=common_params # Força os novos hiperparâmetros
        )
    # Novo treino do zero
    else:
        print(f"\n[INIT] Criando novo agente {algo_name} com Attention...")
        model = ModelClass(
            policy_name,
            env,
            verbose=0,
            tensorboard_log=base_logdir,
            seed=seed_val,
            **common_params
        )
        
        if args.pretrained_path:
            transfer_weights(args.pretrained_path, model, verbose=1)

    # Callbacks
    save_freq = max(1, 200_000 // env.num_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=model_path, 
        name_prefix=f"{args.algo}_model"
    )
    logging_callback = LoggingCallback(
        verbose=0, 
        log_interval=1, 
        enable_hall_of_fame=not args.no_hall_of_fame
    )

    # Loop de Treinamento
    print(f"\n>>> INICIANDO TREINO: {algo_name} <<<")
    print(f"Steps Totais: {args.total_timesteps}")
    print(f"Log Dir: {tb_path}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList([checkpoint_callback, logging_callback]),
            tb_log_name=run_name,
            progress_bar=True,
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\n[STOP] Interrompido pelo usuário. Salvando...")

    # Salvamento Final
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    
    final_steps = model.num_timesteps
    save_filename = f"final_{args.algo}_model_{final_steps}_steps.zip"
    full_save_path = os.path.join(model_path, save_filename)
    
    model.save(full_save_path)
    print(f"\n[DONE] Modelo salvo em: {full_save_path}")
    
    # Hall of Fame
    if not args.no_hall_of_fame:
        hof_path = os.path.join(tb_path, "hall_of_fame")
        os.makedirs(hof_path, exist_ok=True)
        logging_callback.save_hall_of_fame(hof_path)

if __name__ == "__main__":
    main()