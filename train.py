# train.py
import os
import argparse
import torch
import torch.nn as nn
import math
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize 

# Imports locais
from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import SharedPolicyVecEnv

# Arquitetura do AutoEncoder 
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Configurações da Atenção
        self.embed_dim = 64  # Tamanho de cada "gaveta" de informação
        self.num_tokens = 10 # 10 blocos lógicos do ambiente
        self.num_heads = 4   # 4 Cabeças de atenção (64 / 4 = 16 por cabeça)
        
        # Projeções semânticas
        # Cada camada linear pega uma fatia bruta do vetor 172 e transforma em 64
        self.block_projs = nn.ModuleList([
            nn.Linear(40, self.embed_dim),   # Token 0: Skills (4 slots * 10 features)
            nn.Linear(3, self.embed_dim),    # Token 1: Self Stats (HP, Lvl, XP)
            nn.Linear(7, self.embed_dim),    # Token 2: Contexto PvE (Inimigos, Eventos)
            nn.Linear(10, self.embed_dim),   # Token 3: Social / PvP (Arena, Outro Agente)
            nn.Linear(13, self.embed_dim),   # Token 4: Vizinho 0 (Norte)
            nn.Linear(13, self.embed_dim),   # Token 5: Vizinho 1 (Sul)
            nn.Linear(13, self.embed_dim),   # Token 6: Vizinho 2 (Leste)
            nn.Linear(13, self.embed_dim),   # Token 7: Vizinho 3 (Oeste)
            nn.Linear(6, self.embed_dim),    # Token 8: Gear/Portas (Raridades, Flags)
            nn.Linear(54, self.embed_dim),   # Token 9: Detalhes de Itens (Embeddings profundos)
        ])

        # --- 2. Mecanismo de Atenção ---
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1, # Evita overfitting precoce
            batch_first=True # Importante: (Batch, Seq, Feature)
        )
        
        # Normalização para estabilizar o gradiente
        self.norm = nn.LayerNorm(self.embed_dim * self.num_tokens)
        
        # Projeção final para conectar com a LSTM do SB3 (640 -> 256)
        self.linear_out = nn.Linear(self.embed_dim * self.num_tokens, features_dim)
        self.act = nn.ReLU() 

    def forward(self, obs):
        batch_size = obs.shape[0]

        # Fatiar o vetor de observação (Slicing baseado no env.py)
        # OBS: Se mudar o env.py, precisa ajustar esses índices
        raw_blocks = [
            obs[:, 0:40],     # Skills
            obs[:, 40:43],    # Self
            obs[:, 43:50],    # PvE
            obs[:, 50:60],    # Social
            obs[:, 60:73],    # Vizinho 0
            obs[:, 73:86],    # Vizinho 1
            obs[:, 86:99],    # Vizinho 2
            obs[:, 99:112],   # Vizinho 3
            obs[:, 112:118],  # Gear
            obs[:, 118:172],  # Items
        ]

        # Projetar cada bloco para ter o mesmo tamanho (64)
        # Resultado: Lista de 10 tensores de shape (Batch, 64)
        projected_tokens = [proj(block) for proj, block in zip(self.block_projs, raw_blocks)]

        # Empilhar para criar a sequência temporal
        # Shape final: (Batch, 10, 64) -> "Uma frase com 10 palavras de tamanho 64"
        x_seq = torch.stack(projected_tokens, dim=1)

        # Auto-Atenção (Onde a mágica acontece)
        # O modelo aprende: "Dado que 'Self' está com HP baixo, olhe para 'Items'"
        attn_output, _ = self.attn(x_seq, x_seq, x_seq)
        attn_output = attn_output + x_seq


        # Achatamento e Saída
        # (Batch, 10, 64) -> (Batch, 640)
        x_flat = attn_output.reshape(batch_size, -1)
        
        # Normaliza e projeta para 256 (para a LSTM)
        x_norm = self.norm(x_flat)
        return self.act(self.linear_out(x_norm))

def get_dynamic_hyperparams(num_agents, n_steps=512):
    total_buffer_size = num_agents * n_steps
    target_minibatches = 8 
    ideal_batch_size = total_buffer_size // target_minibatches
    batch_size = 2 ** round(math.log2(max(256, ideal_batch_size)))
    if batch_size > total_buffer_size:
        batch_size = int(2 ** math.floor(math.log2(total_buffer_size)))

    if num_agents >= 64:
        learning_rate = 1e-4  
        ent_coef = 0.01       
    else:
        learning_rate = 3e-4  
        ent_coef = 0.03       

    return batch_size, learning_rate, ent_coef

def transfer_weights(old_model_path, new_model, verbose=0):
    if verbose >= 1:
        print(f"\n--- INICIANDO TRANSFER LEARNING ---")
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
                        help="Escolha o algoritmo: 'ppo' ou 'lstm'")
    
    # Argumentos Gerais
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

    # Definição da seed
    seed_val = None
    if args.seed is not None and str(args.seed).lower() != "random":
        try:
            seed_val = int(args.seed)
            print(f"\n[CONFIG] Modo Determinístico: Seed {seed_val}")
            set_random_seed(seed_val)
        except ValueError:
            print("[WARN] Seed inválida. Usando aleatória.")
    
    # Caminho
    algo_name = args.algo.upper() 
    run_name = f"{algo_name}_{args.suffix}"
    
    base_logdir = "logs"
    base_models_dir = "models"
    
    tb_path = os.path.join(base_logdir, run_name)
    model_path = os.path.join(base_models_dir, run_name)
    
    # Caminho para salvar/carregar as estatísticas de normalização
    vec_norm_path = os.path.join(model_path, "vec_normalize.pkl")

    # Inicialização do ambiente
    print(f"\n[ENV] Inicializando BuriedBrains ({args.num_agents} Agentes)...")
    base_env = BuriedBrainsEnv(
        max_episode_steps=args.max_episode_steps, 
        sanctum_floor=args.sanctum_floor,
        verbose=args.verbose,
        num_agents=args.num_agents,
        seed=seed_val
    )
    env = SharedPolicyVecEnv(base_env)

    # Envolvendo com VecNormalize para acabar com o "Dente de Serra"
    # norm_obs=False: a AttentionNet já lida com os dados brutos/flags. Não queremos estragar os bools.
    # norm_reward=True: ESSENCIAL. Transforma -300/+400 em uma escala digerível (ex: -2 a +2).
    # clip_reward=10.0: Segurança contra exploding gradients.
    print("[ENV] Aplicando Normalização de Recompensa (VecNormalize)...")
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Hiperparâmetros Dinâmicos
    auto_batch, auto_lr, auto_ent = get_dynamic_hyperparams(args.num_agents, n_steps=512)

    print(f"\n[AUTO-TUNING] Configuração para {args.num_agents} Agentes:")
    print(f"  - Batch Size: {auto_batch}")
    print(f"  - Learning Rate: {auto_lr}")
    print(f"  - Entropy Coef: {auto_ent}")

    common_params = {
        "learning_rate": auto_lr,
        "n_steps": 512,         
        "batch_size": auto_batch,   
        "n_epochs": 10,
        "gamma": 0.99, # 995 para ~200 passos. 99 para 100 passos.     
        "gae_lambda": 0.95,
        "ent_coef": auto_ent,       
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 256], vf=[256, 256]), 
    }

    if args.algo == 'lstm':
        policy_kwargs["lstm_hidden_size"] = 256
        policy_kwargs["enable_critic_lstm"] = True # Essencial para que o Crítico veja a história
        policy_name = "MlpLstmPolicy"
        ModelClass = RecurrentPPO
    else:
        policy_name = "MlpPolicy"
        ModelClass = PPO

    common_params["policy_kwargs"] = policy_kwargs

    # Inicialização do Modelo
    model = None
    
    # Resumir Treino
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"\n[LOAD] Carregando checkpoint: {args.resume_path}")
        
        # Carregar estatísticas do VecNormalize se existirem        
        resume_dir = os.path.dirname(args.resume_path)
        stats_path = os.path.join(resume_dir, "vec_normalize.pkl")
        
        if os.path.exists(stats_path):
            print(f"[LOAD] Carregando estatísticas de normalização: {stats_path}")
            env = VecNormalize.load(stats_path, env)
            # Garante que continua treinando e normalizando
            env.training = True 
            env.norm_reward = True
        else:
            print("[WARN] Arquivo vec_normalize.pkl não encontrado! O treino pode ficar instável.")

        model = ModelClass.load(
            args.resume_path, 
            env=env,
            device='cuda',
            custom_objects=common_params 
        )
    # Novo Treino
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
    
    # Callback customizado para salvar o VecNormalize periodicamente
    from stable_baselines3.common.callbacks import BaseCallback
    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self, save_path, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
        def _on_step(self) -> bool:
            if self.n_calls % save_freq == 0:
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            return True

    save_freq = max(1, 200_000 // env.num_envs)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=model_path, 
        name_prefix=f"{args.algo}_model"
    )
    
    # Salva as estatísticas do ambiente junto com o modelo
    vec_norm_callback = SaveVecNormalizeCallback(model_path)
    
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
            callback=CallbackList([checkpoint_callback, logging_callback, vec_norm_callback]),
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
    
    # Salva o arquivo de normalização final
    env.save(vec_norm_path)
    print(f"\n[DONE] Modelo salvo em: {full_save_path}")
    print(f"[DONE] Estatísticas de normalização salvas em: {vec_norm_path}")
    
    # Salva ou não o hall da fama
    if not args.no_hall_of_fame:
        hof_path = os.path.join(tb_path, "hall_of_fame")
        os.makedirs(hof_path, exist_ok=True)
        logging_callback.save_hall_of_fame(hof_path)

if __name__ == "__main__":
    main()