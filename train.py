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
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

# Imports locais
from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.logging_callbacks import LoggingCallback
from buriedbrains.wrappers import IslandWrapper, FlattenParallelWrapper

# ------------------------------------------------
# CONFIGURAÇÃO DE HIPERPARÂMETROS (RL TUNING)
# ------------------------------------------------
# Limite físico da GPU para batch size (depende da VRAM)
MAX_GPU_BATCH_SIZE = 32000 # diminui overhead de memória na GPU
N_CORES = 10  # Quantidade de cores para paralelismo

# Configurações de treino
DEFAULT_MAX_EP_STEPS = 15_000 
DEFAULT_SANCTUM_FLOOR = 20
DEFAULT_NUM_AGENTS = 64     # Padrão seguro
NORM_REWARD_CLIP = 10.0      
CHECKPOINT_FREQ_BASE = 200_000 

# Transformer
NET_EMBED_DIM = 96          # Rico em detalhes (Itens complexos)
NET_NUM_HEADS = 4           # 24 dim por cabeça
NET_DROPOUT = 0.1

# Policy / LSTM (Otimizada para memória)
NET_FEATURES_DIM = 256      # Reduz gargalo da LSTM
NET_ARCH_WIDTHS = [256, 256] 
LSTM_HIDDEN_SIZE = 256      

# Definição do vetor de observação (Shape: 198)
OBS_SIZES = {
    'skills': 40,
    'self_stats': 3,
    'pve_context': 7,
    'social_pvp': 7,
    'current_node': 13,
    'neighbor_n': 13,
    'neighbor_s': 13,
    'neighbor_e': 13,
    'neighbor_w': 13,
    'flags_gear': 8,
    'items': 68
}

# Configurações de Treino PPO
TRAIN_TOTAL_TIMESTEPS = 5_000_000
TRAIN_N_STEPS = 512                 # Tamanho do buffer por agente (Horizonte)
TRAIN_GAMMA = 0.99                  
TRAIN_GAE_LAMBDA = 0.95
TRAIN_VF_COEF = 0.5
TRAIN_MAX_GRAD_NORM = 0.5
TRAIN_N_EPOCHS = 10

# Tuning Dinâmico de Learning Rate
THRESHOLD_HIGH_AGENTS = 64
LR_HIGH_AGENTS = 1e-4      # Mais conservador para muitos agentes
ENTROPY_HIGH_AGENTS = 0.01

LR_LOW_AGENTS = 3e-4       # Mais agressivo para poucos agentes
ENTROPY_LOW_AGENTS = 0.03

# Logging
LOG_INTERVAL_TB = 5         
LOG_INTERVAL_HOF = 100      
HOF_TOP_N = 10

# Transformer-based Feature Extractor
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=NET_FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        
        self.embed_dim = NET_EMBED_DIM
        self.num_heads = NET_NUM_HEADS
        
        # Calcula os slices automaticamente baseado no dicionário OBS_SIZES        
        self.slices = []
        current_idx = 0
        
        ordered_keys = [
            'skills', 'self_stats', 'pve_context', 'social_pvp',
            'current_node', 'neighbor_n', 'neighbor_s', 'neighbor_e', 'neighbor_w',
            'flags_gear', 'items'
        ]
        
        self.block_projs = nn.ModuleList()
        
        for key in ordered_keys:
            size = OBS_SIZES[key]
            self.block_projs.append(nn.Linear(size, self.embed_dim))
            self.slices.append((current_idx, current_idx + size))
            current_idx += size
            
        self.num_tokens = len(ordered_keys)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=NET_DROPOUT,
            batch_first=True
        )
                 
        self.norm = nn.LayerNorm(self.embed_dim * self.num_tokens)
        self.linear_out = nn.Linear(self.embed_dim * self.num_tokens, features_dim)
        self.act = nn.ReLU() 

    def forward(self, obs):
        batch_size = obs.shape[0]

        raw_blocks = []
        for start, end in self.slices:
            raw_blocks.append(obs[:, start:end])

        projected_tokens = [proj(block) for proj, block in zip(self.block_projs, raw_blocks)]
        x_seq = torch.stack(projected_tokens, dim=1)
        attn_output, _ = self.attn(x_seq, x_seq, x_seq)
        x_seq = attn_output + x_seq

        x_flat = x_seq.reshape(batch_size, -1) 
        x_norm = self.norm(x_flat)
        
        return self.act(self.linear_out(x_norm))

# Função para ajustar hiperparâmetros dinamicamente
def get_dynamic_hyperparams(num_agents, n_steps=TRAIN_N_STEPS):
    total_buffer_size = num_agents * n_steps
        
    # Queremos que: total_buffer / n_minibatches <= MAX_GPU_BATCH_SIZE
    # Então: n_minibatches >= total_buffer / MAX_GPU_BATCH_SIZE
    
    # Calcula quantos pedaços são necessários
    n_minibatches = math.ceil(total_buffer_size / MAX_GPU_BATCH_SIZE)
    
    # Arredonda para a próxima potência de 2 para eficiência    
    n_minibatches = max(1, 2 ** math.ceil(math.log2(n_minibatches)))
    
    # O batch_size real que o SB3 vai usar por update
    actual_batch_size = total_buffer_size // n_minibatches    

    if num_agents >= THRESHOLD_HIGH_AGENTS:
        learning_rate = LR_HIGH_AGENTS  
        ent_coef = ENTROPY_HIGH_AGENTS       
    else:
        learning_rate = LR_LOW_AGENTS  
        ent_coef = ENTROPY_LOW_AGENTS       

    print(f"\n[AUTO-TUNING] Memória VRAM Otimizada:")
    print(f"  - Buffer Total (RAM): {total_buffer_size} steps")
    print(f"  - Divisor (Minibatches): {n_minibatches}")
    print(f"  - Carga na GPU (Batch Size): {actual_batch_size} (Meta: <= {MAX_GPU_BATCH_SIZE})")

    return actual_batch_size, learning_rate, ent_coef

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

def make_env(rank, seed_base, args, agents_per_core):
    """
    Cria uma 'Ilha' isolada. Apenas o rank 0 exibe logs detalhados (verbose).
    """
    def _init():
        # Se o usuário passou --verbose 2, apenas a Ilha 0 terá verbose 2.
        # As outras (rank > 0) ficam em 0 para não poluir o terminal.
        current_verbose = args.verbose if rank == 0 else 0
        
        env = BuriedBrainsEnv(
            max_episode_steps=args.max_episode_steps, 
            sanctum_floor=args.sanctum_floor,
            verbose=current_verbose, # <--- Agora dinâmico!
            num_agents=agents_per_core,
            seed=seed_base + rank,
            enable_logging_buffer=True 
        )
        return IslandWrapper(env)
    return _init

# Função Principal
def main():
    parser = argparse.ArgumentParser(description="Script Unificado de Treinamento (PPO vs LSTM)")
    
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'lstm'], 
                        help="Escolha o algoritmo: 'ppo' ou 'lstm'")
    
    # Defaults agora usam as CONSTANTES definidas no topo
    parser.add_argument('--total_timesteps', type=int, default=TRAIN_TOTAL_TIMESTEPS)
    parser.add_argument('--suffix', type=str, default="Experiment")
    parser.add_argument('--max_episode_steps', type=int, default=DEFAULT_MAX_EP_STEPS)
    parser.add_argument('--sanctum_floor', type=int, default=DEFAULT_SANCTUM_FLOOR)
    parser.add_argument('--num_agents', type=int, default=DEFAULT_NUM_AGENTS)
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
    vec_norm_path = os.path.join(model_path, "vec_normalize.pkl")

    # Inicialização do ambiente
    # Verifica divisão exata nos cores
    if args.num_agents % N_CORES != 0:
        raise ValueError(f"Erro: {args.num_agents} agentes não dividem igualmente em {N_CORES} núcleos.")
    
    agents_per_core = args.num_agents // N_CORES

    print(f"\n[PARALLEL] Iniciando {N_CORES} processos com {agents_per_core} agentes cada.")

    # Cria a lista de construtores
    env_fns = [make_env(i, seed_val if seed_val else 0, args, agents_per_core) for i in range(N_CORES)]
    
    # Inicia os processos
    parallel_env = SubprocVecEnv(env_fns)
    
    # Cola os resultados de volta
    env = FlattenParallelWrapper(parallel_env)

    # Normalização de Recompensa
    print(f"[ENV] Aplicando Normalização de Recompensa (Clip={NORM_REWARD_CLIP})...")
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=NORM_REWARD_CLIP)

    # Hiperparâmetros Dinâmicos
    auto_batch, auto_lr, auto_ent = get_dynamic_hyperparams(args.num_agents, n_steps=TRAIN_N_STEPS)

    print(f"\n[AUTO-TUNING] Configuração para {args.num_agents} Agentes:")
    print(f"  - Batch Size: {auto_batch}")
    print(f"  - Learning Rate: {auto_lr}")
    print(f"  - Entropy Coef: {auto_ent}")

    common_params = {
        "learning_rate": auto_lr,
        "n_steps": TRAIN_N_STEPS,         
        "batch_size": auto_batch,   
        "n_epochs": TRAIN_N_EPOCHS,
        "gamma": TRAIN_GAMMA,     
        "gae_lambda": TRAIN_GAE_LAMBDA,
        "ent_coef": auto_ent,       
        "vf_coef": TRAIN_VF_COEF,
        "max_grad_norm": TRAIN_MAX_GRAD_NORM,
    }

    # Configuração da Rede Neural usando CONSTANTES
    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": NET_FEATURES_DIM},
        "net_arch": dict(pi=NET_ARCH_WIDTHS, vf=NET_ARCH_WIDTHS), 
    }

    if args.algo == 'lstm':
        policy_kwargs["lstm_hidden_size"] = LSTM_HIDDEN_SIZE
        policy_kwargs["enable_critic_lstm"] = True 
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
        
        resume_dir = os.path.dirname(args.resume_path)
        stats_path = os.path.join(resume_dir, "vec_normalize.pkl")
        
        if os.path.exists(stats_path):
            print(f"[LOAD] Carregando estatísticas de normalização: {stats_path}")
            env = VecNormalize.load(stats_path, env)
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
    
    # Callback customizado para salvar o VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self, save_path, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
        def _on_step(self) -> bool:
            if self.n_calls % save_freq == 0:
                self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
            return True

    # Calcula frequência de save
    save_freq = max(1, CHECKPOINT_FREQ_BASE // env.num_envs)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq, 
        save_path=model_path, 
        name_prefix=f"{args.algo}_model"
    )
    
    vec_norm_callback = SaveVecNormalizeCallback(model_path)
    
    logging_callback = LoggingCallback(
        log_interval=LOG_INTERVAL_TB,       
        hof_save_interval=LOG_INTERVAL_HOF, 
        top_n=HOF_TOP_N,                    
        enable_hall_of_fame=not args.no_hall_of_fame
    )

    print(f"\n>>> INICIANDO TREINO: {algo_name} <<<")
    print(f"Steps Totais: {args.total_timesteps}")
    print(f"Log Dir: {tb_path}")
    print(f"Checkpoints a cada: {save_freq} steps (por agente)")
    
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
    env.save(vec_norm_path)
    
    print(f"\n[DONE] Modelo salvo em: {full_save_path}")
    print(f"[DONE] Estatísticas de normalização salvas em: {vec_norm_path}")
    
    if not args.no_hall_of_fame:
        hof_path = os.path.join(tb_path, "hall_of_fame")
        os.makedirs(hof_path, exist_ok=True)
        logging_callback.save_hall_of_fame(hof_path)

if __name__ == "__main__":
    main()