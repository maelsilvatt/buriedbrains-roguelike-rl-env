# optimize.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_FILESYSTEM"] = "1"

import time
import argparse
import optuna
import torch
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Ambiente personalizado
from buriedbrains.env import BuriedBrainsEnv

# --- Constantes de Otimização ---
# Ajuste conforme necessário - mais passos dão uma avaliação melhor, mas demora mais.
N_TRAINING_STEPS_PER_TRIAL = 200_000
# Número de episódios para AVALIAR cada modelo treinado
N_EVAL_EPISODES = 10
# Número total de combinações de parâmetros a testar
N_TRIALS = 50 
# Número de trabalhos paralelos (1 é mais seguro para GPU)
N_JOBS = 1

# --- Definição da Função Objetivo (Core do Optuna) ---

def objective(trial, model_type='ppo', env_kwargs=None):
    """
    Função que o Optuna chama para testar uma combinação de hiperparâmetros.

    :param trial: Objeto 'trial' do Optuna, usado para sugerir parâmetros.
    :param model_type: String 'ppo' ou 'lstm' para escolher o modelo.
    :param env_kwargs: Dicionário com argumentos para o BuriedBrainsEnv (opcional).
    :return: Métrica de desempenho (ex: recompensa média) a ser otimizada.
    """
    if env_kwargs is None:
        env_kwargs = {} # Usa defaults do ambiente se não especificado

    print(f"\n--- Iniciando Trial {trial.number} ({model_type.upper()}) ---")
    start_time = time.time()

    # 1. Sugerir Hiperparâmetros    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 1.0])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    vf_coef = trial.suggest_categorical("vf_coef", [0.4, 0.5, 0.6])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    # Garante que batch_size <= n_steps
    if batch_size > n_steps:
        batch_size = n_steps

    # Arquitetura da Rede (policy_kwargs)    
    net_arch_size = trial.suggest_categorical("net_arch_size", [64, 128, 256])
    policy_kwargs = dict(net_arch=dict(pi=[net_arch_size, net_arch_size], vf=[net_arch_size, net_arch_size]))

    # Parâmetros Específicos do LSTM
    if model_type == 'lstm':
        policy_kwargs['lstm_hidden_size'] = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        # policy_kwargs['n_lstm_layers'] = trial.suggest_categorical("n_lstm_layers", [1, 2]) # Adicionar se quiser testar
        policy_kwargs['enable_critic_lstm'] = trial.suggest_categorical("enable_critic_lstm", [True, False])

    # Seleciona a Classe do Modelo e Política
    if model_type == 'lstm':
        model_class = RecurrentPPO
        policy = "MlpLstmPolicy"
    else: # PPO Padrão
        model_class = PPO
        policy = "MlpPolicy"

    # 2. Criar o Ambiente Vetorizado
    #    Usamos make_vec_env para facilitar, mas DummyVecEnv funciona igual
    #    n_envs=1 pois a avaliação é sequencial aqui.
    vec_env = make_vec_env(lambda: BuriedBrainsEnv(**env_kwargs), n_envs=1, vec_env_cls=DummyVecEnv)

    # 3. Criar e Treinar o Modelo
    try:
        model = model_class(
            policy,
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None, # Desabilitar logs do tensorboard para HPO
            verbose=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model.learn(total_timesteps=N_TRAINING_STEPS_PER_TRIAL, progress_bar=False) # Sem progress bar para não poluir

        # 4. Avaliar o Modelo
        #    evaluate_policy roda o modelo em modo determinístico por padrão
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=N_EVAL_EPISODES)

        print(f"Trial {trial.number} Concluído. Recompensa Média: {mean_reward:.2f} +/- {std_reward:.2f}")

    except Exception as e:
        print(f"Trial {trial.number} FALHOU: {e}")
        # Penaliza trials que falham (ex: OOM error, erro de cálculo)
        # Pode retornar um valor muito baixo ou optuna.TrialPruned()
        mean_reward = -float('inf') # Ou outro valor baixo apropriado

    finally:
        # 5. Limpeza (Importante para liberar memória da GPU)
        if 'model' in locals():
            del model
        if 'vec_env' in locals():
            vec_env.close()
        torch.cuda.empty_cache() # Tenta limpar cache da GPU

    end_time = time.time()
    print(f"Duração do Trial {trial.number}: {end_time - start_time:.1f} segundos")

    # 6. Retornar a Métrica
    return mean_reward

# --- Bloco Principal de Execução ---

if __name__ == "__main__":
    # Argumentos de linha de comando para escolher qual modelo otimizar
    parser = argparse.ArgumentParser(description="Otimização de Hiperparâmetros para BuriedBrains")
    parser.add_argument('model_type', type=str, choices=['ppo', 'lstm'], help="Tipo de modelo a otimizar ('ppo' ou 'lstm')")
    parser.add_argument('--study_name', type=str, default=None, help="Nome para o estudo Optuna (útil para retomar)")
    parser.add_argument('--n_trials', type=int, default=N_TRIALS, help="Número de trials a executar")

    # Adicione argumentos para o ambiente se quiser testar HPO em diferentes dificuldades
    parser.add_argument('--max_episode_steps', type=int, default=30000, help="Limite de passos por episódio no ambiente")
    parser.add_argument('--budget_multiplier', type=float, default=1.0, help="Multiplicador de dificuldade (budget) do ambiente")

    args = parser.parse_args()

    # Cria ou carrega um estudo Optuna
    # Usar um banco de dados permite pausar e retomar a otimização (boa prática)
    study_name = args.study_name if args.study_name else f"{args.model_type}_buriedbrains_study"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name, # Salva o progresso
        load_if_exists=True,  # Permite retomar estudo interrompido
        direction="maximize", # Queremos maximizar a recompensa média
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), # Opcional: Corta trials ruins cedo
    )

    # Prepara os argumentos do ambiente
    env_kwargs_for_opt = {
        'max_episode_steps': args.max_episode_steps,
        'budget_multiplier': args.budget_multiplier
        # Adicione outros parâmetros do __init__ do seu Env se necessário
    }

    print(f"Iniciando otimização para: {args.model_type.upper()}")
    print(f"Número de Trials: {args.n_trials}")
    print(f"Ambiente: max_steps={env_kwargs_for_opt['max_episode_steps']}, budget_mult={env_kwargs_for_opt['budget_multiplier']}")
    print(f"Estudo Optuna: {study_name} (Salvo em {storage_name})")

    try:
        # Executa a otimização
        study.optimize(
            lambda trial: objective(trial, model_type=args.model_type, env_kwargs=env_kwargs_for_opt),
            n_trials=args.n_trials,
            n_jobs=N_JOBS, # Use n_jobs=1 se tiver problemas de memória GPU
            timeout=None # define um tempo limite em segundos (ex: 3600*6 para 6 horas)
        )
    except KeyboardInterrupt:
        print("Otimização interrompida pelo usuário.")

    # Exibe os resultados
    print("\n--- Otimização Concluída ---")
    print("Número de trials finalizados: ", len(study.trials))

    best_trial = study.best_trial
    print("Melhor Trial:")
    print("  Valor (Recompensa Média): ", best_trial.value)
    print("  Melhores Hiperparâmetros: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Para visualizar os resultados com optuna-dashboard (instale via pip)
    # Executar no terminal: optuna-dashboard sqlite:///nome_do_estudo.db