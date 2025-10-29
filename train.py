# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silencia logs chatos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # evita warning do oneDNN
os.environ["TF_USE_LEGACY_FILESYSTEM"] = "1"  # evita bug de verificação no Windows

import time
import gymnasium as gym
import numpy as np
import torch
import argparse # para automação

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Importa o ambiente personalizado
from buriedbrains.env import BuriedBrainsEnv

class LoggingCallback(BaseCallback):
    """
    Callback customizado para registrar métricas detalhadas do BuriedBrains
    E salvar as "melhores histórias" (Hall da Fama).
    """
    def __init__(self, log_interval: int = 10, verbose: int = 1, top_n: int = 10):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.top_n = top_n # Quantas histórias "top" salvar

        # Listas para guardar dados do intervalo (para o TensorBoard)
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_levels = []
        self.episode_floors = []
        self.episode_lengths = []
        self.episode_enemies_defeated = []
        self.episode_invalid_actions = []
        self.episode_total_actions = []
        
        # --- ETAPA 3: O HALL DA FAMA ---
        self.hall_of_fame_level = [] # Top N por Nível de EXP
        self.hall_of_fame_floor = [] # Top N por Andar Alcançado
        self.hall_of_fame_enemies = [] # Top N por Inimigos Mortos
        # --- FIM DA ETAPA 3 ---

        self.episode_count = 0
        self.max_floor_ever = 0

    def _update_hall_of_fame(self, story: dict, hall_of_fame: list, metric_key: str):
            """
            Adiciona a história a uma lista do hall da fama se ela for boa o suficiente,
            mantendo a lista ordenada e com tamanho 'top_n'.
            Versão otimizada.
            """
            
            # Pega a pontuação da história atual
            new_score = story.get(metric_key, 0)

            # Caso 1: A lista ainda não está cheia.
            if len(hall_of_fame) < self.top_n:
                hall_of_fame.append(story)
                # Reordena (rápido com < top_n itens)
                hall_of_fame.sort(key=lambda s: s[metric_key], reverse=True)
                return # Adicionado e pronto

            # Caso 2: A lista está cheia. Precisamos verificar se a nova pontuação é melhor que a pior.
            # Como a lista está sempre ordenada, o pior está na última posição.
            worst_score = hall_of_fame[-1].get(metric_key, 0)
            
            if new_score > worst_score:
                # A nova história merece um lugar!
                hall_of_fame.pop() # Remove o pior
                hall_of_fame.append(story) # Adiciona o novo
                hall_of_fame.sort(key=lambda s: s[metric_key], reverse=True) # Reordena

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            # --- Coleta de Dados do Episódio Finalizado ---
            self.episode_count += 1
            info = self.locals["infos"][i]
            final_info = info.get("final_info", info)
            
            if not isinstance(final_info, dict):
                 if self.verbose > 1: print("[Callback WARN] Não foi possível encontrar final_info.")
                 continue 

            final_status = final_info.get("final_status")
            if not final_status:
                 if self.verbose > 1: print("[Callback WARN] 'final_status' não encontrado em final_info.")
                 continue 

            # Coleta de métricas para o TensorBoard (como antes)
            self.episode_rewards.append(self.locals["rewards"][i])
            self.episode_wins.append(1 if final_status.get("win") else 0)
            self.episode_levels.append(final_status.get("level", 0))
            current_floor = final_status.get("floor", 0)
            self.episode_floors.append(current_floor)
            self.episode_lengths.append(final_status.get("steps", 0))
            self.episode_enemies_defeated.append(final_status.get("enemies_defeated", 0))
            self.episode_invalid_actions.append(final_status.get("invalid_actions", 0))
            self.episode_total_actions.append(final_status.get("steps", 1))
            self.max_floor_ever = max(self.max_floor_ever, current_floor)

            # --- ETAPA 3: COLETA E FILTRAGEM DA HISTÓRIA ---
            agent_name = final_status.get('agent_name', 'Agente_Desconhecido')
            full_log = final_status.get('full_log', ['Log não capturado.'])
            
            # Cria o objeto da "história"
            story = {
                'agent_name': agent_name,
                'level': final_status.get('level', 0),
                'floor': current_floor,
                'enemies_defeated': final_status.get('enemies_defeated', 0),
                'log_content': full_log # A gravação completa
            }
            
            # Tenta adicionar a história a cada Hall da Fama
            self._update_hall_of_fame(story, self.hall_of_fame_level, 'level')
            self._update_hall_of_fame(story, self.hall_of_fame_floor, 'floor')
            self._update_hall_of_fame(story, self.hall_of_fame_enemies, 'enemies_defeated')
            # --- FIM DA ETAPA 3 ---

            # --- Loga Médias a Cada `log_interval` Episódios (para o TensorBoard) ---
            if self.episode_count % self.log_interval == 0:
                # ... (cálculo de médias: mean_reward, win_rate, avg_level, etc.) ...
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                win_rate = np.mean(self.episode_wins) if self.episode_wins else 0
                avg_level = np.mean(self.episode_levels) if self.episode_levels else 0
                avg_floor = np.mean(self.episode_floors) if self.episode_floors else 0
                max_floor_interval = np.max(self.episode_floors) if self.episode_floors else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                avg_enemies = np.mean(self.episode_enemies_defeated) if self.episode_enemies_defeated else 0
                total_invalid = sum(self.episode_invalid_actions)
                total_actions = sum(self.episode_total_actions)
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0

                # Loga para o TensorBoard
                self.logger.record("rollout/ep_rew_mean", mean_reward)
                self.logger.record("custom/win_rate", win_rate)
                self.logger.record("custom/avg_level", avg_level)
                self.logger.record("custom/avg_floor_reached", avg_floor)
                self.logger.record("custom/max_floor_reached_interval", max_floor_interval)
                self.logger.record("custom/max_floor_reached_ever", self.max_floor_ever)
                self.logger.record("custom/avg_episode_length", avg_length)
                self.logger.record("custom/avg_enemies_defeated", avg_enemies)
                self.logger.record("custom/rate_invalid_actions", invalid_rate)
                
                self.logger.dump(step=self.num_timesteps)

                # Printa no console se verbose > 0
                if self.verbose > 0:
                    print(f"--- Intervalo Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count} (Timestep {self.num_timesteps}) ---")
                    print(f"  Avg Reward: {mean_reward:.2f}")
                    print(f"  Win Rate: {win_rate:.2f}")
                    print(f"  Avg Level: {avg_level:.2f}")
                    print(f"  Avg Floor Reached: {avg_floor:.2f}")
                    print(f"  Max Floor (Ever): {self.max_floor_ever}")
                    print(f"  Avg Enemies Defeated: {avg_enemies:.2f}")
                    print(f"  Invalid Action Rate: {invalid_rate:.3f}")
                    print("-" * (len(f"--- Intervalo Episódios {self.episode_count - self.log_interval + 1}-{self.episode_count} (Timestep {self.num_timesteps}) ---")))

                # Limpa as listas do intervalo
                self.episode_rewards.clear()
                self.episode_wins.clear()
                self.episode_levels.clear()
                self.episode_floors.clear()
                self.episode_lengths.clear()
                self.episode_enemies_defeated.clear()
                self.episode_invalid_actions.clear()
                self.episode_total_actions.clear()

        return True

    # --- ETAPA 4: FUNÇÃO PARA SALVAR (será chamada no final do treino) ---
    def save_hall_of_fame(self, save_dir: str):
        """Salva as melhores histórias em arquivos de texto."""
        if self.verbose > 0:
            print(f"\nSalvando Hall da Fama em {save_dir}...")
        
        # Helper interno para salvar uma lista específica
        def _save_list(hall_list: list, sub_folder: str, metric_key: str):
            path = os.path.join(save_dir, sub_folder)
            os.makedirs(path, exist_ok=True)
            
            for i, story in enumerate(hall_list):
                metric_val = story[metric_key]
                name = story['agent_name']
                # Define um nome de arquivo limpo (Rank_1__level_74__Agente_12345.txt)
                filename = f"Rank_{i+1:02d}__{metric_key}_{metric_val}__{name}.txt"
                
                try:
                    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
                        f.write(f"AGENTE: {name}\n")
                        f.write(f"MÉTRICA: {metric_key.upper()} = {metric_val}\n")
                        f.write(f"ANDAR FINAL: {story['floor']}\n")
                        f.write(f"NÍVEL FINAL: {story['level']}\n")
                        f.write(f"INIMIGOS DERROTADOS: {story['enemies_defeated']}\n")
                        f.write("="*50 + "\n\nHISTÓRIA DO AGENTE:\n" + "="*50 + "\n")
                        f.writelines(story['log_content'])
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  [Callback ERROR] Falha ao salvar história: {filename}. Erro: {e}")
        
        # Salva cada categoria
        _save_list(self.hall_of_fame_level, "top_por_nivel", "level")
        _save_list(self.hall_of_fame_floor, "top_por_andar", "floor")
        _save_list(self.hall_of_fame_enemies, "top_por_inimigos", "enemies_defeated")
        
        if self.verbose > 0:
            print("Hall da Fama salvo com sucesso.")

def main():
    """
    Script principal para treinar um agente PPO no ambiente BuriedBrains.
    """
    
    # --- 0. PARSER DE ARGUMENTOS ---
    parser = argparse.ArgumentParser(description="Script de Treinamento BuriedBrains")
    
    # Argumento para (não) usar LSTM
    parser.add_argument('--no_lstm', action='store_true', help="Usar PPO padrão em vez de RecurrentPPO (LSTM)")
    
    # Argumentos do Treino
    parser.add_argument('--total_timesteps', type=int, default=5_000_000, help="Total de passos de treino (model.learn)")
    parser.add_argument('--suffix', type=str, default="", help="Sufixo para o nome da run (ex: 'Baseline_NoLSTM')")

    # Argumentos do Ambiente
    parser.add_argument('--max_episode_steps', type=int, default=30000, help="Limite de passos por episódio no ambiente")
    parser.add_argument('--budget_multiplier', type=float, default=1.0, help="Multiplicador de dificuldade (budget) do ambiente")
    
    args = parser.parse_args()
    # --- FIM DO PARSER ---


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
    print("Verificando o ambiente BuriedBrainsEnv base...")
    temp_env = BuriedBrainsEnv()
    try:
        check_env(temp_env)
        print("Verificação do ambiente bem-sucedida!")
    except Exception as e:
        print(f"Erro na verificação do ambiente: {e}")
        temp_env.close()
        return
    temp_env.close()

    # --- 2. Criação do ambiente vetorizado ---
    print("Criando ambiente vetorizado para treinamento...")
    env = DummyVecEnv([lambda: BuriedBrainsEnv(
        verbose=1,
        max_episode_steps=args.max_episode_steps,
        budget_multiplier=args.budget_multiplier
    )])

    # --- 3. Configuração de logs e diretórios ---
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_logdir = os.path.normpath("logs").replace("\\", "/")
    base_models_dir = os.path.normpath("models").replace("\\", "/")

    os.makedirs(base_logdir, exist_ok=True)
    os.makedirs(base_models_dir, exist_ok=True)

    run_name = f"{model_class.__name__}_{policy_class}_{timestamp}"
    if args.suffix:
        run_name += f"_{args.suffix}"
    run_models_dir = os.path.join(base_models_dir, run_name)
    os.makedirs(run_models_dir, exist_ok=True)

    tb_path = os.path.join(base_logdir, run_name)
    os.makedirs(tb_path, exist_ok=True)

    # --- 4. Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=run_models_dir, name_prefix="bb_model")
    logging_callback = LoggingCallback(verbose=0, log_interval=2)
    callback_list = CallbackList([checkpoint_callback, logging_callback])

# Define os parâmetros otimizados para PPO (sem LSTM)
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
            "net_arch": {
                "pi": [256, 256], # Baseado no net_arch_size: 256
                "vf": [256, 256]
            }
        }
    }

    # Define os parâmetros otimizados para RecurrentPPO (com LSTM)
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
            "net_arch": {
                "pi": [64, 64], # Baseado no net_arch_size: 64
                "vf": [64, 64]
            },
            "lstm_hidden_size": 128,
            "enable_critic_lstm": False
        }
    }

    # Seleciona os parâmetros com base no modelo
    if use_lstm:
        print("Carregando modelo RecurrentPPO com hiperparâmetros otimizados (LSTM)...")
        model_kwargs = lstm_params
    else:
        print("Carregando modelo PPO com hiperparâmetros otimizados (MLP)...")
        model_kwargs = ppo_params

    # Cria o modelo usando desempacotamento de dicionário (**)
    model = model_class(
        policy_class,
        env,
        verbose=0,
        tensorboard_log=base_logdir, 
        device=device,
        **model_kwargs  # Aplica todos os parâmetros otimizados
    )

    TIMESTEPS = args.total_timesteps
    print(f"Iniciando treinamento por {TIMESTEPS} passos...")
    print(f"Logs do TensorBoard serão salvos em: {tb_path}")

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback_list,
        tb_log_name=run_name,  
        progress_bar=True,
    )

    # --- 6. Salvamento do modelo final ---
    model.save(os.path.join(run_models_dir, f"final_model_{TIMESTEPS}"))
    
    # --- 7. Salvar o Hall da Fama ---
    print("Salvando histórias do Hall da Fama...")
    # Acessa o callback de logging (índice [1] da lista, pois o [0] é o CheckpointCallback)
    if len(callback_list.callbacks) > 1 and isinstance(callback_list.callbacks[1], LoggingCallback):
        logging_callback_instance = callback_list.callbacks[1]
        
        # Chama a função de salvar que criamos dentro do callback
        logging_callback_instance.save_hall_of_fame(tb_path)
        
        print(f"Histórias dos melhores agentes salvas em: {tb_path}")
    else:
        print("[WARN] LoggingCallback não encontrado. Histórias não foram salvas.")    

    print(f"Treinamento concluído. Modelo final salvo em {tb_path}")

    env.close()

if __name__ == "__main__":
    main()
 