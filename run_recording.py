import argparse
import os
import sys
from sb3_contrib import RecurrentPPO
from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.wrappers import SharedPolicyVecEnv
from visualizer.utils.recorder import BuriedBrainsRecorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output', type=str, default="records/replay.json")
    parser.add_argument('--recording_duration', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--sanctum_floor', type=int, default=20)
    args = parser.parse_args()

    # Tratamento da extensão do arquivo
    if not args.output.endswith('.json'):
        args.output += '.json'
    
    # Cria diretório se não existir
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Configuração ---")
    print(f"Modelo: {args.model_path}")
    print(f"Seed: {args.seed} | Agentes: {args.num_agents}")
    print(f"--------------------")

    # 1. Ambiente
    base_env = BuriedBrainsEnv(
        max_episode_steps=args.recording_duration, 
        sanctum_floor=args.sanctum_floor,
        verbose=0,
        num_agents=args.num_agents, 
        seed=args.seed 
    )
    env = SharedPolicyVecEnv(base_env)

    # 2. Carrega Modelo
    print("Carregando modelo...")
    try:
        # Carrega na CPU para evitar erro de VRAM se estiver treinando outra coisa
        model = RecurrentPPO.load(args.model_path, env=env, device='cpu') 
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo.\n{e}")
        return

    # 3. Recorder
    recorder = BuriedBrainsRecorder(env)

    # 4. Loop
    obs = env.reset()
    lstm_states = None
    episode_starts = [True] * env.num_agents
    
    recorder.record_step(0) # Frame Inicial

    print(f"Gravando {args.recording_duration} passos...")
    for step in range(args.recording_duration):
        # Ação Determinística = True para replay fiel
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True 
        )
                
        # Nota: O step do SharedPolicyVecEnv faz isso internamente, 
        # mas precisamos dos dados pós-step.
        
        obs, rewards, dones, infos = env.step(action)
        episode_starts = dones
        
        # Convertemos a ação do SB3 (numpy array) para um dict 
        # que o recorder possa entender melhor, ou deixamos ele pegar do ENV.
        # No script recorder.py, ele pega `actions_dict`.
        actions_dict = {
            env.agent_ids[i]: action[i] for i in range(len(env.agent_ids))
        }

        recorder.record_step(step + 1, actions_dict=actions_dict)
        
        if all(dones):
            print(f"Todos os agentes terminaram no passo {step}.")
            break

    recorder.save_to_json(args.output)

if __name__ == "__main__":
    main()