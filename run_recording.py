import argparse
from sb3_contrib import RecurrentPPO
from buriedbrains.env import BuriedBrainsEnv
from buriedbrains.wrappers import SharedPolicyVecEnv
from visualizer.recorder import BuriedBrainsRecorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Caminho do .zip treinado")
    parser.add_argument('--output', type=str, default="replay_final.json")
    args = parser.parse_args()

    # 1. Recria o Ambiente (Mesmas configs do treino)
    base_env = BuriedBrainsEnv(
        max_episode_steps=2000, # Um episódio longo o suficiente
        sanctum_floor=20,
        verbose=0,
        num_agents=2, # Ou mais, dependendo do que quer mostrar
        seed=123 # Seed fixa para garantir que o replay seja 'sinistro'
    )
    env = SharedPolicyVecEnv(base_env)

    # 2. Carrega o Modelo
    print(f"Carregando modelo: {args.model_path}")
    model = RecurrentPPO.load(args.model_path, env=env)

    # 3. Inicializa Gravador
    recorder = BuriedBrainsRecorder(env)

    # 4. Loop de Execução
    obs = env.reset()
    lstm_states = None
    episode_starts = [True] * env.num_agents
    
    # Grava o Frame 0 (Inicial)
    recorder.record_step(0)

    print("Gravando episódio...")
    for step in range(2000):
        # Ação do Modelo
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True # Importante para replay consistente
        )
        
        # Passo no ambiente
        obs, rewards, dones, infos = env.step(action)
        episode_starts = dones
        
        # Grava o Frame T
        recorder.record_step(step + 1)
        
        # Se todos morrerem/acabarem, encerra
        if all(dones):
            print("Episódio finalizado.")
            break

    # 5. Salva o Arquivo
    recorder.save_to_json(args.output)

if __name__ == "__main__":
    main()