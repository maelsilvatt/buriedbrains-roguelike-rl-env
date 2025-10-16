# test_env.py
import gymnasium as gym
import random

# Importa a sua classe de ambiente principal do pacote que criamos
from buriedbrains.env import BuriedBrainsEnv

def main():
    """
    Script de teste para validar a funcionalidade básica do ambiente BuriedBrainsEnv.
    Ele executa vários episódios com um agente que toma ações aleatórias.
    O objetivo é garantir que os métodos reset() e step() funcionam sem erros.
    """
    print("Criando o ambiente BuriedBrainsEnv...")
    env = BuriedBrainsEnv()

    NUM_EPISODES = 10
    MAX_STEPS_PER_EPISODE = 200

    print(f"Iniciando teste com {NUM_EPISODES} episódios...")

    for episode in range(NUM_EPISODES):
        print("\n" + "="*30)
        print(f"--- Iniciando Episódio {episode + 1} ---")
        print("="*30)

        # 1. Reseta o ambiente para começar um novo episódio
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # 2. Loop principal do episódio
        while not (terminated or truncated) and step_count < MAX_STEPS_PER_EPISODE:
            step_count += 1
            
            # 3. Escolhe uma ação aleatória do espaço de ações
            action = env.action_space.sample()

            # 4. Executa a ação no ambiente
            observation, reward, terminated, truncated, info = env.step(action)

            # 5. Imprime o resultado do passo
            print(
                f"Passo: {step_count:3} | "
                f"Ação: {action} | "
                f"Recompensa: {reward:6.2f} | "
                f"Terminado: {terminated} | "
                f"Info: {info}"
            )
            
            total_reward += reward

        print("-" * 30)
        print(f"Episódio {episode + 1} finalizado após {step_count} passos.")
        print(f"Recompensa Total do Episódio: {total_reward:.2f}")

    # 6. Fecha o ambiente
    env.close()
    print("\nTeste concluído com sucesso!")

if __name__ == "__main__":
    main()