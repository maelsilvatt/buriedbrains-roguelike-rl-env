# test_env.py
import gymnasium as gym
import random

# Importa a sua classe de ambiente principal
from buriedbrains.env import BuriedBrainsEnv

def test_combat_encounter(env: BuriedBrainsEnv):
    """
    Teste focado em forçar e validar um único encontro de combate.
    """
    print("\n" + "="*40)
    print("--- INICIANDO TESTE DE ENCONTRO DE COMBATE ---")
    print("="*40)

    observation, info = env.reset()
    
    # 1. Procura por uma sala inicial que tenha um inimigo
    action_to_start_combat = -1
    successors = list(env.graph.successors(env.current_node))
    
    for i, node in enumerate(successors):
        content = env.graph.nodes[node].get('content', {})
        if content.get('enemies'):
            print(f"Inimigo encontrado na sala adjacente {i}. Forçando movimento...")
            action_to_start_combat = 5 + i # Ações de movimento são 5 e 6
            break
            
    if action_to_start_combat == -1:
        print("Nenhum inimigo encontrado nas salas iniciais. O combate não será testado neste episódio.")
        return

    # 2. Move o agente para a sala com o inimigo para iniciar o combate
    observation, reward, terminated, truncated, info = env.step(action_to_start_combat)
    
    if not env.combat_state:
        print("ERRO: O agente se moveu para uma sala com inimigo, mas o combate não iniciou.")
        return
        
    print("\n--- COMBATE INICIADO ---")
    step_count = 0
    while not (terminated or truncated):
        step_count += 1
        
        # 3. Em combate, escolhe uma ação de combate aleatória (0 a 4)
        combat_action = random.randint(0, 4)
        
        observation, reward, terminated, truncated, info = env.step(combat_action)
        
        print(
            f"Turno de Combate: {step_count:2} | "
            f"Ação: {combat_action} | "
            f"Recompensa: {reward:6.2f} | "
            f"HP Agente: {env.agent_state['hp']:.0f} | "            
            f"HP Inimigo: {(env.combat_state['enemy']['hp'] if env.combat_state else 0):.0f} | "
            f"Terminado: {terminated}"
        )
        
        if env.combat_state is None:
            print("--- COMBATE FINALIZADO ---")
            break
            
def test_random_exploration(env: BuriedBrainsEnv, num_episodes: int, max_steps: int):
    """
    Teste original de exploração aleatória, agora com mais informações.
    """
    print("\n" + "="*40)
    print("--- INICIANDO TESTE DE EXPLORAÇÃO ALEATÓRIA ---")
    print("="*40)
    
    for episode in range(num_episodes):
        print(f"\n--- Iniciando Episódio de Exploração {episode + 1} ---")
        observation, info = env.reset()
        terminated, truncated = False, False
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps:
            step_count += 1
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # >> ADICIONADO: Mostra o conteúdo da sala atual <<
            current_content = env.graph.nodes[env.current_node].get('content', {})
            enemy_info = f"Inimigos: {current_content.get('enemies', [])}"
            
            print(
                f"Passo: {step_count:3} | "
                f"Ação: {action} | "
                f"Recompensa: {reward:6.2f} | "
                f"Info da Sala: {enemy_info}"
            )
        print(f"Episódio de Exploração {episode + 1} finalizado.")

def main():
    """
    Script de teste principal.
    """
    print("Criando o ambiente BuriedBrainsEnv...")
    env = BuriedBrainsEnv()

    # Executa o teste de combate primeiro para garantir a validação
    test_combat_encounter(env)
    
    # Em seguida, executa a exploração aleatória
    test_random_exploration(env, num_episodes=3, max_steps=20)

    env.close()
    print("\nTeste concluído!")

if __name__ == "__main__":
    main()