# buriedbrains/agent_rules.py
from typing import Dict, Any

def create_initial_agent(name: str = "Player") -> Dict[str, Any]:
    """
    Cria e retorna o dicionário de estado para um agente iniciante.
   
    """
    agent = {
        'name': name,
        'level': 1,
        'hp': 30,
        'max_hp': 30,
        'base_stats': {
            'flat_damage_bonus': 10,
            'damage_reduction': 0.0
        },
        'exp': 0,
        'exp_to_level_up': 200,
        'equipment': {}, # Slots para equipamentos
        'skills': [],    # Lista de habilidades conhecidas
    }
    return agent

def check_for_level_up(agent: Dict[str, Any]) -> bool:
    """
    Verifica se o agente tem experiência suficiente para subir de nível.
    Se sim, atualiza os atributos do agente diretamente e retorna True.
    Caso contrário, retorna False.
   
    """
    leveled_up = False
    # Usar um loop 'while' permite múltiplos level-ups de uma vez se o ganho de XP for grande
    while agent.get('exp', 0) >= agent.get('exp_to_level_up', float('inf')):
        leveled_up = True
        
        # Consome o XP necessário e mantém o excedente
        agent['exp'] -= agent['exp_to_level_up']
        agent['level'] += 1
        
        # Aumenta a dificuldade para o próximo nível (ex: +20%)
        agent['exp_to_level_up'] = int(agent['exp_to_level_up'] * 1.2)
                
        # Aumenta os atributos base do agente permanentemente
        agent['max_hp'] += 10
        agent['hp'] = agent['max_hp']  # Cura totalmente o agente ao subir de nível
        agent['base_stats']['flat_damage_bonus'] += 1
        agent['base_stats']['damage_reduction'] += 0.008

        # Poderíamos adicionar um log aqui para a classe do ambiente capturar
        # print(f"O agente {agent['name']} subiu para o nível {agent['level']}!")

    return leveled_up